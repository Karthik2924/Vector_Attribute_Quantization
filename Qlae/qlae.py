import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import yaml
import inspect
from blocks import *
from infomec import*
from fast_data import*
from qlae_helper import*
import h5py
import os
import tqdm
import pandas as pd
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
initialize(version_base=None, config_path="configs/")
config=compose(config_name="train_ae.yaml")

with open("configs/encoder_partial/image.yaml", "r") as file:
    encoder_partial = yaml.safe_load(file)

with open("configs/decoder_partial/style_image.yaml", "r") as file:
    style_decoder_partial = yaml.safe_load(file)

with open("configs/model_partial/quantized_ae.yaml", "r") as file:
    latent_partial = yaml.safe_load(file)


econv = encoder_partial['conv_partial']
edense = encoder_partial['dense_partial']
latent = latent_partial['latent_partial']
ddense = style_decoder_partial['dense_partial']
ddconv = style_decoder_partial['style_conv_transpose_partial']
dconv = style_decoder_partial['conv_partial']

# dataset = falcor3d_data()
# dataset_name = "falcor3d"
# lsize = 14

dataset = isaac3d_data()
lsize = 18
dataset_name = "isaac3d"

device = 'cuda'

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(econv,edense)
        self.latent = QuantizedLatent(num_latents=lsize, num_values_per_latent = 12, optimize_values=True)
        self.decoder = StyleImageDecoder(ddense,ddconv,dconv,256,(1,3,64,64))
        self.lambdas = {'binary_cross_entropy': 1.0, 'quantization': 0.01, 'commitment': 0.01, 'l2': 0.1, 'l1': 0.0}
    
    def forward(self, x):
        outs = self.encoder(x)
        outs.update(**self.latent(outs['pre_z']))
        outs.update(**self.decoder(outs['z_hat']))
        return outs
    
    @staticmethod
    def quantization_loss(continuous, quantized):
        return torch.mean(torch.square(continuous.detach() - quantized))

    @staticmethod
    def commitment_loss(continuous, quantized):
        return torch.mean(torch.square(continuous - quantized.detach()))
    
    def batched_loss(self, data, *args, key=None, **kwargs):
        #data['x'] = torch.from_numpy(data['x']).to(device)
        data['x']= data['x'].to(device)
        outs = self.forward(data['x'])
        #outs = model(data['x'])
        quantization_loss = self.quantization_loss(outs['z_continuous'], outs['z_quantized'])
        commitment_loss = self.commitment_loss(outs['z_continuous'], outs['z_quantized'])
        binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(outs['x_hat_logits'], data['x'].float(), reduction='mean')
        
        loss = (
            self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss +
            self.lambdas['quantization'] * quantization_loss +
            self.lambdas['commitment'] * commitment_loss
        )
        metrics = {
            'loss': loss.item(),
            'binary_cross_entropy_loss': binary_cross_entropy_loss.item(),
            'quantization_loss': quantization_loss.item(),
            'commitment_loss': commitment_loss.item(),}

        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return loss, aux

    def param_labels_group(self):
        param_labels = {}
        unreg = []
        reg = []
        for name, param in self.named_parameters():
            if 'bias' in name:
                param_labels[name] = 'unregularized'
                unreg.append(param)
            elif 'latent' in name:
                param_labels[name] = 'unregularized'
                unreg.append(param)
            else:
                param_labels[name] = 'regularized'
                reg.append(param)

        return unreg,reg

    def construct_optimizer(self):
        # weight_decay = config.model_partial.lambdas.get('l2', 0.0)
        weight_decay = 0.1
        
        unregularized_params, regularized_params = self.param_labels_group()
        optimizer = torch.optim.AdamW(
            [
                {'params': regularized_params, 'weight_decay': weight_decay},
                {'params': unregularized_params}
            ],
            lr=config.optim.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False
        )
        return optimizer

def eval(model):
    model.eval
    #ev_data = test_set[:250].to(device)
    ev_data = dataset.get_test_batch(250)

    with torch.no_grad():
        outs = model(ev_data['x'].to(device))
        tlabel = ev_data['z']
        latent = outs['z_hat']
        #latent = torch.cat(ei_list,1) + torch.tensor([0,5,11,17,23,29,35],device = device)
        #print(latent.shape)
        for i in range(1,20):
            ev_data = dataset.get_test_batch(250)
            outs = model(ev_data['x'].to(device))
            z = outs['z_hat']
            #z = torch.cat(ei_list,1) + torch.tensor([0,5,11,17,23,29,35],device = device)
            latent = torch.cat([latent,z],0)
            tlabel = np.concatenate([tlabel,ev_data['z']])
    #print(latent.shape)
    res = compute_infomec(tlabel, latent.float().detach().cpu().numpy(), False)
    return res

batch = dataset.get_train_batch(128)
for j in range(3):
    model = Model().to(device)
    optimizer = model.construct_optimizer()
    eval_results = []
    model.train()
    for i in tqdm.tqdm(range(100000),ncols = 100):
    #for i in tqdm.tqdm(range(100),ncols = 100):

        batch = dataset.get_train_batch(128)
        loss,aux = model.batched_loss(batch)
    
        model.zero_grad()
    
        if i%2000 == 0:
        #if i%50 ==0:
            store = {}
            print(aux['metrics'])
            store.update(aux['metrics'])
            res = eval(model)
            store.update(res)
            model.train()
            eval_results.append(store)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    
    folder_path = f"results/{dataset_name}/trial{j}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created successfully!")
    df = pd.DataFrame(eval_results)
    torch.save(model.state_dict(),folder_path + '/model.pt')
    df.to_csv(folder_path+ '/qlae_eval.csv', index=True)  







