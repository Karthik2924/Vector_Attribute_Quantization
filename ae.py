import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data import *
device  = 'cuda'
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.express as px
import tqdm
import seaborn as sns
from torchvision.utils import make_grid
import os
import json
from infomec import *
from blocks import *
from accelerate import Accelerator
accelerator = Accelerator()
import tqdm
seed = 1234
gen = torch.manual_seed(seed)
from fast_data import*
method = "ae"
data_set = "shapes3d"
#data_set = "falcor3d"
dp = shapes3d_dataset()
#dp = isaac3d_data('isaac')
#dp = falcor3d_data('falcor')
# data_set = "dsprites"

# # ds = Shapes3d()
# # perm = torch.randperm(480000,generator = gen)
# # train_set = ds.images[perm[:460000]]
# # test_set = ds.images[perm[460000:]]
# # test_labels = ds.labels[perm[460000:]]
# from disentanglement_datasets import DSprites
# dataset = DSprites(root="./data", download=True)
# print("dataset = dsprites")
# data_size = dataset[:]['input'].shape
# arrangement = torch.randperm(data_size[0],generator = gen)
# train_ind = arrangement[:-20000]
# test_ind = arrangement[-20000:]
# train_set = dataset[train_ind]['input'].float().unsqueeze(1)
# test_set = dataset[test_ind]['input'].float().unsqueeze(1)
# train_labels = dataset[train_ind]['latent'][:,1:]
# test_labels = dataset[test_ind]['latent'][:,1:]

# dataset = isaac3d()
# dataset = falcor3d()
# generator1 = torch.Generator().manual_seed(1234)
# s_set,t_set = torch.utils.data.random_split(dataset, [0.95,0.05], generator=generator1)
# train_loader = DataLoader(s_set, batch_size=64, shuffle=True, sampler=None,
#            batch_sampler=None, num_workers=4, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None, prefetch_factor=2,
#            persistent_workers=False)
# # utrain_loader = DataLoader(u_set, batch_size=64, shuffle=True, sampler=None,
# #            batch_sampler=None, num_workers=8, collate_fn=None,
# #            pin_memory=True, drop_last=False, timeout=0,
# #            worker_init_fn=None, prefetch_factor=2,
# #            persistent_workers=False)
# test_loader = DataLoader(t_set, batch_size=250, shuffle=True, sampler=None,
#            batch_sampler=None, num_workers=8, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None, prefetch_factor=2,
#            persistent_workers=False)


test_set = dp.images[dp.test_ind[:5000]]
test_labels = dp.labels[dp.test_ind[:5000]]
def transform(x):
    return torch.from_numpy(x).permute(0,3,1,2)/255.
img_size = 128

class AE(nn.Module):
    def __init__(self, in_channels, latent_dim,):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        #self.encoder = nn.Sequential(conv_block(3,16),conv_block(16,32),conv_block(32,64),conv_block(64,128))
        self.encoder = nn.Sequential(conv_block(in_channels,8),conv_block(8,16),conv_block(16,32),conv_block(32,48),conv_block(48,60))
        self.l1 = nn.Linear(240,latent_dim)
        self.l2 = nn.Linear(latent_dim,240)
        self.decoder = nn.Sequential(deconv_block(60,48),deconv_block(48,32),deconv_block(32,16),deconv_block(16,8),deconv_block(8,in_channels))
        # self.encoder = nn.Sequential(conv_block(in_channels,8),conv_block(8,16),
        #                              conv_block(16,32),conv_block(32,48),
        #                              conv_block(48,60),conv_block(60,120))
        # self.l1 = nn.Linear(480,latent_dim)
        # self.l2 = nn.Linear(latent_dim,480)
        # self.decoder = nn.Sequential(deconv_block(120,60),deconv_block(60,48),
        #                              deconv_block(48,32),deconv_block(32,16),
        #                              deconv_block(16,8),deconv_block(8,in_channels))

    def forward(self, x):
        enc = self.encoder(x)
        #print(enc.shape)
        #z = self.l1(enc.reshape(-1,480))
        z = self.l1(enc.reshape(-1,240))

        re = self.l2(z)
        re = self.decoder(re.reshape(enc.shape))
        return re
    def get_enc(self,x):
        enc = self.encoder(x)
        z = self.l1(enc.reshape(enc.shape[0],-1))
        return z

# def eval(model):
#     model.eval()
#     #ev_data = test_set[:250].to(device)
#     ev_data,test_labels,_ = next(iter(test_loader))
#     with torch.no_grad():
#         latent = model.get_enc(ev_data.to(device))
#         #print(latent.shape)
#         for i in range(1,20):
#             data,tlabels,_ = next(iter(test_loader))
#             z = model.get_enc(data.to(device))
#             test_labels = torch.cat([test_labels,tlabels])
#             #z = model.get_enc(test_set[i*250:(i+1)*250].to(device))
#             latent = torch.cat([latent,z],0)
#     #print(latent.shape)
    
#     res = compute_infomec(test_labels.cpu().numpy(), latent.detach().cpu().numpy(), False)
#     return res

def eval(model):
    model.eval()
    ev_data = transform(test_set[:250]).to(device)
    with torch.no_grad():
        latent = model.get_enc(ev_data.to(device))
        #print(latent.shape)
        for i in range(1,20):
            z = model.get_enc(transform(test_set[i*250:(i+1)*250]).to(device))

            latent = torch.cat([latent,z],0)
    
    res = compute_infomec(test_labels.cpu().numpy(), latent.detach().cpu().numpy(), False)
    return res

def save(seed,lsize):
    folder_path = f"new_results/{method}/{data_set}/seed_{seed}/latent_size_{lsize}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created successfully!")
    else:
        print("Folder already exists.")
    
    torch.save(ae.state_dict(),folder_path + "/model.pt")

    with open(folder_path + "/config.json", "w") as outfile: 
    	json.dump(config, outfile)
    df.to_csv(folder_path+'/logs.csv', index=False)

    torch.save(ae.state_dict(),folder_path + "/model.pt")
    fig = px.line(data_frame=df, x=df.index*5000, y=['infoe','infom','infoc','tr_loss','val_loss'])
    fig.update_layout(
        title="AE Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss/Metric",
        width = 500,
        height = 500,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="Black"
        )
    
    )
    fig.write_image(folder_path+'/plot.png')


#seed_list = [123,1234,12345,123456,1234567]
seed_list = [1234,12345,123456,1234567]
#seed_list = [1234567]

for s in seed_list:

    b_seed = s
    os.environ['PYTHONHASHSEED'] = str(b_seed)
    # Torch RNG
    torch.manual_seed(b_seed)
    torch.cuda.manual_seed(b_seed)
    torch.cuda.manual_seed_all(b_seed)

# latent_sizes = [6,12,18,36,60]

#latent_sizes = [5,15,25,40,60]
    #latent_sizes = [9,18,27,45,72,100] #isaac3d dataste
    #latent_sizes = [7,14,21,35,56,100] #falcor dataste
    latent_sizes = [12,18]
    
    
    for lsize in latent_sizes:
        if data_set == "dsprites":
            ae = AE(1,lsize).to(device)
        else:
            ae = AE(3,lsize).to(device)
    
        lr = 1e-4
        optimizer = torch.optim.Adam(ae.parameters(),lr=lr,amsgrad = False)
        ae, optimizer = accelerator.prepare(ae, optimizer)
        step = -1
        logs = {
            'tr_loss':[],
            'val_loss':[],
            'infoe':[],
            'infom':[],
            'infoc':[],
            'nmi':[]
        }
        for i in tqdm.tqdm(range(100000),ncols = 100):
        #for i in tqdm.tqdm(range(2),ncols = 5):
    
            #batch = next(iter(validation_loader))
            step+=1
            #perm = torch.randperm(size)[:128]
            #perm = torch.randint(0,size,(256,))
            # ind = torch.randint(0,460000,(128,))
            # data = train_set[ind].to(device)
            data,label,raw_label = dp.get_train_batch()

            #data,label,one_hot_label = next(iter(train_loader))
            data = data.to(device)
            recon = ae(data)
            optimizer.zero_grad()
        
            loss = F.mse_loss(recon,data)
            #loss = recon_loss + vq_loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(ae.parameters(), 1.0)
    
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
        
            optimizer.step()
    
            #if i%1 == 0:
            if i%5000 == 0:
                
                print(f"loss = {loss.item()},step = {i}")
                #ind = torch.randint(0,20000,(128,))
                #ev_data = test_set[ind].to(device)
                
                ev_data,_,_ = dp.get_test_batch()
                #ev_data,_,_ = next(iter(test_loader))
                ev_data = ev_data.to(device)
                ae.eval()
                with torch.no_grad():
                    re = ae(ev_data)
                    ev_loss = F.mse_loss(re,ev_data)
                    print(f"val_loss = {ev_loss.item()}")
                    res = eval(ae)
                # logs[f"step{step}"] = {"tr_loss":loss.item(),
                #                         "eval_loss":ev_loss.item(),
                #                       "infomec":res} 
                logs['tr_loss'].append(loss.item())
                logs['val_loss'].append(ev_loss.item())
                logs['infoe'].append(res['infoe'])
                logs['infom'].append(res['infom'])
                logs['infoc'].append(res['infoc'])
                logs['nmi'].append(res['nmi'])
                ae.train()
        
        df = pd.DataFrame(logs)
        #print(df.head())
        config  = {
        "name" : f'{data_set}vanilla_ae_latent_size_{lsize}',
        "latent_dim" : lsize,
        "clip_grad_norm": 1.0, #(none if no clipping)
        "learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
        "seed": s
            
        }
        save(config['seed'],config['latent_dim'])

        
        
