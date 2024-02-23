import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from data import *
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
from torch import optim
from infomec import *
from blocks import *
from fast_data import *
from aq_blocks import *

dataset_name = 'isaac'

s_config = {
    "groups" : 6,
    "img_width" : 64,
    "latent_sizes" :[10,10,10,8,4,15]}
i_config = {
    "groups" : 9,
    "img_width" : 128,
    "latent_sizes" : [3,8,5,4,4,4,6,4,4] }
f_config = {
    "groups" : 7,
    "img_width" : 128,
    "latent_sizes" : [5,6,6,6,6,6,6] }

if dataset_name == 'falcor':
    cfg = f_config
    ds = falcor3d_data()
if dataset_name == 'shapes3d':
    cfg = s_config
    ds = shapes3d_dataset()
    
if dataset_name == 'isaac':
    cfg = i_config
    ds = isaac3d_data()
# ds = isaac3d_data()
# ds = shapes3d_dataset()

tset = torch.from_numpy(ds.images[ds.test]).permute(0,3,1,2)/255.
tlabels = ds.labels[ds.test]
def eval(model):
    model.eval
    #ev_data = test_set[:250].to(device)
    ev_data = tset[:250].to(device)

    with torch.no_grad():
        outs = model(ev_data.to(device))
        #print(outs.keys())
        latent = torch.cat(outs['elist'],1) + torch.tensor(ds.cumulate,device = device)
        #print(latent.shape)
        for i in range(1,20):
            outs = model(tset[i*250:(i+1)*250].to(device))
            z = torch.cat(outs['elist'],1) + torch.tensor(ds.cumulate,device = device)
            latent = torch.cat([latent,z],0)
    #print(latent.shape)
    
    res = compute_infomec(tlabels.cpu().numpy(), latent.float().detach().cpu().numpy(), True)
    return res

class AE(nn.Module):
    def __init__(self, num_hiddens, embedding_dim, commitment_cost,cfg,dist_type = 'cosine'):
        super(AE, self).__init__()
        self.encoder = attribute_encoder(3,cfg = cfg)
        self.decoder = Decoder(cfg['groups'],num_hiddens = cfg['groups']*40,cfg = cfg)
        self.embedding_dim = embedding_dim
        self.dist_type = dist_type
        self.cfg = cfg
        self.lsize = len(self.cfg['latent_sizes'])
        self.latent_sizes = self.cfg['latent_sizes']
        self.vq_list = []
        for i in self.latent_sizes:
            self.vq_list.append(AttributeQuantizer(i,64,0.25))

        self.vq_list = nn.ModuleList(self.vq_list)
    def forward(self,x,labels = None):
        out = self.encoder(x)
        #print(latent.shape)
        #return latent,self.decoder(latent)
        q = torch.zeros(out['pre_q'].shape).to(device)
        #print(self.lsize)
        plist = [None]*self.lsize
        elist = [None]*self.lsize
        llist= [None]*self.lsize
        if labels !=None:
            for i,layer in enumerate(self.vq_list):
                llist[i],q[:,i,:],plist[i],_,elist[i] = layer(out['pre_q'][:,i,:],labels[:,i])
            #out.update(layer(out["pre_q"]))
        else:
            for i,layer in enumerate(self.vq_list):
                #print(i)
                #print(out['pre_q'].shape)
                llist[i],q[:,i,:],plist[i],_,elist[i] = layer(out['pre_q'][:,i,:],labels)
        out.update({"llist":llist,"q":q,"plist":plist,"elist":elist})

        #print(q)
        out.update(self.decoder(q))
        return out#recon,llist,plist,elist
        # return recon,[loss0,loss1,loss2,loss3,loss4,loss5],[perplexity0,perplexity1,perplexity2,perplexity3,perplexity4,perplexity5],[encoding_indices0,encoding_indices1,encoding_indices2,encoding_indices3,encoding_indices4,encoding_indices5]
    def build_optimizer(self,lr = 0.001):
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.3)
        for name, param in self.named_parameters():
            if "bias" in name:
                param.weight_decay = 0
            if "embedding" in name:
                param.weight_decay = 0
        
    def train_step(self,data,labels):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(data,labels)
        recon_loss = torch.sqrt(F.mse_loss(out['recon'],data))
        commitment_loss = sum(out['llist'])
        loss = recon_loss + commitment_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {"commitment_loss":commitment_loss.item(),"recon":recon_loss.item(),"loss":loss.item()}

groups = cfg['groups']

for j in range(5):
    
    ae = AE(120,64,0.25,cfg).to(device)
    ae.build_optimizer(0.001)
    
    import tqdm
    eval_list = []
    for i in tqdm.tqdm(range(15000),ncols = 100):
        a,b,c = ds.get_sup_batch(64)
        out = ae.train_step(a.to(device),b.to(device).long())
        if i%500 == 0:
            out.update(eval(ae))
            ae.train()
            print(out)
            eval_list.append(out)
        if i==10000:
            ae.build_optimizer(0.0001)
    
    #folder_path = "new_results/dvq/falcor3d"
    folder_path = f"new_results/dvq/{dataset_name}/trial{j}/"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created successfully!")
    torch.save(ae.state_dict(),folder_path+'model_sup.pt')
    
    df = pd.DataFrame(eval_list)
    df.to_csv(folder_path + 'eval_sup.csv', index=True)
    
    ae.build_optimizer(0.0001)
    eval_list = []
    for i in tqdm.tqdm(range(1000),ncols = 100):
        a,b,c = ds.get_unsup_batch(64)
        out = ae.train_step(a.to(device),None)
        if i%50 == 0:
            out.update(eval(ae))
            ae.train()
            print(out)
            eval_list.append(out)
    
    torch.save(ae.state_dict(),folder_path+'model_unsup.pt')
    
    df = pd.DataFrame(eval_list)
    df.to_csv(folder_path + 'eval_unsup.csv', index=True)

