import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import*






def l2norm(t):
    return F.normalize(t, p = 2, dim = 1)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens,cfg):
        super(Decoder, self).__init__()
        #print(num_hiddens//2)
        groups = cfg['groups']
        self.img_width = cfg['img_width']
        self.gn1 = nn.GroupNorm(groups, groups)
        # print(in_channels,num_hiddens,groups)

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1,groups = groups)
        self.gn2 = nn.GroupNorm(groups, num_hiddens)
        self.dconv0 = deconv_block(num_hiddens,num_hiddens//4)
        self.dconv1 = deconv_block(num_hiddens//4,num_hiddens//4)
        self.dconv2 = deconv_block(num_hiddens//4,num_hiddens//8)
        self.dconv3 = deconv_block(num_hiddens//8,num_hiddens//16)
        self.output = nn.Conv2d(in_channels = num_hiddens//16,
                                out_channels = 3,kernel_size = 1,stride = 1)
        
        
    def forward(self, inputs):
        inputs = self.gn1(inputs)
        x = F.gelu(self._conv_1(inputs))
        x = self.gn2(x)
        x = self.dconv0(x)
        if self.img_width == 128:
            x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        return {"recon":self.output(x)}

class AttributeQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost,dist_type = 'cosine'):
        super(AttributeQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        nn.init.orthogonal_(self._embedding.weight)
        self._embedding.weight.requires_grad = False
        self._commitment_cost = 1
        self.qloss = 0.000
        self.dist_type = dist_type
    def compute_distance(self,inp,labels = None):
        if self.dist_type == "l2":
            distances= (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight.data**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.data.t()))
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inp.device)
            encodings.scatter_(1, encoding_indices, 1)
            return distances,encoding_indices,encodings
            
        if self.dist_type == 'cosine':
            
            distances = l2norm(inp)@l2norm(self._embedding.weight.data).T
            if labels!=None: 
                encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
                encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inp.device)
                encodings.scatter_(1, encoding_indices, 1)
            else:
                encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
                encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inp.device)
                encodings.scatter_(1, encoding_indices, 1)
                
            return distances,encoding_indices,encodings
        
    def forward(self, inputs,labels):

        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances,encoding_indices,encodings = self.compute_distance(flat_input,labels)

        if labels != None:

            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

            closs =  (torch.ones(distances.shape[0],device = flat_input.device) - torch.gather(distances,1,labels.unsqueeze(1)).squeeze(1)).mean()

            loss = closs
            quantized = inputs + (quantized - inputs).detach()
      
            perplexity = torch.tensor(1)
            
            # convert quantized from BHWC -> BCHW
            return loss,quantized,perplexity,encodings,encoding_indices
            return {"closs":loss, "quantized":quantized, "perplexity":perplexity,"encodings":encodings,"encoding_indices":encoding_indices}#, encodings,encoding_indices#,None

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs,reduction = 'mean')
        q_latent_loss = F.mse_loss(quantized, inputs.detach(),reduction = 'mean')
        #loss = q_latent_loss + self._commitment_cost * e_latent_loss
        #loss = e_latent_loss*self._commitment_cost + q_latent_loss*self.qloss
        closs =  (torch.ones(distances.shape[0],device = flat_input.device) - torch.gather(distances,1,encoding_indices).squeeze(1)).mean()
        loss = closs

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss,quantized,perplexity,encodings,encoding_indices

        # convert quantized from BHWC -> BCHW
        return {"closs":loss, "quantized":quantized, "perplexity":perplexity,"encodings":encodings,"encoding_indices":encoding_indices}

class attribute_encoder(nn.Module):
    def __init__(self,in_channels,cfg):
        super(attribute_encoder,self).__init__()
        self.img_width = cfg['img_width']
        groups = cfg['groups']
        self.conv1 = conv_block(in_channels = in_channels,
                               out_channels = 20)
        self.conv2 = conv_block(in_channels = 20,
                                out_channels = groups*10)
        self.conv3 = conv_block(in_channels = groups*10,out_channels = groups*30)
        self.conv4 = conv_block(in_channels = groups*30,out_channels = groups*30)
        #self.res_stack = ResidualStack(120,120,num_residual_layers,120)
        self.gconv1 = group_conv_block(in_channels = groups*30,out_channels = groups*20,kernel = 3, stride = 1,padding = 1,groups = groups)
        self.gn1 = nn.GroupNorm(groups,groups*20)
        self.gconv4 = group_conv_block(in_channels = groups*20,out_channels = groups,kernel = 1,stride = 1,groups = groups,padding = 0)
        self.gn2 = nn.GroupNorm(groups,groups)
    def forward(self,x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.conv3(z)
        if self.img_width == 128:
            z = self.conv4(z)
        #z = self.res_stack(z)
        # print(z.shape)
        # latent = F.gelu(self.gn1(self.gconv1(z)))
        out = {"lat":z}
        z = F.gelu(self.gn1(self.gconv1(z)))
        z = self.gconv4(z)
        z = self.gn2(z)
        out["pre_q"] = z
        return out#,z.view(-1,6)


