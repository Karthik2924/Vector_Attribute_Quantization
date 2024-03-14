import h5py
import numpy as np
import torch
import yaml
with open("config.yaml", "r") as stream:
    cfg = yaml.safe_load(stream)
device = cfg['device']
#device=  'cuda'

class falcor3d_data():
    def __init__(self, balanced_subset_size = 5000):
        super(falcor3d_data, self).__init__()
        self.subset_size = balanced_subset_size
        self.dir = '/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images/'
        self.mul = np.array([46656,7776,1296,216,36,6,1])
        self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/train-rec.labels"))
        self.size = self.raw_labels.shape[0]
        self.nlatents = self.raw_labels.shape[1]
        self.l = [5,6,6,6,6,6,6]
        self.cumulate = [0,5,11,17,23,29,35]
        #self.images = torch.from_numpy(np.load('/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images.npy', mmap_mode='r+')).permute(0,3,1,2)/255.0
        self.images =np.load('/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images.npy', mmap_mode='r+')
        self.labels = (self.raw_labels*(torch.tensor(self.l)-1)).long()

        self.get_variables()
    def get_variables(self):
        self.perm = torch.randperm(self.size)
        train_size = int(self.size*0.95)
        self.train_ind = self.perm[:train_size]
        self.train_size = self.train_ind.shape[0]
        self.test_ind = self.perm[train_size:]
        self.test_size = self.test_ind.shape[0]
        self.sup,self.unsup,self.test = self.get_balanced_subset(self.subset_size)
    def get_train_batch(self,batch_size=64):
        ind = self.train_ind[torch.randint(0,self.train_size,(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]
    def get_test_batch(self,batch_size=250):
        ind = self.test_ind[torch.randint(0,self.test_size,(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]
    def get_sup_batch(self,batch_size = 64):
        ind = self.sup[torch.randint(0,len(self.sup),(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
    def get_unsup_batch(self,batch_size = 64):
        ind = self.unsup[torch.randint(0,len(self.unsup),(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
    def get_t_batch(self,batch_size = 64):
        ind = self.test[torch.randint(0,len(self.test),(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
    def get_ind(self,lab):
        return np.sum(lab*self.mul)
    
    def get_balanced_subset(self,size):
        arng = np.arange(size)
        lab = np.zeros((size,len(self.l)))
        for i in range(len(self.l)):
            lab[:,i] = np.random.permutation(arng)%self.l[i]

        ind = list(map(self.get_ind,lab))        
        full_ind = torch.arange(self.size)
        full_ind[ind] = -1
        full_ind = full_ind[full_ind>-1]
        perm = torch.randperm(full_ind.shape[0])
        unsup_ind = full_ind[perm[:-5000]]
        test_ind = full_ind[perm][-5000:]
        return torch.tensor(ind).int(),unsup_ind,test_ind

class isaac3d_data(falcor3d_data):
    
    def __init__(self, balanced_subset_size = 10000):
        super(isaac3d_data, self).__init__()

        self.dir = '/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images/'
        self.mul = np.array([184320,46080,7680,1920,480,120,24,3,1])
        self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/labels.npy"))
        self.size = self.raw_labels.shape[0]
        self.nlatents = self.raw_labels.shape[1]
        self.l = [3,8,5,4,4,4,6,4,4]
        self.cumulate = [0,3,11,16,20,24,28,34,38]
        self.images = np.load('/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images.npy', mmap_mode='r+')
        self.labels = (self.raw_labels*(torch.tensor(self.l)-1)).long()

        self.get_variables()

class shapes3d_dataset(falcor3d_data):
    def __init__(self, subset_size = 5000):
        super(shapes3d_dataset, self).__init__()

        img_dir = "/mnt/beegfs/ksanka/data/3dshapes/3dshapes.h5"
        with h5py.File(img_dir, 'r') as file:
            self.images = file['images'][:]
            self.raw_labels = torch.from_numpy(file['labels'][:])
        
        # self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/train-rec.labels"))
        self.size = self.raw_labels.shape[0]
        self.nlatents = self.raw_labels.shape[1]
        self.l = [10,10,10,8,4,15]
        self.cumulate = [0,10,20,30,38,42]
        self.mul = np.array([48000,4800,480,60,15,1])
        #for i in range(6):
        self.labels = torch.zeros_like(self.raw_labels)
        self.labels[:,0] = self.raw_labels[:,0]*10
        self.labels[:,1] = self.raw_labels[:,1]*10
        self.labels[:,2] = self.raw_labels[:,2]*10
        self.labels[:,3] = (((self.raw_labels[:,3]*7)-5.25)*2).int()
        self.labels[:,4] = self.raw_labels[:,4].int()
        self.labels[:,5] = ((self.raw_labels[:,5]+30)/4.2857).int()
        self.get_variables()

if __name__ == "__main__":
    print("testing...")
    batch_size = 32
    ds = shapes3d_dataset()
    a,b,c = ds.get_sup_batch(batch_size)
    assert a.shape == (batch_size,3,64,64), f"shape of image_batch should be ({batch_size},3,64,64)"
    assert b.shape == (batch_size,6) and c.shape ==(batch_size,6), f"shape of label shoudl be ({batch_size},6)"
    a,b,c = ds.get_unsup_batch(batch_size)
    assert a.shape == (batch_size,3,64,64), f"shape of image_batch should be ({batch_size},3,64,64)"
    assert b.shape == (batch_size,6) and c.shape ==(batch_size,6), f"shape of label shoudl be ({batch_size},6)"
    a,b,c = ds.get_test_batch(batch_size)
    assert a.shape == (batch_size,3,64,64), f"shape of image_batch should be ({batch_size},3,64,64)"
    assert b.shape == (batch_size,6) and c.shape ==(batch_size,6), f"shape of label shoudl be ({batch_size},6)"

    ds = falcor3d_data()
    a,b,c = ds.get_sup_batch(batch_size)
    assert a.shape == (batch_size,3,128,128), f"shape of image_batch should be ({batch_size},3,128,128)"
    assert b.shape == (batch_size,7) and c.shape ==(batch_size,7), f"shape of label shoudl be ({batch_size},7)"
    a,b,c = ds.get_unsup_batch(batch_size)
    assert a.shape == (batch_size,3,128,128), f"shape of image_batch should be ({batch_size},3,128,128)"
    assert b.shape == (batch_size,7) and c.shape ==(batch_size,7), f"shape of label shoudl be ({batch_size},7)"
    a,b,c = ds.get_test_batch(batch_size)
    assert a.shape == (batch_size,3,128,128), f"shape of image_batch should be ({batch_size},3,128,128)"
    assert b.shape == (batch_size,7) and c.shape ==(batch_size,7), f"shape of label shoudl be ({batch_size},7)"

    ds = isaac3d_data()
    a,b,c = ds.get_sup_batch(batch_size)
    assert a.shape == (batch_size,3,128,128), f"shape of image_batch should be ({batch_size},3,128,128)"
    assert b.shape == (batch_size,9) and c.shape ==(batch_size,9), f"shape of label shoudl be ({batch_size},9)"
    a,b,c = ds.get_unsup_batch(batch_size)
    assert a.shape == (batch_size,3,128,128), f"shape of image_batch should be ({batch_size},3,128,128)"
    assert b.shape == (batch_size,9) and c.shape ==(batch_size,9), f"shape of label shoudl be ({batch_size},9)"
    a,b,c = ds.get_test_batch(batch_size)
    assert a.shape == (batch_size,3,128,128), f"shape of image_batch should be ({batch_size},3,128,128)"
    assert b.shape == (batch_size,9) and c.shape ==(batch_size,9), f"shape of label shoudl be ({batch_size},9)"
    print("all tests passed")
    


        
        