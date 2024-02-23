
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class isaac3d_data():
    def __init__(self, dataset_name="isaac"):
        super(isaac3d_data, self).__init__()

        self.dir = '/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images/'
        self.mul = np.array([184320,46080,7680,1920,480,120,24,3,1])
        self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/labels.npy"))
        self.size = self.raw_labels.shape[0]
        self.nlatents = self.raw_labels.shape[1]
        self.l = [3,8,5,4,4,4,6,4,4]
        self.cumulate = [0,3,11,16,20,24,28,34,38]
        #self.images = torch.from_numpy(np.load('/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images.npy', mmap_mode='r+')).permute(0,3,1,2)/255.0
        self.images = np.load('/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images.npy', mmap_mode='r+')
        self.labels = (self.raw_labels*(torch.tensor(self.l)-1)).long()

        self.perm = torch.randperm(self.size)
        train_size = int(self.size*0.95)
        self.train_ind = self.perm[:train_size]
        self.train_size = self.train_ind.shape[0]
        self.test_ind = self.perm[train_size:]
        self.test_size = self.test_ind.shape[0]
    def get_train_batch(self,batch_size=64):
        ind = self.train_ind[torch.randint(0,self.train_size,(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]
    def get_test_batch(self,batch_size=250):
        ind = self.test_ind[torch.randint(0,self.test_size,(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]


class falcor3d_data():
    def __init__(self, dataset_name="falcor"):
        super(falcor3d_data, self).__init__()

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

        self.perm = torch.randperm(self.size)
        train_size = int(self.size*0.95)
        self.train_ind = self.perm[:train_size]
        self.train_size = self.train_ind.shape[0]
        self.test_ind = self.perm[train_size:]
        self.test_size = self.test_ind.shape[0]
    def get_train_batch(self,batch_size=64):
        ind = self.train_ind[torch.randint(0,self.train_size,(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]
    def get_test_batch(self,batch_size=250):
        ind = self.test_ind[torch.randint(0,self.test_size,(batch_size,))]
        return torch.from_numpy(self.images[ind]).permute(0,3,1,2)/255.0,self.labels[ind],self.raw_labels[ind]
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]

class falcor3d(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,root_dir = '/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images/', transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        #self.label_dir = "/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/labels.npy"
        self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/train-rec.labels"))
        self.images = np.load('/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images.npy', mmap_mode='r+')
        self.lsize = []
        for i in range(7):
            self.lsize.append(torch.unique(self.raw_labels[:,i]).shape[0])
        self.labels = (self.raw_labels*(torch.tensor(self.lsize)-1)).long()
    
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s
        image = torch.from_numpy(self.images[idx]).permute(2,0,1)/255.

        if self.transform:
            image = self.transform(image)

        return image,self.labels[idx],self.raw_labels[idx]


class isaac3d(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,root_dir = '/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images/', transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        #self.label_dir = "/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/labels.npy"
        self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/labels.npy"))
        self.image = np.load('/mnt/beegfs/ksanka/data/issac3d/Isaac3D_down128/images.npy', mmap_mode='r+')
        self.lsize = []
        for i in range(9):
            self.lsize.append(torch.unique(self.raw_labels[:,i]).shape[0])
        self.labels = (self.raw_labels*(torch.tensor(self.lsize)-1)).int()
    
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        
        
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        #img_name = self.root_dir + str(idx).zfill(6) + '.png'
        image = torch.from_numpy(self.image[idx]).permute(2,0,1)/255.

        if self.transform:
            image = self.transform(image)

        return image,self.labels[idx],self.raw_labels[idx]

class Shapes3d(Dataset):
    def __init__(self, img_dir = "/mnt/beegfs/ksanka/data/3dshapes/3dshapes.h5", transform=None, target_transform=None):
        with h5py.File(img_dir, 'r') as file:
            images = file['images'][:]
            labels = file['labels'][:]
        self.images = torch.from_numpy(images).permute(0,3,1,2)/255.0
        self.labels = torch.from_numpy(labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def save(img,path,s):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig(os.path.join(path,f'{s}.pdf'), dpi=400)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
def get_ind(lab):
    # c5 = 1
    # c4 = 15
    # c3 = 60
    # c2 = 480
    # c1 = 4800
    # c0 = 48000
    mul = np.array([48000,4800,480,60,15,1])
    return np.sum(lab*mul)
    
def get_data(size,device = 'cuda',one_hot = False):
    #size = 100
    arng = torch.arange(size)
    a1 = np.random.permutation(arng)%10
    a2 = np.random.permutation(arng)%10
    a3 = np.random.permutation(arng)%10
    a4 = np.random.permutation(arng)%8
    a5 = np.random.permutation(arng)%4
    a6 = np.random.permutation(arng)%15
    
    lab = np.stack([a1,a2,a3,a4,a5,a6],1)
    ind = list(map(get_ind,lab))
    file_path = '/mnt/beegfs/ksanka/data/3dshapes/3dshapes.h5'
    with h5py.File(file_path, 'r') as file:
        images = file['images'][:]
        # raw_labels = file['labels'][:]
    
    elabels = torch.from_numpy(lab+[0,10,20,30,38,42]).to(device)
    labels = torch.from_numpy(lab).to(device)
    img = torch.from_numpy(images[ind,:]).to(device).permute(0,3,1,2)/255.0
    # if one_hot:
    one_hot_labels = torch.nn.functional.one_hot(elabels, num_classes=57).sum(dim = 1).to(device).float()
    return img,labels,one_hot_labels

# if __name__ == "__main__":
#     print("testing...")
#     ds = Shapes3d()
#     tloader = DataLoader(ds,batch_size = 128,shuffle=True)
#     img,lab = next(iter(tloader))
#     print(f"img batch shape = {img.shape}, label batch shape = {lab.shape}")
#     print("Everything passed")
    
        