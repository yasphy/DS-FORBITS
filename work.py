import torch
from PIL import Image
import numpy as np
import torchvision
import os
from torch.utils.data import DataLoader
class TFDataset(torch.utils.data.Dataset):
        def __init__(self, root, transforms=None):
            self.root = root
            self.transforms = transforms
    
        def __getitem__(self, idx):
            # load images
            img_path = self.root[idx]
            img = Image.open(img_path).convert("RGB")
            newsize = (1000, 1000)
            img = img.resize(newsize)
    
            if self.transforms is not None:
                img= self.transforms(img)
            return img
        def __len__(self):
            return len(self.root)
def image(im):
    im=np.squeeze(im)
    im=torch.from_numpy(im)
    im=im.permute(1,2,0).numpy()
    return im
def datasetmaker(dirname,name):
    m=TFDataset([os.path.join(dirname,name)],torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(m, batch_size=4, shuffle=True)
    train_features = next(iter(train_dataloader))
    #img=(train_features[0]).permute(1,2,0).numpy()
    return train_features
    
