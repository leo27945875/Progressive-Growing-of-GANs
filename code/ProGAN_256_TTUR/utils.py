import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        self.imagePaths = glob.glob(os.path.join(folder, '*.jpg'))
    
    def __getitem__(self, i):
        img = Image.open(self.imagePaths[i])
        img = self.transform(img)
        return img.float()
    
    def __len__(self):
        return len(self.imagePaths)
    

def GPUToNumpy(tensor, reduceDim=None, isSqueeze=True):
    if type(tensor) is np.array or type(tensor) is np.ndarray:
        return tensor
    
    if isSqueeze:
        if reduceDim is not None:
            return tensor.squeeze(reduceDim).cpu().detach().numpy().transpose(1, 2, 0)
        else:
            return tensor.squeeze(         ).cpu().detach().numpy().transpose(1, 2, 0)
    
    else:
        if len(tensor.shape) == 3:
            return tensor.cpu().detach().numpy().transpose(1, 2, 0)
        elif len(tensor.shape) == 4:
            return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)


def GetImageLoader(folder, imgSize, imgChannels, batchSize, nWorker=1):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(imgSize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize([0.5] * imgChannels, 
                                 [0.5] * imgChannels)
        ])
    dataset    = ImageDataset(folder, transform)
    dataLoader = DataLoader(dataset, 
                            batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=nWorker,
                            pin_memory=True)
    return dataLoader, dataset


def TestGenerator(gen, noises, nBlock, alpha, device, title=""):
    nImg = noises.size(0)
    gen.eval()
    with torch.no_grad():
        fakeImgs = gen(noises, nBlock, alpha)
        f, axes  = plt.subplots(1, nImg)
        for i in range(nImg):
            fakeImg = fakeImgs[i]
            fakeImg = np.clip(GPUToNumpy(fakeImg) * 0.5 + 0.5, 0, 1)
            axes[i].imshow(fakeImg)
        
    plt.title(title)
    plt.show()
    

def DownSmapleImages(srcFolder, dstFolder, rate=1/16):
    imgPaths = glob.glob(os.path.join(srcFolder, '*.jpg'))
    for imgPath in imgPaths:
        srcImg = plt.imread(imgPath)
        h, w = srcImg.shape[0], srcImg.shape[1]
        
        img = cv2.resize(srcImg, (int(w * rate), int(h * rate)), interpolation=cv2.INTER_CUBIC)
        path = os.path.join(dstFolder, os.path.split(imgPath)[-1])
        plt.imsave(path, img)
        print(f"\rSaved {path}", end="")
    
    print("")


def SaveCheckPoint(gen, dis, genOpt, disOpt, imgSize, epoch, folder):
    torch.save(gen   , os.path.join(folder, f"TTURGenModel_Size{imgSize}_Epoch{epoch}.pth"))
    torch.save(dis   , os.path.join(folder, f"TTURDisModel_Size{imgSize}_Epoch{epoch}.pth"))
    torch.save(genOpt, os.path.join(folder, f"TTURGenOpt_Size{imgSize}_Epoch{epoch}.pth"  ))
    torch.save(disOpt, os.path.join(folder, f"TTURDisOpt_Size{imgSize}_Epoch{epoch}.pth"  ))
    
    # checkpoint = {"gen": gen.state_dict(),
    #               "dis": dis.state_dict(),
    #               "genOpt": genOpt.state_dict(),
    #               "disOpt": disOpt.state_dict()}
    # torch.save(checkpoint, os.path.join(folder, f"TTURCheckpoint_Size{imgSize}_Epoch{epoch}.pth"))
    


def LoadCheckPoint(imgSize, epoch, folder, isLoadOpt=True):
    gen = torch.load(os.path.join(folder, f"TTURGenModel_Size{imgSize}_Epoch{epoch}.pth"))
    dis = torch.load(os.path.join(folder, f"TTURDisModel_Size{imgSize}_Epoch{epoch}.pth"))
    if isLoadOpt:
        genOpt = torch.load(os.path.join(folder, f"TTURGenOpt_Size{imgSize}_Epoch{epoch}.pth"))
        disOpt = torch.load(os.path.join(folder, f"TTURDisOpt_Size{imgSize}_Epoch{epoch}.pth"))
        return gen, dis, genOpt, disOpt
    else:
        return gen, dis
    
    # return torch.load(os.path.join(folder, f"TTURCheckpoint_Size{imgSize}_Epoch{epoch}.pth"))


def SeedEverything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def GetAlpha(epoch, totalEpochs, initAlpha=1e-5):
    return min(1, initAlpha + (epoch - 1) / (0.5 * totalEpochs))











