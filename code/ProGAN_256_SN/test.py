import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import args
from model import Generator, Discriminator
from utils import LoadCheckPoint, TestGenerator, GPUToNumpy


def SimpleTest(
        nIter       = 10,
        nImage      = 6,
        nBlock      = 5,
        nEpoch      = 30,
        factors     = args.CHANNEL_FACTORS, 
        imgChannels = args.IMG_CHANNELS,
        maxChannels = args.MAX_CHANNELS,
        noiseType   = "normal",
        device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ):
    noise = None
    imgSize = 4 * 2 ** (nBlock - 1)
    checkpoint = LoadCheckPoint(imgSize, nEpoch, args.MODEL_SAVE_FOLDER)
    
    gen = Generator(maxChannels, imgChannels, factors)
    gen.to(device)
    gen.load_state_dict(checkpoint["gen"])
    
    dis = Discriminator(imgChannels, maxChannels, factors)
    dis.to(device)
    dis.load_state_dict(checkpoint["dis"])
    
    for _ in range(nIter):
        if noiseType == "normal":
            noise = torch.randn(nImage, maxChannels, 1, 1, dtype=torch.float, device=device)
        elif noiseType == "uniform":
            noise = torch.rand (nImage, maxChannels, 1, 1, dtype=torch.float, device=device)
        
        TestGenerator(gen, noise, nBlock, 1, device, title=f"Size={imgSize}, Epoch={nEpoch}")
    
    return gen, dis, noise


def GenerateImages(generator, nImage, noiseLen, saveFolder, batchSize=256, 
                   device="cuda:0", ext="jpg", genParams={}, isNoise4D=True):
    nLeft = nImage
    i = 0
    generator.eval()
    with torch.no_grad():
        while nLeft > 0:
            n = min(nLeft, batchSize)
            nLeft -= batchSize
            
            noise = torch.randn(n, noiseLen, dtype=torch.float, device=device)
            if isNoise4D:
                noise = noise.view(n, noiseLen, 1, 1)
            
            generator.to(device)
            imgs = generator(noise, **genParams)
            for j in range(n):
                i += 1
                img = imgs[j]
                img = np.clip((GPUToNumpy(img) + 1) * 0.5, 0, 1)
                imgName = os.path.join(saveFolder, f"{i}.{ext}")
                plt.imsave(imgName, img)
                print(f"\r{i}", end="")
            
        print("")
    
    return imgs


if __name__ == "__main__":
    gen, dis, noise = SimpleTest(1)
    imgs = GenerateImages(gen, 
                          30000, 
                          args.MAX_CHANNELS, 
                          args.GENERATE_IMGS_FOLDER,
                          genParams={"nBlock": 5, "alpha": 1})
    
