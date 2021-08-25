import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SpectralNorm(nn.Module):
    def __init__(self, module, nPowerIter=1):
        super(SpectralNorm, self).__init__()
        self.module = spectral_norm(module, n_power_iterations=nPowerIter)
        
    def forward(self, x):
        return self.module(x)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / ((x ** 2).mean(dim=1, keepdim=True) + self.eps) ** 0.5


class BatchStd(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        b, h, w = x.size(0), x.size(2), x.size(3)
        std = torch.std(x, dim=0).mean().repeat(b, 1, h, w)
        return torch.cat([x, std], dim=1)
        

class HeConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias=False)
        self.bias = nn.Parameter(torch.zeros([1, outChannels, 1, 1]))
        self.scale = (gain / (inChannels * kernelSize ** 2)) ** 0.5
        
        self.InitWeights()
        
    def InitWeights(self):
        nn.init.normal_(self.conv.weight)
    
    def forward(self, x):
        return self.conv(x) * self.scale + self.bias


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, isPixelNorm=True, pNormEps=1e-8):
        super().__init__()
        self.conv1 = HeConv2d(inChannels , outChannels)
        self.conv2 = HeConv2d(outChannels, outChannels)
        self.lReLU = nn.LeakyReLU(0.2)
        self.pNorm = PixelNorm(pNormEps) if isPixelNorm else None
    
    def forward(self, x):
        h = self.lReLU(self.conv1(x))
        h = self.pNorm(h) if self.pNorm else h
        h = self.lReLU(self.conv2(h))
        h = self.pNorm(h) if self.pNorm else h
        return h


class SNConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias=False))
        self.bias = nn.Parameter(torch.zeros([1, outChannels, 1, 1]))
        self.scale = (gain / (inChannels * kernelSize ** 2)) ** 0.5
        
        self.InitWeights()
        
    def InitWeights(self):
        nn.init.normal_(self.conv.module.weight)
    
    def forward(self, x):
        return self.conv(x)


class SNConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, isPixelNorm=True, pNormEps=1e-8):
        super().__init__()
        self.conv1 = SNConv2d(inChannels , outChannels)
        self.conv2 = SNConv2d(outChannels, outChannels)
        self.lReLU = nn.LeakyReLU(0.2)
        self.pNorm = PixelNorm(pNormEps) if isPixelNorm else None
    
    def forward(self, x):
        h = self.lReLU(self.conv1(x))
        h = self.pNorm(h) if self.pNorm else h
        h = self.lReLU(self.conv2(h))
        h = self.pNorm(h) if self.pNorm else h
        return h


class Generator(nn.Module):
    def __init__(self, maxChannels, imgChannels, factors):
        super().__init__()
        self.initBlock = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(maxChannels, maxChannels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            HeConv2d(maxChannels, maxChannels),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initToRGB = nn.Sequential(
            HeConv2d(maxChannels, imgChannels, 1, 1, 0)
        )
        self.upSample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.proBlocks = nn.ModuleList([])
        self.rgbBlocks = nn.ModuleList([self.initToRGB])
        
        lastChannels = maxChannels
        for factor in factors:
            channels = int(maxChannels * factor)
            self.proBlocks.append(nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    ConvBlock(lastChannels, channels)
            ))
            self.rgbBlocks.append(nn.Sequential(
                    HeConv2d(channels, imgChannels, 1, 1, 0)
            ))
            lastChannels = channels
        
        self.float()
            
    
    def forward(self, x, nBlock, alpha=1.):
        h = self.initBlock(x)
        if nBlock == 1:
            return self.initToRGB(h)
        
        for i in range(nBlock - 1):
            last = h
            h = self.proBlocks[i](last)
        
        out = self.rgbBlocks[nBlock - 1](h)
        up  = self.rgbBlocks[nBlock - 2](self.upSample2(last))
        return (1 - alpha) * up + alpha * out


class Discriminator(nn.Module):
    def __init__(self, imgChannels, maxChannels, factors):
        super().__init__()
        self.initBlock = nn.Sequential(
            BatchStd(),
            SNConv2d(maxChannels + 1, maxChannels),
            nn.LeakyReLU(0.2),
            SNConv2d(maxChannels, maxChannels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            SNConv2d(maxChannels, 1, 1, 1, 0),
            nn.Flatten()
        )
        self.initFromRGB = nn.Sequential(
            SNConv2d(imgChannels, maxChannels, 1, 1, 0)
        )
        self.downSample2 = nn.AvgPool2d(2)
        self.proBlocks   = nn.ModuleList([])
        self.rgbBlocks   = nn.ModuleList([self.initFromRGB])

        for i in range(len(factors) - 2, -2, -1):
            lastChannels = int(maxChannels * factors[i + 1])
            channels     = int(maxChannels * factors[i] if i >= 0 else maxChannels)
            self.proBlocks.insert(0, nn.Sequential(
                    SNConvBlock(lastChannels, channels, isPixelNorm=False),
                    nn.AvgPool2d(2)
            ))
            self.rgbBlocks.insert(1, nn.Sequential(
                    SNConv2d(imgChannels, lastChannels, 1, 1, 0)
            ))
        
        self.float()
        
    
    def forward(self, x, nBlock, alpha=1.):
        if nBlock == 1:
            return self.initBlock(self.initFromRGB(x))
        
        last = x
        for i in range(nBlock - 2, -1, -1):
            if i == nBlock - 2:
                out  = self.proBlocks[i](self.rgbBlocks[i + 1](last))
                down = self.downSample2 (self.rgbBlocks[i    ](last))
                h = (1 - alpha) * down + alpha * out
            else:
                h = self.proBlocks[i](last)
            
            last = h
        
        return self.initBlock(h)


class DiscriminatorLoss(nn.Module):
    def __init__(self, ganLossWeight=1, driftLossWeight=0.001):
        super().__init__()
        self.w = [ganLossWeight, driftLossWeight]
    
    def forward(self, dis, fakeImgs, realImgs, fakeOuts, realOuts, nBlock, alpha):
        ganLoss = fakeOuts.mean() - realOuts.mean()
        drift   = (realOuts ** 2).mean()
        return self.w[0] * ganLoss +  self.w[1] * drift


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, fakeOuts):
        return -fakeOuts.mean()



if __name__=='__main__':
    factors     = [1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
    nBlock      = 5
    maxChannels = 256
    imgChannels = 3
    batchSize   = 32
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler      = torch.cuda.amp.GradScaler()
    
    gen = Generator(maxChannels, imgChannels, factors).to(device)
    dis = Discriminator(imgChannels, maxChannels, factors).to(device)
    
    genOpt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    disOpt = torch.optim.Adam(dis.parameters(), lr=1e-3)
    
    gen.requires_grad_(True)
    dis.requires_grad_(True)
    
    with torch.cuda.amp.autocast():
        noise = torch.rand(batchSize, maxChannels, 1, 1).to(device)
        fakeImgs = gen(noise, nBlock, 0.5)
        fakeOuts = dis(fakeImgs, nBlock, 0.5)

    loss = fakeOuts.mean()
    scaler.scale(loss).backward()
    scaler.step(genOpt)
    scaler.update()
    
    
    
    
    
    
    
    

