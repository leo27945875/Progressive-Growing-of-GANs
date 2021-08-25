import args
from utils import GetImageLoader, TestGenerator, LoadCheckPoint, SaveCheckPoint
from model import Generator, Discriminator, DiscriminatorLoss, GeneratorLoss

import torch
import torch.optim as optim


torch.backends.cudnn.benchmarks = True


if __name__ == '__main__':
    
    modelFolder = args.MODEL_SAVE_FOLDER
    imgFolder   = args.IMAGES_FOLDER
    nDataWorker = args.DATA_LOAD_WORKERS
    maxChannels = args.MAX_CHANNELS
    imgChannels = args.IMG_CHANNELS
    factors     = args.CHANNEL_FACTORS
    genLR       = args.LEARNING_RATE
    disLR       = args.LEARNING_RATE * args.TTUR_FACTOR
    savePeriod  = args.SAVE_PERIOD
    batchSizes  = args.BATCH_SIZES
    epochs      = args.EACH_IMG_SIZE_EPOCHS
    startEpoch  = args.START_EPOCH
    startBlocks = args.START_NUM_BLOCK
    startAlpha  = args.START_ALPHA
    nBlocks     = len(factors) + 1
    
    assert len(batchSizes) == nBlocks
    assert startEpoch > 0
    
    
    if args.IS_LOAD_MODEL:
        gen, dis = LoadCheckPoint(4 * 2 ** (args.LOAD_NUM_BLOCK - 1), 
                                  args.LOAD_EPOCH, 
                                  modelFolder)
    else:
        gen = Generator(maxChannels, imgChannels, factors)
        dis = Discriminator(imgChannels, maxChannels, factors)
        
    genOpt = optim.Adam(gen.parameters(), lr=genLR, betas=(0, 0.99))
    disOpt = optim.Adam(dis.parameters(), lr=disLR, betas=(0, 0.99))
    
    genLoss = GeneratorLoss()
    disLoss = DiscriminatorLoss(args.GAN_LOSS_PENALTY, args.GRADIENT_PENALTY, args.DRIFT_PENALTY)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen    .to(device)
    dis    .to(device)
    genLoss.to(device)
    disLoss.to(device)
    
    fixedNoise = torch.randn(3, maxChannels, 1, 1, dtype=torch.float, device=device)
    
    for nBlock in range(startBlocks, nBlocks + 1):
        imgSize    = 4 * 2 ** (nBlock - 1)
        batchSize  = batchSizes[nBlock - 1]
        startEpoch = 1 if nBlock > startBlocks else startEpoch
        print('=' * 30 + f" Training imgSize=[{imgSize}, {imgSize}] " + '=' * 30)
        
        loader, dataset = GetImageLoader(imgFolder, imgSize, imgChannels, batchSize, nDataWorker)
        alpha = 1e-5 + (startEpoch - 1) / (0.5 * epochs) if startAlpha is None else startAlpha
        nData = len(dataset)

        for epoch in range(startEpoch, epochs + 1):
            gen.train()
            dis.train()
            totalLossGan, totalLossDis, cumBatch = 0, 0, 0
            for i, realImgs in enumerate(loader):
                nowBatchSize = realImgs.size(0)
                
                # Train the discriminator
                gen.requires_grad_(False)
                dis.requires_grad_()
                
                noises   = torch.randn(nowBatchSize, maxChannels, 1, 1, dtype=torch.float, device=device)
                realImgs = realImgs.to(device)
                fakeImgs = gen(noises, nBlock, alpha).detach()
                
                realOuts = dis(realImgs, nBlock, alpha)
                fakeOuts = dis(fakeImgs, nBlock, alpha)
                lossDis  = disLoss(dis, fakeImgs, realImgs, fakeOuts, realOuts, nBlock, alpha)
                    
                disOpt.zero_grad()
                lossDis.backward()
                disOpt.step()
                    
                # Train the generator
                gen.requires_grad_()
                dis.requires_grad_(False)
                
                noises   = torch.randn(nowBatchSize, maxChannels, 1, 1, dtype=torch.float, device=device)
                fakeImgs = gen(noises  , nBlock, alpha)
                fakeOuts = dis(fakeImgs, nBlock, alpha)
                lossGen  = genLoss(fakeOuts)
        
                genOpt.zero_grad()
                lossGen.backward()
                genOpt.step()
                
                # Print training info
                cumBatch     += nowBatchSize
                totalLossDis += lossDis.item()
                totalLossGan += lossGen.item()
                print(f"\r| Epoch {epoch}/{epochs} | Batch {cumBatch}/{nData} | Alpha {round(alpha, 4)} | => lossGen = {totalLossGan / cumBatch} | lossDis = {totalLossDis / cumBatch}", end="")
                
                # Update alpha
                alpha += nowBatchSize / (0.5 * epochs * nData)
                alpha = min(alpha, 1)
        
            print("")
            
            # Test generator
            if args.IS_SAVE_MOEDL and (epoch % savePeriod == 0 or epoch == 1):
                SaveCheckPoint(gen, dis, imgSize, epoch, modelFolder)
                
            TestGenerator(gen, fixedNoise, nBlock, alpha, 
                          device, title=f"Size={imgSize}, Epoch={epoch}")
    
    
    
    

