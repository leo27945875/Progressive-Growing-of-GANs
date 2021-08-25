## 1. 參數設定: 
在args.py中進行參數的設定。
* IMAGES_FOLDER        = 資料集位置
* MODEL_SAVE_FOLDER    = 模型儲存位置
* GENERATE_IMGS_FOLDER = 生成圖片儲存位置 

* IS_LOAD_MODEL        = 是否載入先前訓練的模型權重
* LOAD_NUM_BLOCK       = 載入的模型訓練到第幾層
* LOAD_EPOCH           = 載入的模型訓練到第幾個epoch

* CHANNEL_FACTORS      = 設定每層的channel數與MAX_CHANNELS的比值
* MAX_CHANNELS         = 設定最高的channel數
* IMG_CHANNELS         = 設定圖片的channel數

* GAN_LOSS_PENALTY     = Adversarial loss的權重
* GRADIENT_PENALTY     = Gradient panelty的權重
* DRIFT_PENALTY        = Drift loss的權重

* START_NUM_BLOCK      = 開始訓練的層數
* START_EPOCH          = 開始訓練的epoch
* EACH_IMG_SIZE_EPOCHS = 每添加新層所要訓練的epoch數

* START_ALPHA          = 開始訓練的alpha值
* BATCH_SIZES          = 每添加新層所要訓練的batch size大小
* LEARNING_RATE        = 學習率
* TTUR_FACTOR          = discriminator的學習率是generator的幾倍
* DATA_LOAD_WORKERS    = 多少個worker來讀入訓練資料集

* IS_SAVE_MOEDL        = 是否儲存模型
* SAVE_PERIOD          = 每幾個epoch儲存1次模型

## 訓練模型:
參數設定好後直接run train.py即可訓練模型。

## 測試模型:
訓練好並儲存好模型參數後，可以使用test.py來測試模型。裡面有兩個method可用:
* SimpleTest(
    nIter       = 10,
    nImage      = 6,
    nBlock      = 5,
    nEpoch      = 50,
    factors     = args.CHANNEL_FACTORS, 
    imgChannels = args.IMG_CHANNELS,
    maxChannels = args.MAX_CHANNELS,
    noiseType   = "normal",
    noises      = None,
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)
* GenerateImages(
    generator, 
    nImage, 
    noiseLen, 
    saveFolder, 
    batchSize  = 128, 
    device     = "cuda:0", 
    ext        = "jpg", 
    genParams  = {}, 
    isNoise4D  = True
)

SimpleTest可以生成一些圖片做簡單的小測試。 \
GenerateImages是確定模型可用以後大量生成圖片至[GENERATE_IMGS_FOLDER]。