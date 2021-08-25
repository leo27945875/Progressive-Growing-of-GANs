IMAGES_FOLDER        = "../data"
MODEL_SAVE_FOLDER    = "./models"
GENERATE_IMGS_FOLDER = "./generate_imgs" 

IS_LOAD_MODEL        = False
LOAD_NUM_BLOCK       = 5
LOAD_EPOCH           = 1

CHANNEL_FACTORS      = [1, 1, 1, 1/2]
MAX_CHANNELS         = 256
IMG_CHANNELS         = 3

GAN_LOSS_PENALTY     = 1
GRADIENT_PENALTY     = 10
DRIFT_PENALTY        = 0.001

START_NUM_BLOCK      = 5
START_EPOCH          = 1
EACH_IMG_SIZE_EPOCHS = 30

START_ALPHA          = None
BATCH_SIZES          = [32, 32, 32, 32, 32]
LEARNING_RATE        = 1e-3
TTUR_FACTOR          = 1
DATA_LOAD_WORKERS    = 8

IS_SAVE_MOEDL        = False
SAVE_PERIOD          = 1
