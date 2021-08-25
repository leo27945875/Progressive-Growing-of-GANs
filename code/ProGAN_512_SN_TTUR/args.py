IMAGES_FOLDER        = r"..\data"
MODEL_SAVE_FOLDER    = r".\models"
GENERATE_IMGS_FOLDER = r".\generate_imgs" 

IS_LOAD_MODEL        = True
LOAD_NUM_BLOCK       = 5
LOAD_EPOCH           = 13

CHANNEL_FACTORS      = [1, 1, 1, 1/2]
MAX_CHANNELS         = 512
IMG_CHANNELS         = 3

GAN_LOSS_PENALTY     = 1
DRIFT_PENALTY        = 0.001

START_NUM_BLOCK      = 5
START_EPOCH          = 14
EACH_IMG_SIZE_EPOCHS = 50

START_ALPHA          = 1
BATCH_SIZES          = [32, 32, 32, 32, 32]
LEARNING_RATE        = 1e-3
TTUR_FACTOR          = 5
DATA_LOAD_WORKERS    = 8

IS_SAVE_MODEL        = True
SAVE_PERIOD          = 1
