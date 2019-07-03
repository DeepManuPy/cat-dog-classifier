import os
# initilaize the path to the original input directory of images
ORIG_INPUT_DATASET = "dogs-vs-cats"
# initilaize the base path to the *new* directory
BASE_PATH = "dataset"
# define the names of train and test directories
TRAIN = "train"
VAL = "val"
TEST = "test"
# initilaize the list of class label names
CLASSES = ["cat","dog"]
# set the batch size
BATCH_SIZE = 32
# set path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output","catvsdog.model"])

TRAIN_PLOT_PATH = os.path.sep.join(["output","train_history.png"])
