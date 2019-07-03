from deepmanupy import config
from imutils import paths
import split_folders
import shutil
import os

#loop over the data splits
for split in (config.TRAIN):
    print("[INFO] processing '%s split'"%(split))
    p = os.path.sep.join([config.ORIG_INPUT_DATASET,split])
    imagePaths = list(paths.list_images(p))

    #loop over the image paths
    for imagePath in imagePaths:
        #extract class label name from filename
        filename = imagePath.split(os.path.sep)[-1]
        label = filename.split(".")[0]
        # construct the path to the output directory
        dirPath = os.path.sep.join([config.BASE_PATH,split,label])

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        p = os.path.sep.join([dirPath,filename])
        shutil.copy2(imagePath,p)

p = os.path.sep.join([config.BASE_PATH,config.TRAIN])
split_folders.ratio(p,output="data",seed=1337,ratio=(.8,.2))
