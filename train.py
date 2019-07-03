import matplotlib
matplotlib.use("Agg")
# import necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Flatten,Dense,BatchNormalization,Dropout,Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from deepmanupy import config
from imutils import paths
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

def plot_training(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
# derive the paths to the training, validation directories
trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
# determine the total number of image paths in training, validation directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
# initilaize the training data augmentation object
trainAug = ImageDataGenerator(
rotation_range = 15,
zoom_range = 0.15,
width_shift_range = 0.2,
height_shift_range = 0.2,
horizontal_flip = True,
fill_mode = "nearest"
)
valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939],dtype="float32")
trainAug.mean = mean
valAug.mean = mean
# initialize the training generator
trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)
# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu",kernel_initializer="he_uniform")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process

for layer in baseModel.layers:
	layer.trainable = False
print("[INFO] compiling model...")
opt = Adam(lr=6e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
checkPoint = ModelCheckpoint("model_{val_loss:.3f}.model",
                            save_weights_only=False,save_best_only=True,
                            verbose=0,mode="min")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=8,
    callbacks=[checkPoint])

plot_training(H,8,config.TRAIN_PLOT_PATH)



trainGen.reset()
valGen.reset()
for layer in baseModel.layers[15:]:
	layer.trainable = True
print("[INFO] re-compiling model..")
opt = Adam(lr=6e-4)
checkPoint = ModelCheckpoint("model_{val_loss:.3f}.model",
                            save_weights_only=False,save_best_only=True,
                            verbose=0,mode="min")
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.load_weights("weights_0.061.hdf5")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=10,
	callbacks=[checkPoint]
)
plot_training(H,10,config.TRAIN_PLOT_PATH)
