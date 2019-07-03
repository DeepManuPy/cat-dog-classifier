from keras.models import load_model
from deepmanupy import config
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",type=str,required=True,help="Path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
output = image.copy()
output = imutils.resize(output,width=400)

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(224,224))

image = image.astype("float32")
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
image = image - mean
print("[INFO] Loading Model...")
model = load_model(config.MODEL_PATH)

preds = model.predict(np.expand_dims(image,axis=0))[0]
i = np.argmax(preds)
label = config.CLASSES[i]

text = "{}: {:.2f}%".format(label,preds[i]*100)
cv2.putText(output,text,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

cv2.imshow("Output",output)
cv2.imwrite("output.png",output)
cv2.waitKey(0)
