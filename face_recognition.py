import numpy as np
import cv2
from PIL import Image
from keras.models import model_from_json
import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout,Activation,Flatten
from keras.optimizers import Adam
#from keras import regularizers
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
import joblib
#from sklearn.decomposition import PCA
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
from scipy import misc

import warnings
warnings.filterwarnings("ignore")

learning_rate = .001

def img_to_matrix(image):
    image=cv2.resize(image, (48,48))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("../../master_output.mp4")
cap.set(3,640)
cap.set(4,480)
i=0
json_file = open('../model_resampling_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("../facenet_resampling_1.h5")
loaded_model.compile(loss='mean_absolute_error', optimizer=Adam())
classifier=joblib.load('../GBM_resampling_1.joblib.pkl')
#princ=joblib.load('./PCA.joblib.pkl')

fourccc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('./master_output_recognized.mp4', fourccc, 20.0, (height,width))

labelss=['deni', 'sriram', 'rohit', 'bhadra', 'kavin']
while True:
    ret, img = cap.read()
    #print(img)
    #img = cv2.flip(img, -1)
    #print(np.asarray(img).shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        framess=np.asarray(img[y-int(0.1*h):y+h+int(0.1*h),x-int(0.1*w):x+w+int(0.1*w)])
        framess=cv2.resize(framess,(150,150))
        framess=np.divide(framess,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #framess=Image.fromarray(img[y:y+h,x:x+w])
        """framess.save('./not_kavin/{0}.png'.format(i))
        i+=1"""
        score=loaded_model.predict(framess.reshape(1,150,150,3))
        #co=princ.transform(score)
        cls=classifier.predict(score)
        #print(cls)
        print(cls)
        text=labelss[cls[0]]
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(img, "streaming on raspberry Pi", (1,1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        video_writer.write(img)
    #cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
