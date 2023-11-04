#final part of project ---------->>>>>>>>>>
import numpy as np
import cv2
import matplotlib.pyplot as plt
import urllib
from keras.models import load_model

print("model loading...... ")

label_map = ['bishal','debashish','deepak','hitesh','sambid']
model = load_model('Face_CET.h5')
print("model loaded")

def preprocess(img):
    img = cv2.resize(img,(200,200))
    img = cv2.cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img = img.reshape(1,200,200,1)
    img = img/255
    return img


face_data = "haarcascade_frontalface_default.xml"
#face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY) 
classifier = cv2.CascadeClassifier(face_data)

URL = "http://192.168.42.129:8080/shot.jpg"

ret = True 
while ret:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),np.uint8)
    image = cv2.imdecode(image,-1)    
    faces = classifier.detectMultiScale(image,1.2,5)
    for x,y,w,h in faces:
        face_img = image[y:y+h,x:x+w].copy()
        
        pred = model.predict_classes(preprocess(face_img))
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),5)
        cv2.putText(image,label_map[pred],(x,y),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow('live vid',image)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()        






















