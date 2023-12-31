import numpy as np
import cv2
import os

data_dir = os.path.join(os.getcwd(),'Consolidate')
img_dir = os.path.join(os.getcwd(),'images')

images = []
labels = []

for i in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,i))
    img = cv2.resize(img,(200,200))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    images.append(img)
    labels.append(i.split('_')[0].lower())
    
images = np.array(images)
labels = np.array(labels)

import pickle  #Used to create pickle files(.p) which are nothing but compressed files of given name 

with open(os.path.join(data_dir , 'images.p'),'wb') as f:
    pickle.dump(images,f)

with open(os.path.join(data_dir , 'labels.p'),'wb') as f:
    pickle.dump(labels,f)





