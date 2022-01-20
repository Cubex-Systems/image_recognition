import cv2
import numpy as np
from keras.models import load_model

model= load_model('model_file.h5')

labels_dict = {0:'template_1', 1:'template_2', 2:'template_3', 3:'template_4'}

frame = cv2.imread('ML/train/template_4/blank-template-4.jpg')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

last = cv2.resize(gray,(48,48))
last = last/255.0
last= np.reshape(last,(1,48,48,1))
result = model.predict(last)
label = np.argmax(result,axis=1)[0]
print(label)
print(labels_dict[label])