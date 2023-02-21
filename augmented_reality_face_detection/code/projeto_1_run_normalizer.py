import cv2
import os
from projeto_1_a_haar_classifier import Haar_Eyes
from projeto_1_b_dnn import Dnn
from projeto_1_normalizer import normalize
from virtual_objects import normalize_obj
import numpy as np
from projeto_1_classifier_fisher import NeighborClassifierFisher
from projeto_1_classifer_eigen import NeighborClassifierEigen
from projeto_1_classify import load_all
from projeto_1_classify import show_label_name


haar_eyes = Haar_Eyes()
dnn = Dnn()

video_capture = cv2.VideoCapture(0)

eigen_classifier = NeighborClassifierEigen()
fisher_classifier = NeighborClassifierFisher()
X,y = load_all()

eigen_classifier.fit_nearest_neighbor(X,y,5)
fisher_classifier.fit_nearest_neighbor(X,y,10)

#directory = "C:\\Users\\duart\\OneDrive\\Ambiente de #Trabalho\\Mestrado\\VAR\\projeto_1\\tp1_45140_b"
#os.chdir(directory)
#print(os.listdir(directory))  
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    detections = dnn.get_detections(frames)
    image,faces_eyes = dnn.classify_faces(detections,frames)
    if (len(faces_eyes)!=0 and faces_eyes!=None):
        norm,face = normalize(image,faces_eyes)
        #print("norm",norm)
       
        #label_ar = np.array(arr)
        label=3

        if (len(norm)>0):
            arr = []
            arr.append(norm)
            label_arr= np.array(arr).astype(np.int32)
            
            #label = eigen_classifier.predict(label_ar)
            label = fisher_classifier.predict(label_arr)
            #print(label)
            normalize_obj(faces_eyes[0][0].astype(np.int32),frames,label)
            #labeln = show_label_name(label)
           



     # Display the resulting frame
    cv2.imshow('Video', frames)
   # if cv2.waitKey(1) & 0xFF == ord('e'):
       # str_name = "nome"+str(random.randint(10, 100))+".jpg"
       # print(str_name)qqq
       # cv2.imwrite(str_name, norm)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
       
        break
