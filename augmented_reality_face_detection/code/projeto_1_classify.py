from ctypes.wintypes import HDESK
import os
import cv2
import numpy as np
from projeto_1_eigen_faces import EigenFaces
from projeto_1_helper import Helper
from projeto_1_classifer_eigen import NeighborClassifierEigen
from projeto_1_classifier_fisher import NeighborClassifierFisher

folder_pj="./images"
folder_ar="./images_arman"
folder_tav="./images_tavora"
folder_du="./images_duarte"

def load_all():
    imgs_0,y0= Helper().load_images(folder_ar,0)
    imgs_1,y1= Helper().load_images(folder_pj,1)
    imgs_2,y2= Helper().load_images(folder_tav,2)
    imgs_3,y3= Helper().load_images(folder_du,3)

    #join images
    imgs = np.concatenate((imgs_0, imgs_1,imgs_2,imgs_3))
    width = imgs.shape[1]
    height = imgs.shape[2]

    #X
    X= imgs
    #y
    #y = Helper().create_y(imgs,imgs_0,imgs_1,imgs_2)
    y = np.concatenate((y0, y1,y2,y3))

    #X,y = Helper().remove_values(X,y,4)

    #   print(X.shape)
    #print(len(y))


#Labels of y
    #labels = np.unique(y)

    return X,y


def show_label_name(label):
       
    if (label==0):
        return "Arman"
    
    elif(label==1):
        return "Pedro Jorge"
    elif(label==2):
        return "Taveira"
    elif(label==3):
        return "Duarte"
    


'''
X,y = load_all()


eigen_classifier = NeighborClassifierEigen()
eigen_classifier.fit_nearest_neighbor(X,y,5)






arr_imgs = X[30:31]

predicted_labes = eigen_classifier.predict(arr_imgs)
print("--------------------- EigenFaces Classification -------------------------------------------")
print("\t")
print("Original y: ", y)
print("Predicted y: ", predicted_labes)


print("--------------------- FisherFaces Classification -------------------------------------------")
print("\t")

fisherClassifier = NeighborClassifierFisher()
fisherClassifier.fit_nearest_neighbor(X,y,5)

labels_Fisher = fisherClassifier.predict(arr_imgs)
print("Original y: ", y)
print("Predicted y: ", labels_Fisher)
'''



