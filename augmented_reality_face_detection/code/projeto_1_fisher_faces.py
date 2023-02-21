from audioop import mul
import os
from re import sub
import cv2
import numpy as np
from projeto_1_eigen_faces import EigenFaces 

class FisherFaces():

    def __init__(self):

        self.__W = None
        self.__face_mean = None

 

    def calculate_W(self,X,y,m):

        #get images parameters of shape
        num_imgs=X.shape[0]
        imgs_original = np.copy(X)

        #m has to be <= num_imgs
        if ( m>= num_imgs):
            raise Exception("Value of m is superior or equal to number of images")

        #get width and height
        width = X.shape[1]
        height = X.shape[2]

        #get labels of y
        labels = np.unique(y)

        #reshape images
        X = np.reshape(X,(num_imgs,width*height))

        #Calculate mean face axis=0
        mean_faces= np.mean(X, axis = 0)
        self.__face_mean = mean_faces

        #calculate mean of each class in axis =0 , and puts it in an np array
        mean_labels = [np.mean(X[y == i], axis=0) for i in labels]

        #calculate Sb
        sB =self.calculate_sb(y,width,height,mean_faces, mean_labels)
        
        #calculate Sw
        sW = self.calculate_sw(X,y,width,height, mean_labels)
        #Calculate W considering all the imgs in the database
        eigen_classifier = EigenFaces()
        Wpca= eigen_classifier.calculate_W(imgs_original,m)
        
        #dot wpca.T sW, wPcA
        sW2 = np.dot(np.dot(Wpca.T, sW), Wpca)
        sb2 = np.dot(np.dot(Wpca.T, sB), Wpca)

        #c-1 “larger eigenvectors” from the matrix
        c_1 = np.dot(np.linalg.inv(sW2), sb2)

        #V
        eig_values, V = np.linalg.eig(c_1)

        #indices
        idx = np.argpartition(eig_values, -m)[-m:]
        indices = idx[np.argsort((-eig_values)[idx])]

        #Vs
        V = np.array([v[indices] for v in V])

        self.__W =  np.dot(Wpca,V)

        #return W
            
    
    def calculate_sb(self,y,width,height,mean_faces, mean_labels):
      
        sB = None
        labels = np.unique(y)
        for i in range(len(mean_labels)):
            #ni length das labels i
            len_label_i = np.sum(y==labels[i])

            #reshape data to (mean,wxh,1)
            aux_reshape_mean_labels = np.reshape(mean_labels[i], (width * height, 1))
            aux_reshape_mean_faces = np.reshape(mean_faces, (width*height,1))

            #subtract means
            sub_means = aux_reshape_mean_labels - aux_reshape_mean_faces

            mult_i = len_label_i*np.dot(sub_means,sub_means.T)

            if (i==0):
                sB = mult_i
            else:   
                sB = sB + mult_i
        
        return sB
    
    def calculate_sw(self,X,y,width,height, mean_labels):

        Sw = None

        for i in range (len(X)):
            
            #aux reshape, to add a 1 in the final dimension
            aux_reshape_image = np.reshape(X[i], (width*height,1))
            aux_reshape_mean_label = np.reshape(mean_labels[y[i]], (width * height, 1))
            #subtract each images by its label mean 
            sub_image_means = aux_reshape_image - aux_reshape_mean_label
            #multiply sub by the transposed image
            mult_i = np.dot(sub_image_means,sub_image_means.T)
    	    
            #Summatory
            if (i==0):
                Sw= mult_i
            else:
                Sw = Sw + mult_i
        
        return Sw
    

    def calculate_ys(self,imgs):

        ys =[]
        imgs = np.reshape(imgs,(imgs.shape[0],imgs.shape[1]*imgs.shape[2]))
        
        for i in range (len(imgs)):

            img_mean = imgs[i] - self.__face_mean
            y=np.real(np.dot(self.__W.T,img_mean))
            ys.append(y)

        return ys


  