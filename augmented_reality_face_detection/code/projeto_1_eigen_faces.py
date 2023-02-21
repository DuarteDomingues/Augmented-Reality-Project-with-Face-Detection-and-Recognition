import os
import cv2
import numpy as np

class EigenFaces():

    def __init__(self):

        self.__W = None
        self.__face_mean = None

    def do_face_normalization(self, imgs):
        #calculate mean face, with all images in dataset
        face_mean = np.mean(imgs,axis=0)
       
        return face_mean
    
    def calculate_W(self,imgs, m):
                
        #get images parameters of shape
        num_imgs=imgs.shape[0]

        #m has to be <= num_imgs
        if ( m>= num_imgs):
            raise Exception("Value of m is superior or equal to number of images")
        

        width=imgs.shape[1]
        height=imgs.shape[2]

        #reshape images
        imgs = np.reshape(imgs,(num_imgs,width*height))

        #media
        face_mean = self.do_face_normalization(imgs)
        self.__face_mean = face_mean

        #Calculate A
        A = np.array([face - face_mean for face in imgs ])
        A=A.T

        #Calculate R
        R = np.dot(A.T, A)

        #V
        eig_values, V = np.linalg.eig(R)

        #indices
        idx = np.argpartition(eig_values, -m)[-m:]
        indices = idx[np.argsort((-eig_values)[idx])]

        #Vs
        V = np.array([v[indices] for v in V])

        #Calculate Ws, norm in axis=0
        W = np.dot(A,V)
        W = W /  np.linalg.norm(W, axis = 0)   

        self.__W = W
        
        
        

        return W
    
    def calculate_ys(self,imgs):

        ys =[]

        imgs = np.reshape(imgs,(imgs.shape[0],imgs.shape[1]*imgs.shape[2]))

        for i in range (len(imgs)):

            img_mean = imgs[i] - self.__face_mean
            y=np.dot(self.__W.T,img_mean)
            ys.append(y)


        return ys

    