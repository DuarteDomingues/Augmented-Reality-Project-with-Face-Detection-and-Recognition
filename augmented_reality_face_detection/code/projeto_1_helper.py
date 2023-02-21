import os
import cv2
import numpy as np
import random

class Helper():


    def load_images(self,folder,label):
        y=[]
        imgs=[]
        for file_i in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,file_i), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
                y.append(label)
        return np.asarray(imgs),y
    
    def get_indexes_remove(self,X):
        indx_arr = np.zeros(len(X))
        for i in range(len(indx_arr)):
            indx_arr[i] = i
        
        return indx_arr
    
    def remove_values(self,X,y,n):
        xcopy = X.copy()
        
        arr_idxs = self.get_indexes_remove(X)
        arr_idx = self.delete_random_elems(arr_idxs, n)
       # X = np.delete(arr,)
        y=np.delete(y,arr_idx)
        return X,y

        

    
    def delete_random_elems(self,input_list, n):
        to_delete = set(random.sample(range(len(input_list)), n))
        return to_delete