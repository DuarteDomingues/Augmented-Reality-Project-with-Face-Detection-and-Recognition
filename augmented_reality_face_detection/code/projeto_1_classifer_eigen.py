import cv2
import numpy as np
from projeto_1_eigen_faces import EigenFaces 
from sklearn.neighbors import KNeighborsClassifier


class NeighborClassifierEigen():


    def __init__(self,n_neighbors=3):

        self.__neighbor_classifier_eigen = KNeighborsClassifier(n_neighbors)
        self.__eigen = EigenFaces()


    def fit_nearest_neighbor(self,X,y,m):

        self.__eigen.calculate_W(X,m)
        feature_vectors = self.__eigen.calculate_ys(X)

        feature_vectors = np.array(feature_vectors)
        self.__neighbor_classifier_eigen.fit(feature_vectors,y)
    

    def predict(self,imgs_predict):

        feature_vector = self.__eigen.calculate_ys(imgs_predict)
        predicted_labels = self.__neighbor_classifier_eigen.predict(feature_vector)

        return predicted_labels







	    