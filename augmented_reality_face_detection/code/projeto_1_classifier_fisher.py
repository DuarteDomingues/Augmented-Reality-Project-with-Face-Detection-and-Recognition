import cv2
import numpy as np
from projeto_1_fisher_faces import FisherFaces
from sklearn.neighbors import KNeighborsClassifier


class NeighborClassifierFisher():


    def __init__(self,n_neighbors=3):

        self.__neighbor_classifier_fisher = KNeighborsClassifier(n_neighbors)
        self.__fisher = FisherFaces()


   
    def fit_nearest_neighbor(self,X,y,m):

        self.__fisher.calculate_W(X,y,m)
        feature_vectors = self.__fisher.calculate_ys(X)

        feature_vectors = np.array(feature_vectors)
        self.__neighbor_classifier_fisher.fit(feature_vectors,y)
    

    def predict(self,imgs_predict):

        feature_vector = self.__fisher.calculate_ys(imgs_predict)
        predicted_labels = self.__neighbor_classifier_fisher.predict(feature_vector)

        return predicted_labels
	   








	    