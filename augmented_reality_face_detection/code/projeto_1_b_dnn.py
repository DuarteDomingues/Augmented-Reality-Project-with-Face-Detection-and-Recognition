import numpy as np
import cv2
from projeto_1_a_haar_classifier import Haar_Eyes



class Dnn():

	def __init__(self):
		protoFile = "./deploy.prototxt"
		caffeModel = "./res10_300x300_ssd_iter_140000.caffemodel"

		self.__detector = cv2.dnn.readNetFromCaffe(protoFile, caffeModel)
		self.__haar_eyes = Haar_Eyes()


#image = cv2.imread(imagePath)

	def get_detections(self,image):
		# load the input image and construct an input blob for the image
		# by resizing to a fixed 300x300 pixels and then normalizing it
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0,(224, 224), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and
		# predictions
	
		self.__detector.setInput(blob)
		detections = self.__detector.forward()
		return detections

	def classify_faces(self,detections,image):	
		
		faces_eyes=[]
		(h, w) = image.shape[:2]
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]).astype(np.int64)
				eyes = self.__haar_eyes.get_eyes(image, box)
				faces_eyes.append((box,eyes))
				

		return image,faces_eyes



		