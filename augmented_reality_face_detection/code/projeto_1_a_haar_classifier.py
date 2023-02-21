import cv2
import os


class Haar_Eyes():

    def __init__(self):
        self.__cascEyesPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_eye_tree_eyeglasses.xml"

    def  get_eyes(self,frames,face_square) :

        
        #Cascade Classifiers
        eyesCascade = cv2.CascadeClassifier(self.__cascEyesPath)

        eyes = eyesCascade.detectMultiScale(frames)
        eyes_detected=[]
        for (ex,ey,ew,eh) in eyes:
            if ex >= face_square[0] and ey >= face_square[1] and ex+ew <= face_square[2] and ey+eh <= face_square[3]:
                eyes_detected.append((ex,ey,ew,eh))
                #caso detete 2 olhos
                if (len(eyes_detected) >1 and eyes_detected!=None):
                
                    (ex1,ey1,ew1,eh1) =  eyes_detected[0]
                    (ex2,ey2,ew2,eh2) = eyes_detected[1]
		            #(ex2,ey2,ew2,eh2) =  eyes_detected[1]

                    if (ex1 > ex2):
                    
                        eyes_fixed = [eyes_detected[1],eyes_detected[0]]
                        return eyes_fixed
                    else:
                        return eyes_detected


                #return eyes_detected
            #caso detete 1 olho
        return None

