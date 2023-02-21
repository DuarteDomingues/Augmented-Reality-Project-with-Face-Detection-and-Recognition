import cv2
import os


#paths
cascFacePath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
cascEyesPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_eye_tree_eyeglasses.xml"


#Cascade Classifiers
faceCascade = cv2.CascadeClassifier(cascFacePath)
eyesCascade = cv2.CascadeClassifier(cascEyesPath)

#video
video_capture = cv2.VideoCapture(0)

namePerson="Dudão"

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    '''
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    '''
    # Draw a rectangle around the faces
    i=0
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
      #  cv2.putText(frames,namePerson,(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
        i=i+1

        roiGray = gray[y:y+h, x:x+w]
        roiColor = frames[y:y+h, x:x+w]

        eyes = eyesCascade.detectMultiScale(roiGray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roiColor,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
