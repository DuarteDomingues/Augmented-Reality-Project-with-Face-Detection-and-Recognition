import math
import cv2
import numpy as np

def normalize(img,faces_eyes):
    img_return = np.empty((0))
    face=None
    for face_eyes in faces_eyes:
        (face_,eyes) = face_eyes
        face = face_
        
        if (eyes!=None):
            (center_x_1,center_y_1,center_x_2,center_y_2) = get_center_eyes(eyes)
            #cv2.circle(img, (center_x_1,center_y_1), radius=0, color=(0, 0, 255), thickness=5)
            #cv2.circle(img, (center_x_2,center_y_2), radius=0, color=(0, 0, 255), thickness=5)
            #cv2.line(img, (center_x_1, center_y_1),(center_x_2, center_y_2),(0, 0, 255), thickness=4)
            angle_rot,angle_orientation =  get_angle(center_x_1,center_y_1,center_x_2,center_y_2)
            img_rotated, rot_matrix = rotate(img, angle_rot, angle_orientation)

            rotated1 = np.matmul(rot_matrix, np.array([center_x_1, center_y_1, 1])).astype(int)
            rotated2 = np.matmul(rot_matrix, np.array([center_x_2, center_y_2, 1])).astype(int)
            #cv2.line(img_rotated, (rotated1[0], rotated1[1]), (rotated2[0], rotated2[1]),(0,0,0),3)
            #cv2.imshow('Rotation', img_rotated)

            scale = get_scale(rotated1,rotated2)
            #resize
            img_resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            #cv2.imshow('Rotation', img_resized)
            img_,img_legit = adjust_resize_to_face(img_resized,rotated1,rotated2,scale)

            if img_legit: 
               # cv2.imshow('Selected', img_)
                img_return = img_
			    #img_return = selected_img
      
        return img_return, face
			    


def get_center_eyes(eyes):

    #left eyes coords
    (ex1, ey1, ew1, eh1) = eyes[0]
	#right eye chords
    (ex2, ey2, ew2, eh2) = eyes[1]
    #left eye
    center_x_1 = ex1+ int(ew1/2)
    center_y_1 = ey1 + int(eh1/2)
    #right eye
    center_x_2 = ex2+ int(ew2/2)
    center_y_2 = ey2+ int(eh2/2)
    
    return center_x_1,center_y_1,center_x_2,center_y_2



def get_angle(center_x_1,center_y_1,center_x_2,center_y_2):

    dist_eyes = get_euclidean_dis(center_x_1,center_y_1,center_x_2,center_y_2)
    sinX = abs(center_x_1 - center_x_2)/dist_eyes

    angle_rot= math.degrees(math.asin(sinX))
    angle_orientation = False
    if (center_y_1>center_y_2):
        angle_orientation = True
    return angle_rot,angle_orientation

def rotate(img, angle, angle_orientation):

    center = (int(img.shape[0]/2), int(img.shape[1]/2))
    #decide if right or left
    new_angle = angle - 90 if angle_orientation else 90 - angle
    #get rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, new_angle, 1.)

	# rotates the image with a certain angle, it subtracts 90 because is when
	# its perfetly aligned
    img_rotated = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
	#cv2.imshow('Rotation', img_rotated)

    return img_rotated, rot_matrix

def get_euclidean_dis(x1,y1,x2,y2):

    x_new = (x1 - x2)**2
    y_new = (y1 - y2)**2

    dist = math.sqrt((x_new + y_new))
    return dist

def get_scale(rotated1,rotated2):

    dist = get_euclidean_dis(rotated1[0], rotated1[1], rotated2[0], rotated2[1])
    return 16/dist

def do_resize(img,scale):

    img_resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

def adjust_resize_to_face(img,rotated_eye_left,rotatede_eye_right,scale):

    scale_eye_left = rotated_eye_left*scale
    scale_eye_right = rotatede_eye_right*scale

    x0 = int(scale_eye_left[0]) -16
    y0 = int(scale_eye_right[1]) -24
    x1 = x0 + 46
    y1 = y0 + 56

    img = img[y0 : y1, x0 : x1]

    print(img.shape)
	
    img_exists = True

    if img.shape[0] < 56 or img.shape[1] < 46:
        img_exists=False
    
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img,img_exists





