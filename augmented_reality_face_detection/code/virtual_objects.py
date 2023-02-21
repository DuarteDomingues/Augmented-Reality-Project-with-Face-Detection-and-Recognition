import cv2
import numpy as np

#load object
img_label_1 =  cv2.imread("./hats/hat.jpg", 1)
img_label_2 =  cv2.imread("./hats/hat2.jpg", 1)
img_label_3 = cv2.imread("./hats/hat3.jpg", 1)
img_label_4 =  cv2.imread("./hats/hat4.jpg", 1)


#print(img1)
def normalize_obj(face_coords,img, class_label):
    
    len_obj = int((face_coords[3] - face_coords[1])/ 1.55)
    obj_x = [face_coords[1] - len_obj+ 50, face_coords[1] +50]
    obj_y = [face_coords[0]-30 , face_coords[2]+30 ]

    #print("objx",obj_x)
    #print("objy",obj_y)

    if obj_x[0] > -1 and obj_x[1] > -1 and obj_y[0] > -1 and obj_y[1] > -1:

        img_cut = img[obj_x[0] :obj_x[1], obj_y[0] : obj_y[1]]
        
        img1= img_label_1
        if (class_label==0):
            img1 = img_label_1
        elif(class_label==2):
            img1 =img_label_2
        elif(class_label==3):
            img1 =img_label_3

        elif(class_label==4):
            img1 =img_label_4
        
        
        masked_img = do_mask(img_cut,img1)
        img[obj_x[0] :obj_x[1], obj_y[0] : obj_y[1]] = masked_img

        cv2.imshow('frame', img)
    else: 
        return None
    

def do_mask(img,img_obj):
    #mask  
   # print("img shape",img.shape)
 
    resize_img =  cv2.resize(img_obj, ((img.shape[1]), (img.shape[0])), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask_not = cv2.bitwise_not(mask)

    img_masked = cv2.bitwise_and(resize_img, resize_img, mask = mask)
		
    img1_bitwise_and = cv2.bitwise_and(img, img, mask = mask_not)

		
    final = cv2.add(img1_bitwise_and, img_masked)



    return final
