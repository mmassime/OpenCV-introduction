import cv2
import numpy as np

def Smooth(img, typ, size):
    if typ == 0: #gaussian
        return cv2.GaussianBlur(img, (size,size), 0)
    else: 
        return cv2.bilateralFilter(img, size, 200,200)

def PutSubtitle(img, text, org=(20,450), font= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2):
    #putText with predefined parameters
    return cv2.putText(img, text, org, font, fontScale, color, thickness)

def Grab(img, lower, upper, ksize=(3,3), ero=0, dil=0):
    #grab the image using tresholding, erosion and dilation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones(ksize, np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=ero)
    dilation = cv2.dilate(erosion, kernel, iterations=dil)
    improvementMask = cv2.merge((mask,cv2.bitwise_and(mask,dilation),dilation))
    #return the actual mask and the mask with different colors for improvements
    return dilation, improvementMask
