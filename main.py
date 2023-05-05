import cv2
from utils import PutSubtitle, Smooth, Grab
import numpy as np
import imutils

#read and resize all images and videos used in the project
cap = cv2.VideoCapture('vid/VID_20230321_133007.mp4')
basket= cv2.resize(cv2.imread("img/Basket.png"), (130, 130))
pokeball = cv2.resize(cv2.imread("img/poke.png"), (120, 120))
dragon = cv2.resize(cv2.imread("img/dragon.png"), (120, 120))
hasb = cv2.resize(cv2.imread("img/hasb.jpg"), (130, 130))
sun = cv2.VideoCapture('vid/sun.gif')
template = cv2.resize(cv2.imread("img/ss4.png"), (90,90))

#get frame per second and start at frame 0
fps = int(cap.get(cv2.CAP_PROP_FPS))

#output video
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (854, 480))

f=0

while(cap.isOpened()):
    #at second 25 we restart the video
    if f == 25*fps:
        cap.release()
        cap = cv2.VideoCapture('vid/VID_20230321_133007.mp4')
    #at second 35 we change video
    elif f == 35*fps:
        cap.release()
        cap = cv2.VideoCapture('vid/VID_20230320_102316.mp4')
        cap.set(cv2.CAP_PROP_POS_MSEC,7000)
    
    #we read a frame from current video
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (854, 480))
        #grayscale
        if f < 4*fps:
            if (f > 1*fps and f < 2*fps) or (f > 3*fps and f < 4*fps):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.merge((frame, frame, frame))
            
            frame = PutSubtitle(frame, "Switching between color and grayscale")
        #smoothing
        elif f < 8*fps:
            size = 2 * ((f -100) // 5) +1 
            frame = Smooth(frame,0,size)
            frame = PutSubtitle(frame, "Image Blurring using Gaussian filter with increasing kernel size.", org=(20,420))
            frame = PutSubtitle(frame, "This operation is done to remove noise from the image")
            #frame = PutSubtitle(frame, "Using Gaussian filter with increasing kernel size.")
        elif f < 12*fps:
            size = 2 * ((f -100) // 20) + 1
            frame = Smooth(frame,1,size)
            frame = PutSubtitle(frame, "Image Blurring using a Bilateral filter with increasing kernel size", org=(20,420))
            frame = PutSubtitle(frame, "This operation also reduces noise but tries to keep edges sharp")
        #grabbing
        elif f < 20*fps:
            _, frame = Grab(Smooth(frame,0,1), [24,50,0],[160,255,255], ksize=(3,3), ero=2, dil=7)
            frame = PutSubtitle(frame, "Object grabbing using threshold",color=(255,255,255), org=(50,400))
            frame = PutSubtitle(frame, "in red you can see points added by dilation",color=(255,255,255), org=(50,430))
            frame = PutSubtitle(frame, "in blue you can see points removed by erosion",color=(255,255,255), org=(50,460))
        #edge detection
        elif f < 25*fps:
            blackmask = np.zeros((480,854), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = ((f-560)//60)*2 + 1
            frame = Smooth(frame,0,5)
            xedges = cv2.convertScaleAbs(cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=size))
            yedges = cv2.convertScaleAbs(cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=size))
            #print(type(blackmask[0][0]), type(xedges[0][0]))
            frame = cv2.merge((yedges, blackmask, xedges))
            frame = PutSubtitle(frame, "Sobel edge detection with increasing kernel size", color=(255,255,255))
        #circle detection
        elif f < 35*fps:
            new_frame = Smooth(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),1,15)
            if f < 30*fps:
                circles = cv2.HoughCircles(new_frame, cv2.HOUGH_GRADIENT,1.1, 100000, param1=15, param2=15,minRadius=60,maxRadius=70)
                frame = PutSubtitle(frame, "Circle detection using Hough transform, Since I want to ", org=(20,420))
                frame = PutSubtitle(frame, "detect only the main circle I use strict parameters", )
            else:
                circles = cv2.HoughCircles(new_frame, cv2.HOUGH_GRADIENT,1.1, 10, param1=40, param2=10,minRadius=10,maxRadius=70)
                frame = PutSubtitle(frame, "Decreasing param1, param2 and minDist increases the " , org = (20, 420))
                frame = PutSubtitle(frame, "ammount of circles dected")
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw inner circle
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3) 
        #template matching    
        elif f < 37*fps:
            res = cv2.matchTemplate(frame,template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  
            threshold = 0.4
            if max_val >= threshold:
                templatew = template.shape[1]
                templateh = template.shape[0]
                top_left = max_loc
                bottom_right = (top_left[0] + templatew, top_left[1]+templateh)
                cv2.rectangle(frame, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv2.LINE_4)
            frame = PutSubtitle(frame, "Find the ball using template matching")
        elif f < 40*fps:
            background = np.zeros((480,854), dtype=np.uint8) 
            frame = cv2.matchTemplate(frame,template, cv2.TM_CCOEFF_NORMED)
            frame = cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            background[44:435, 44:809] = frame
            frame = background
            frame = cv2.merge((frame, frame, frame))
            frame = PutSubtitle(frame, "In this map white indicates the likelihood of the object being there",org=(10,450), color=(255,255,255),fontScale=0.7)
        #object replacement
        elif f< 60*fps:
            mask, _= Grab(frame, [18,60,50],[50,200,150], ksize=(3,3), ero=2, dil=15)
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            x,y,w,h = 0,0,0,0
            #using a basketball
            if f<= 44*fps:
                size = 130
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if x < 150:
                        continue
                    if (frame[y:y+size, x:x+size].shape == basket.shape):
                        img2gray = cv2.cvtColor(basket, cv2.COLOR_BGR2GRAY)
                        _, logo_mask = cv2.threshold(img2gray,0, 255, cv2.THRESH_BINARY)
                        logo_mask = cv2.dilate(logo_mask,(3,3), iterations = 2)
                        roi = frame[y:y+size, x:x+size]
                        roi[np.where(logo_mask)] = 0
                        roi += basket
            #using a pokeball
            elif f<= 48*fps:
                size = 120
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if x < 150:
                        continue
                    if (frame[y:y+size, x:x+size].shape == pokeball.shape):
                        img2gray = cv2.cvtColor(pokeball, cv2.COLOR_BGR2GRAY)
                        _, logo_mask = cv2.threshold(img2gray,0, 255, cv2.THRESH_BINARY)
                        roi = frame[y:y+size, x:x+size]
                        roi[np.where(logo_mask)] = 0
                        roi += pokeball
            #using a dragonball
            elif f<= 52*fps:
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if x < 150 or x >450:
                        continue
                    if (frame[y:y+size, x:x+size].shape == dragon.shape):
                        img2gray = cv2.cvtColor(dragon, cv2.COLOR_BGR2GRAY)
                        _, logo_mask = cv2.threshold(img2gray,1, 255, cv2.THRESH_BINARY)
                        roi = frame[y:y+size, x:x+size]
                        roi[np.where(logo_mask)] = 0
                        roi += dragon
            #using hasbhulla
            elif f<= 56*fps:
                size = 130
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if x < 150:
                        continue
                    x = x
                    y = y
                    if (frame[y:y+size, x:x+size].shape == hasb.shape):
                        img2gray = cv2.cvtColor(hasb, cv2.COLOR_BGR2GRAY)
                        _, logo_mask = cv2.threshold(img2gray,250, 255, cv2.THRESH_BINARY_INV)
                        roi = frame[y:y+size, x:x+size]
                        roi[np.where(logo_mask)] = 0
                        roi += hasb
            #using the sun
            elif f <= 61*fps:
                ret, bow = sun.read()
                if ret == False:
                    sun = cv2.VideoCapture('vid/sun.gif')
                    ret, bow = sun.read()
                size = 220
                sunimg = cv2.resize(bow, (size, size))
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if x < 150 or x >450:
                        continue
                    x = x- 50
                    y = y -50
                    if (frame[y:y+size, x:x+size].shape == sunimg.shape):
                        img2gray = cv2.cvtColor(sunimg, cv2.COLOR_BGR2GRAY)
                        _, logo_mask = cv2.threshold(img2gray,80, 255, cv2.THRESH_BINARY)
                        roi = frame[y:y+size, x:x+size]
                        roi[np.where(logo_mask)] = 0
                        roi += sunimg
            frame = PutSubtitle(frame, "Here I'm using grabbing to locate contours of the ball", org=(30,430), fontScale=0.9)
            frame = PutSubtitle(frame, "and then replace it with another image", org=(30,460), fontScale=0.9)
        out.write(frame)
        cv2.imshow("frame", frame)
        f+=1
        if (cv2.waitKey(25) & 0xFF == ord('q')) or (f > 60*fps):
          break
    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
