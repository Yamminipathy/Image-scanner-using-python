from cv2 import cv2
import numpy as np
import mapper
from PIL import Image, ImageEnhance

image= cv2.imread('s.jpg')
image=cv2.resize(image,(1300,800))
ori=image.copy()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Title',gray)

kernel = np.ones((1, 1), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
gray = cv2.erode(gray, kernel, iterations=1)


blurred=cv2.GaussianBlur(gray,(5,5),0)
#cv2.imshow('Blurred',blurred)

edged=cv2.Canny(blurred,30,50)
#cv2.imshow('Canny',edged)

contours,heirarchy=cv2.findContours(edged ,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea,reverse=True)

for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.04*p,True)

    if len(approx)==4:
        target = approx
        break
approx=mapper.mapp(target)
pts=np.float32([ [0,0],[800,0],[800,800],[0,800] ])

op=cv2.getPerspectiveTransform(approx,pts)
dst=cv2.warpPerspective(ori,op,(800,800))

cv2.imshow('Scanned Copy',dst)
cv2.waitKey(0)