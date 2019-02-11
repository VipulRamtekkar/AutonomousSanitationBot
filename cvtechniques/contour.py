from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("toilet19.jpg")
dim=(300,300)
image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray1 = cv2.GaussianBlur(gray, (9, 9), 0) 
gray1=cv2.bilateralFilter(gray,11,17,17)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged1 = cv2.Canny(gray1, 50, 100)
edged1 = cv2.dilate(edged1, None, iterations=1)
edged1 = cv2.erode(edged1, None, iterations=1)

(cnts, _) = cv2.findContours(edged1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key = cv2.contourArea)

found=cv2.minAreaRect(c)
box = np.int0(cv2.cv.BoxPoints(found))
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

cv2.imshow("image",image)
cv2.imshow("edged1",edged1)
cv2.imshow("blurred",gray1)
cv2.waitKey(0)

