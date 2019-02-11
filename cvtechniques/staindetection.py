# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])
dim=(480,480)
image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)

# define the list of boundaries

l1=[17, 0, 0]
u1=[128, 65, 200]

# create NumPy arrays from the boundaries
lower = np.array(l1, dtype = "uint8")
upper = np.array(u1, dtype = "uint8")
 
# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
gray=cv2.bilateralFilter(gray,11,11,17)
gray1 = cv2.GaussianBlur(gray, (9, 9), 0) 

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged1 = cv2.Canny(gray1, 20, 120)
edged1 = cv2.dilate(edged1, None, iterations=1)
edged1 = cv2.erode(edged1, None, iterations=1)

(cnts, _) = cv2.findContours(edged1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key = cv2.contourArea)

found=cv2.minAreaRect(c)
box = np.int0(cv2.cv.BoxPoints(found))
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
dim=(300,300)
image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)

cv2.imshow("image",image)
cv2.imshow("edged1",edged1)
cv2.imshow("blurred",gray1)
cv2.waitKey(0)


