# import the necessary packages
import numpy as np
import cv2

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread('toilet.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)

cv2.imshow("Original", image)
cv2.imshow("Edges", wide)
cv2.imshow("Blurred",blurred)
cv2.waitKey(0)

