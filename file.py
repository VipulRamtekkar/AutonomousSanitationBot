import cv2
import numpy as np 
from time import sleep 

# capture an image from the pi-camera
image=cv2.VideoCapture()
cv2.imshow(image,"image")
cv2.waitKey(0)
