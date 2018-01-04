import cv2
import numpy as np
import time
import serial
import picamera
import serial
import struct

left_motion=0     #defining the times the motion has been done to remember the state of the vehicle  so that the front view can be restored 
right_motion=0
camera=picamera.PiCamera()

'''
arduino_motor = serial.Serial('com3',9600,timeout=1)
arduino_servo = serial.Serial('com4',9600,timeout=1)
'''
def toilet_detection(im):
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (9, 9), 0) 
	#gray1=cv2.bilateralFilter(gray,11,17,17)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	#edged = cv2.erode(edged, None, iterations=1)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
	dilated = cv2.dilate(edged, kernel)
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)
	if c.contourArea()>1000:
		return True
	else:
		return False 

def bot_motion(contours):

    if num_contours()>0:
        d=distance_estimate(c)
        if d>90:
            move_forward(d/2)
            move_forward(d/2)
        else:
            move_forward(d)
        move_forward_incr()
        move_backward()

    while (num_contours()==0):
        right_rotation(5)
        right_motion+=1

def calcenter(contour):
    cv2.moments(contour) #gives the information about centroid, moi, etc.
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center=(cx,cy)
    return center

def find_yellow(image): #returns the yellow colour stains
    hsv_roi =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask= cv2.inRange(hsv_roi, np.array([20,150,150]), np.array([30,255,255]))
    #ycr_roi=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    #mask_2=cv2.inRange(ycr_roi, np.array((0.,165.,165.)), np.array((255.,255.,255.)))
    #mask =cv2.bitwise_or(mask_1,mask_2)
    kern_dilate = np.ones((8,8),np.uint8)
    kern_erode  = np.ones((3,3),np.uint8)
    mask= cv2.erode(mask,kern_erode)    #Eroding
    mask=cv2.dilate(mask,kern_dilate)   #Dilating
    return mask

def contourDetect(gray):
    
    #gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(9,9),0)
    edge=cv2.Canny(gray,20,120)
    edge=cv2.dilate(edge,None,iterations=1)
    #edge=cv2.erode(edge,None,iterations=1)
    (__,contours, _) = cv2.findContours(edge.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
  
    return contours

def num_contours(contours):
    return str(len(contours))

def find_trash(image):
    hsv_roi =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask= cv2.inRange(hsv_roi, np.array([150,150,150]), np.array([170,255,255]))
    #ycr_roi=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    #mask_2=cv2.inRange(ycr_roi, np.array((0.,165.,165.)), np.array((255.,255.,255.)))
    #mask =cv2.bitwise_or(mask_1,mask_2)
    kern_dilate = np.ones((8,8),np.uint8)
    kern_erode  = np.ones((3,3),np.uint8)
    mask= cv2.erode(mask,kern_erode)    #Eroding
    mask=cv2.dilate(mask,kern_dilate)   #Dilating
    return mask

# initializing the camera
def startcam():
    camera.start_preview()
    time.sleep(1)
'''
def right_rotation(time_rot):                                                  
    arduino_motor.write(struct.pack('.2.0.0.1.'+ str(time_rot)+'.'))

def left_rotation(time_rot):
    arduino_motor.write(struct.pack('.2.0.0.1.'+ str(time_rot)+'.'))
                        
def move_forward(dist):                                   
    arduino_motor.write(struct.pack('.1.1.'+str(dist)+'.0.0.'))

def move_forward_incr():
    arduino_motor.write(struct.pack('.1.2.0.0.0.'))

def move_backward():                                            # .1.3.0.0.0
    arduino_motor.write(struct.pack('.1.3.0.0.0.'))

def lift_arm(angle):
                                                                            #Start the serial port to communicate with arduino 
    data.write(struct.pack('>B',angle))
                                                    #code and send the angle to the Arduino through serial port
def open_flap():
    
    pos =180 #open flap                      
    data.write(struct.pack('>B',pos))
    time.sleep(1)
    pos=90    #close flap
    data.write(struct.pack('>B',pos))
'''
def captureImg():                                                       # the entire patch goes in the while loop for iteration
    camera.capture('runimg.jpg')
    img = cv2.imread('runimg.jpg', 1)
    return img

def stopCam():                                                          # stoping the preview of the camera
    camera.stop_preview()

def open_water():
    arduino_servo.write(struct.pack('0'))

def distance_estimate(contour):                                         #takes in contours
    '''
    def find_marker(image):
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     gray = cv2.GaussianBlur(gray, (5, 5), 0)
     edged = cv2.Canny(gray, 35, 125)
     (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
     c = max(cnts, key = cv2.contourArea)
     cv2.imshow("edged",edged)
     return cv2.minAreaRect(c)
     '''

    def distance_to_camera(knownWidth, focalLength, perWidth):
    	return (knownWidth * focalLength) / perWidth

    marker=cv2.minAreaRect(contour)
    KNOWN_DISTANCE = 50
    KNOWN_WIDTH = 50

    #marker = find_marker(image)                                        #check for the camera calibaration
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    #for imagePath in IMAGE_PATHS:
    #image = cv2.imread(imagePath)
    #marker = find_marker(image)
    dist = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    
    return dist
    '''
    box = np.int0(cv2.cv.BoxPoints(marker))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fft" % (inches / 12),
     (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
     2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    '''
if __name__=="__main__":

    start_time=time.time()
    startcam()      # the stains code patch will run for about 420 seconds and then the second patch will start

    while ((time.time()-start_time) < 300):
        im=captureImg()
     #now the image has been captured which needs to be processed for the
        mask=find_yellow(im)                               #stain detection
        contours1=contourDetect(mask)
        c=max(contours1,cv2.contourArea())
        found=cv2.minAreaRect(c)
        #box=np.int0(cv2.boxPoints(found))
        bot_motion(contours)

     #the contour detection      
     #finding the center of the object detected
    while ((time.time()-start_time) < 300):
     #trash picking
        im=captureImg()
        mask=find_trash(im)
        contours=contourDetect(mask)
        c=max(contours,cv2.contourArea())
        found=cv2.minAreaRect(c)
        #box=np.int0(cv2.boxPoints(found))
        #center=calcenter(c)
        bot_motion(contours)
     #keep turning
     #Now the object has been detected
     #get the distance estimate
     #open the servo flap

    while ((time.time()-start_time) < 300):
     #toilet detection
     #find the contour of the toilet
     	im=captureImg()
    	val=toilet_detection(im)
    	if val: 
    		bot_motion(contours)
    	else:
    		while True:
    			right_rotation(5)
        		right_motion+=1
        		im=captureImg()
        		if toilet_detection(im):
        			break
         #keep turning
     #Now the object has been detected
     #get the distance estimate
    #to return to its initial posiiton the patch would be coming back directly
    #signal to take a 360degree turn and then move forward till the
    #sonar tells to stop, would partially call for the return
    stopCam()





















