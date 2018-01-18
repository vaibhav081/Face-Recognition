

import numpy as np
import cv2


cam = cv2.VideoCapture(0) 



face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')



data = [] 


ix = 0 

"""
an infinite loop
calling camera object to read a frame each time by using cam.read()
cam.read() function returns 2 values -> ret and frame
ret is a boolean value ; if camera is returning an object i.e. working properly ret is True else False
frame object should contain the input frame as a numpy matrix
every image is just a collection of 3 RGB components
1 matrix for each color
image is a collection of pixels
for each pixel there are 3 values RGB
combining them we get our image
assuming the ret variable is True we will convert the frame into greyscale cz open cv function for face recognition works for greyscale
"""
file=open("names.txt","a")
s=input('enter your name ')
file.write(s)
file.write('\n')
file.close()

while True :
	
	ret , frame = cam.read()


	if ret==True :
		
		gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) ##convert BGR image(current frame) to grayscale
		


		faces = face_cas.detectMultiScale(gray , 1.3 , 5)

		


		for (x , y , w , h) in faces :
			


			face_component = frame[y:y+h , x:x+w , :] # : means takes all the values from RGB
			
			
			fc = cv2.resize(face_component , (50,50))

			


			if ix%10 == 0 and len(data) < 50 :
				data.append(fc)

			

			cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 2)
		ix += 1 
		cv2.imshow('frame' , frame ) 
		

		"""
		waits for some input from the keyboard ; in every 1msec waits for an input ; if id of that input is 27 which stands for escape key
		or if we have collected 20 faces exit the code
		"""

		

		if cv2.waitKey(1) == 27   or len(data) >=50 : ##display the frame 
			break 

	else :

		
		print ("error")


cv2.destroyAllWindows()


data = np.asarray(data)


print (data.shape)


np.save(s,data) 

