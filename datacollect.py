import cv2
import os


# Collect 500 test images 

#Select device to capture live feed
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0

#Name this Subject1, Subject2, etc
nameID=str(input("enter name of image: ")).lower()

path='Images/'+nameID

isExist = os.path.exists(path)


os.makedirs(path)

while True:

	#read in frames from video
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.3, 5)
	for x,y,w,h in faces:
		count=count+1
		name='./images/'+nameID+'/'+ str(count) + '.jpg'

		#create rectangle around fra,
		cv2.imwrite(name, frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.imshow("WindowFrame", frame)
	cv2.waitKey(1)

	#Number of images to take (adjust as needed)
	if count > 800:
		break

video.release()
cv2.destroyAllWindows()




# #Open video only, not taking photos

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# while True:
# 	ret,frame=video.read()
# 	faces=facedetect.detectMultiScale(frame,1.3, 5)
# 	for x,y,w,h in faces:
# 		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
# 	cv2.imshow("WindowFrame", frame)
# 	key = cv2.waitKey(1)
# 	if key == ord('q'):
# 		break
# video.release()
# cv2.destroyAllWindows()