import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
from cryptography.fernet import Fernet


# facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cap=cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# font=cv2.FONT_HERSHEY_COMPLEX


# model = load_model('keras_model.h5', compile=False)


# def get_className(classNo):
# 	if classNo==0:
# 		return "chetan"
# 	elif classNo==1:
# 		return "next"

# while True:
# 	sucess, imgOrignal=cap.read()
# 	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
# 	for x,y,w,h in faces:
# 		crop_img=imgOrignal[y:y+h,x:x+h]
# 		img=cv2.resize(crop_img, (224,224))
# 		img=img.reshape(1, 224, 224, 3)
# 		prediction=model.predict(img)
# 		classIndex=np.argmax(model.predict(img), axis=-1)
# 		probabilityValue=np.amax(prediction)
# 		if classIndex==0:
# 			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
# 			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
# 			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
# 		elif classIndex==1:
# 			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
# 			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
# 			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

# 		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
# 	cv2.imshow("Result",imgOrignal)
# 	k=cv2.waitKey(1)
# 	if k==ord('q'):
# 		break


# cap.release()
# cv2.destroyAllWindows()






#decrypt version
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
from cryptography.fernet import Fernet

# Load your encryption key
def load_key():
    return open("encryption_key.key", "rb").read()

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5', compile=False)

def get_className(classNo):
    if classNo == 0:
        return "Subject1"
    elif classNo == 1:
        return "Subject2"

# Load the encryption key
key = load_key()

while True:
    success, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis=-1)
        probabilityValue = np.amax(prediction)

        # Draw rectangle around the face
        cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display person's name and likelihood percentage
        if classIndex == 0 or classIndex == 1:  # Can add more
            person_name = get_className(classIndex)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, person_name + " " + str(round(probabilityValue*100, 2)) + "%", 
                        (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Check if recognized person is correct and above threshold
            if probabilityValue >= 0.70:  # Adjust however needed
                # Decrypt the file
                fernet = Fernet(key)
                with open('locked_data.txt.encrypted', 'rb') as file:
                    encrypted_data = file.read()
                decrypted_data = fernet.decrypt(encrypted_data)
                decrypted_file_name = 'example_decrypted.txt'
                with open(decrypted_file_name, 'wb') as file:
                    file.write(decrypted_data)
                print(f"File '{decrypted_file_name}' decrypted successfully!")

    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













