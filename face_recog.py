import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()


font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale=1
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
cap = cv2.VideoCapture(0)
recognizer.read("trainer/trainer.yml")
id=0
while 1:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.2,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        print (id,conf)
        if(conf>50):
            if(id==1):
                id="Rishabh"
            elif(id==2):
                id= "Pranjal"
            elif(id==3):
                id= "Himanshu"

        else:
            id="Unknown"
        cv2.putText(img,str(id),(x,y+h),font, fontScale,(0,0,255))
    cv2.imshow("Face",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()