import numpy as np
import cv2
import os
from sklearn import svm

w = 50
h = 100
c = 3

X_data = np.zeros((100, w * h * c), dtype=np.uint8)
Test_data = np.zeros((12, w * h * c), dtype=np.uint8) #change the first value if you have a different amount of pictures
Y_data = np.zeros((100,), dtype=np.uint8)
eye = np.zeros((1, w * h * c), dtype=np.uint8)

with open('filename.txt', 'r') as f:
    index = 0
    for r in f:
        p, c = r.strip().split()
        #print len(r.strip().split())
        Y_data[index] = int(c)
        img = cv2.imread(os.path.join("Training",p))
        print os.path.join("Training",p)
        cv2.imshow('Training before resize', img)
        nimg = cv2.resize(img,(h,w))
        cv2.imshow('Training after resize', nimg)
        X_data[index] = nimg.flatten()
        index += 1

#start of svm, change C value to get best result
best = 0
bestc = 1
for i in xrange(0, 7):
    clf = svm.SVC(kernel='linear', C=10**i)
    clf.fit(X_data, Y_data)
    correct = 0
    for j in xrange(X_data.shape[0]):
        r = clf.predict(X_data[j].reshape(1,-1))
        r = r[0]
        if r == Y_data[j]:
            correct += 1
    print i, correct
    if correct > best:
        best = correct
        bestc = 10**i

bestsvm = svm.SVC(kernel='linear', C=bestc) #change this C value
bestsvm.fit(X_data, Y_data)

face_cascade = cv2.CascadeClassifier("cascade/mallick_haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascade/haarcascade_eye.xml")

cv2.destroyAllWindows()

img = cv2.VideoCapture(0)

while(True):
	open, frame = img.read()
	if open == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                print ex,ey,ew,eh
                neye = roi_color[ey:ey+eh,ex:ex+ew]
                cv2.imshow('eye before resize',neye)
                #neye = roi_color[ey:ey+eh,ex:ex+ew] #trying to resize image so its more on the eye and not the face at all
                neye = cv2.resize(neye,((ex+ew)/2, (ey+eh)/2)) # trying to get this to zoom in on the eye to get better prediction 
                cv2.imshow('eye', neye)
                neye = neye.flatten()
                eye[0] = neye.shape[0]
                prediction = bestsvm.predict(eye.reshape(1,-1)) #what do I put in this, I get an error with eyes
                print prediction
                if int(prediction) == 0:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)#No Retinoblastoma, just show its detecting a normal eye
                else:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)#there is Retinoblastoma, create a red square around the eye
                cv2.imshow('image',roi_color)
                key = cv2.waitKey(1)
                print key
                if (key==27):
                    exit() #this isnt working 
       
        
img.release()
cv2.destroyAllWindows()