import numpy as np
import cv2
import os
from sklearn import svm
import pickle

w = 50
h = 100
c = 3

X_data = np.zeros((100, w * h * c), dtype=np.uint8)
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
        nimg = cv2.resize(img,(h,w))
        X_data[index] = nimg.flatten()
        index += 1

#start of svm, changes C value to get best result
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

#add picture here
img = cv2.imread('image.jpg')
while(True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            neye = roi_color[ey:ey+eh,ex:ex+ew]
            cv2.imshow('eye',neye)
            cv2.waitKey(1000)
            neye = neye.flatten()
            eye[0] = neye.shape[0]
            prediction = bestsvm.predict(eye.reshape(1,-1))
            print "prediction", prediction
            if int(prediction) == 0:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)#No Retinoblastoma, just show its detecting a normal eye
            else:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)#there is Retinoblastoma, create a red square around the eye
    cv2.putText(roi_color,"If there is a red box around ",(0,200), 1, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(roi_color,"an eye, double check if it ",(0,220), 1, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(roi_color,"has retinoblastoma",(0,240), 1, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("image", roi_color)
    example = cv2.imread('Retinoblastoma_example_image.jpg')
    cv2.imshow("Retinoblastoma example", example)
    cv2.waitKey(10000) #wait time after the boxes are made around the eyes
    break
print "Thanks for running my program, I hope it helped!"
cv2.destroyAllWindows()