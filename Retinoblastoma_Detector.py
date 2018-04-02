import numpy as np
import cv2
import os
from sklearn import svm

#images are not exactly the same size, is 100 good for height and 50 for the width
w = 50
h = 100
c = 3

X_data = np.zeros((100, w * h * c), dtype=np.uint8)
Test_data = np.zeros((9, w * h * c), dtype=np.uint8)
Y_data = np.zeros((100,), dtype=np.uint8)

with open('filename.txt', 'r') as f:
    index = 0
    for r in f:
        p, c = r.strip().split()
        #print len(r.strip().split())
        Y_data[index] = int(c)
        img = cv2.imread(os.path.join("Training",p))
        print os.path.join("Training",p)
        cv2.imshow('Training before resize', img)
        cv2.waitKey(.1)  #change this numbers for the speed
        nimg = cv2.resize(img,(h,w))
        cv2.imshow('Training after resize', nimg)
        cv2.waitKey(.1)  #change this numbers for the speed
        X_data[index] = nimg.flatten()
        index += 1

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


path = "Test" #if this doesn't work than put the entire path name in (example: /Users/NAME/Desktop/Test )
index = 0
for i in os.listdir(path):
    test_img = cv2.imread(os.path.join("Test",i))
    print os.path.join("Test",i)
    cv2.imshow('Test before resize', test_img)
    cv2.waitKey(1)  #speed of images on screen
    test_nimg = cv2.resize(test_img,(h,w))
    cv2.imshow('Test after resize', test_nimg)
    cv2.waitKey(1)  #speed of images on screen
    Test_data[index] = test_nimg.flatten()
    index += 1

for t in xrange(Test_data.shape[0]):
    print t
    print "prediction",bestsvm.predict(Test_data[t].reshape(1,-1))

'''    
correct = 0
with open('accuracy.txt', 'r') as f:
    index = 0
    for r in f:
        p, c = r.strip().split()
        prediction = bestsvm.predict(Test_data[index].reshape(1,-1))
        #prediction = prediction.strip('[')
        #print predictions
        if int(c) == int(prediction):
            correct += 1
        else:
            print "wrong"
        index += 1
        

accuracy = float(correct) / index + 1
print "accuracy",accuracy
cv2.destroyAllWindows()
'''