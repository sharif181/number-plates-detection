import cv2

object = cv2.CascadeClassifier('resource/haarcascade_russian_plate_number.xml')

img = cv2.imread('resource/Cars1.png')
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plates = object.detectMultiScale(grayImg,1.1,4)
print(plates)
for (x,y,w,h) in plates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,233,0),3)
cv2.imshow('Output',img)
cv2.waitKey(0)