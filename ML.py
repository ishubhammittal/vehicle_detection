import cv2

cap = cv2.VideoCapture('carv2.mp4')

car_cascade = cv2.CascadeClassifier('car.xml')

while True:
    ret, frames = cap.read()     
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    print(cars)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('ML assignment', frames )
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
