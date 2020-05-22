import cv2
import dlib
import argparse
import time


cap = cv2.VideoCapture(0)
while True:
    _,image = cap.read()
    # load input image
    if image is None:
        print("Could not read input image")
        exit()
        
    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    start = time.time()
    # apply face detection (hog)
    faces_cnn = cnn_face_detector(image, 1)

    end = time.time()
    print("CNN : ", format(end - start, '.2f'))

    # loop over detected faces
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

        # draw box over face
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

    # for color identification
    img_height, img_width = image.shape[:2]
    cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)

    # display output image
    cv2.imshow("face detection with dlib", image)
    cv2.waitKey(1)

    # save output image 
    cv2.imwrite("cnn_face_detection.jpg", image)

# close all windows
cv2.destroyAllWindows()