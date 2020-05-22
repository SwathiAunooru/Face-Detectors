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
        
    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    start = time.time()

    # apply face detection (hog)
    faces_hog = hog_face_detector(image, 1)

    end = time.time()
    print("Execution Time (in seconds) :")
    print("HOG : ", format(end - start, '.2f'))

    # loop over detected faces
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # draw box over face
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    # write at the top left corner of the image
    start = time.time()
    # for color identification
    img_height, img_width = image.shape[:2]
    cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)

    # display output image
    cv2.imshow("face detection with dlib", image)
    cv2.waitKey(1)

    # save output image 
    cv2.imwrite("cnn_face_detection.jpg", image)

# close all windows
cv2.destroyAllWindows()