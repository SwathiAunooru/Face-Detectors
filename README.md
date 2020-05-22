# Face-Detectors
Performing face detection using  different techniques and understanding their performance for using them according to our requirement.

1. FACE DETECTOR USING HAAR CASCADE
2. FACE DETECTOR USING HOG
3. FACE DETECTOR USING DNN
4. FACE DETECTOR USING CNN


__FACE DETECTOR USING HAAR CASCADE:__

In the Haar cascade model ,First we need to load the required XML classifiers. Then load our input image (or video) in grayscale mode.Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h).
Cascade classifiers are trained on a few hundred sample images of image that contain the object we want to detect, and other images that do not contain those images. And a positive image contain features like, a dark eye region compared to upper-cheeks and a bright nose bridge region compared to the eyes
There are built in cascade classifiers for face ,eyes and we can even train our own model using cascade classifier training for example for car , planes etc.

__pros :__ 
It Works almost real-time on CPU having simple architecture and  Detects faces at different scales and can only detect frontal faces.

__Cons :__ 
The major drawback of this method is that it gives a lot of False predictions.Doesn’t work on non-frontal images.And also it doesn’t work under occlusion


__FACE DETECTOR USING HOG:__
HOG (Histogram of Oriented Gradients) model is built out of 5 HOG filters – front looking, left looking, right looking, front looking but rotated left, and a front looking but rotated right.Here, we first load the face detector. Then we pass it the image through the detector. The second argument is the number of times we want to upscale the image. The more you upscale, the better are the chances of detecting smaller faces. However, upscaling the image will have substantial impact on the computation speed. The output is in the form of a list of faces with the (x, y) coordinates of the diagonal corners.

__Pros :__
Fastest method on CPU . Works very well for frontal and slightly non-frontal faces .Light-weight model as compared to the other three. Works under small occlusion

__Cons :__
The major drawback is that it does not detect small faces as it is trained for minimum face size of 80×80. Thus, you need to make sure that the face size should be more than that in your application. You can however, train your own face detector for smaller sized faces.The bounding box often excludes part of forehead and even part of chin sometimes.Does not work very well under substantial occlusionDoes not work for side face and extreme non-frontal faces, like looking down or up.

__FACE DETECTOR USING DNN__
It is a Deep Neural Network based Face detection, we use the caffemodel and prototxt files. Here we first initialise the model using cv2.dnn.readNetFromCaffe() function, then the image is converted to a blob and passed through the network using the forward() function. The output detections is a 4-D matrix, where Pros Most accurate out of the four methods Runs at real-time on CPU .And works for different face orientations – up, down, left, right, side-face etc. Works even under substantial occlusion Detects faces across various scales ( detects big as well as tiny faces).

__Pros:__
DNN based detector overcomes all the drawbacks of Haar cascade based detector, without compromising on any benefit provided by Haar. We could not see any major drawback for this method except that it is slower than the Dlib HoG based Face Detector.

__FACE DETECTOR USING CNN:__
The CNN based detector detects faces at odd angles (i.e., non-frontal) which the HOG based detector might fail to detect. CNN based detector is computationally heavy and is not suitable for real-time video at the moment. This method uses a Maximum-Margin Object Detector (MMOD ) with CNN based features. The training process for this method is very simple and you don’t need a large amount of data to train a custom object detector. The model can be downloaded from the dlib-models repository. It uses a dataset manually labeled by its Author, Davis King, consisting of images from various datasets like ImageNet, PASCAL VOC, VGG, WIDER, Face Scrub. It contains 7220 images. 

__Pros :__
Works for different face orientations and robust to occlusion
Works very fast on GPU  and has a very easy training process.

__Cons:__
Very slow on CPU .Does not detect small faces as it is trained for minimum face size of 80×80. Thus, you need to make sure that the face size should be more than that in your application. However we can, train our own face detector for smaller sized faces.The bounding box is even smaller than the HoG detector.
 





 
