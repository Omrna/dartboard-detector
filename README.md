# dartboard-detector - Image Processing and Computer Vision

This assignment you experiments with the Viola-Jones object detection framework provided by OpenCV, before combining it with Hough Transform techniques in order to get better results in detecting dartboards in images. 

The Viola Jones detector produces bounding boxes for where it thinks there are dartboards. However these detections tend to be overfitted, producing many false positives. The Hough Transform is used to detect lines and circles within these bounding boxes to make a decision on whether or not it contains a dartboard.

<img width="815" src="https://user-images.githubusercontent.com/15062683/68760374-9c998300-0609-11ea-9c35-0e5afbefd056.png">

## Algorithm

<img width="983" src="https://user-images.githubusercontent.com/15062683/68760665-4aa52d00-060a-11ea-97f5-f4b93db53481.png">

## Performance

The algorithm produced a strong F1 score of 0.91, where all dartboards were corrected in the test images apart from one. Some false positives did occur, due to noise and the reliance on the Viola Jones detector.

<img width="622" alt="Screenshot 2019-11-13 at 11 48 51" src="https://user-images.githubusercontent.com/15062683/68761232-92788400-060b-11ea-8bfc-bee76771252b.png">

## Instructions ##

Compile using:

```
g++ dartboardDetector.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 -lopencv_imgproc -lopencv_objdetect
```

Then run the output file with a chosen image (e.g. dart0 to dart15.jpg)

```
./a.out dart5.jpg
```
