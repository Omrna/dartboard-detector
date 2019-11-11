# dartboard-detector - Image Processing and Computer Vision

This assignment you experiments with the Viola-Jones object detection framework provided by OpenCV, before combining it with Hough Transform techniques in order to get better results in detecting dartboards in images. 

The Viola Jones detector produces bounding boxes for where it thinks there are dartboards. However these detections tend to be overfitted, producing many false positives. The Hough Transform is used to detect lines and circles within these bounding boxes to make a decision on whether or not it contains a dartboard.

These detections are used along with the Hough Transform detecting lines and circles,

## Instructions ##

Compile using:

```
g++ dartboardDetector.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 -lopencv_imgproc -lopencv_objdetect
```

Then run the output file with a chosen image

```
./a.out dart5.jpg
```
