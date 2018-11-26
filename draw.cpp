#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv) {

  Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  rectangle(frame, Point(65, 135), Point(120, 210), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(250, 160), Point(310, 225), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(375, 185), Point(440, 245), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(515, 180), Point(575, 245), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(645, 185), Point(705, 245), Scalar( 0, 0, 255 ), 2);

  rectangle(frame, Point(55, 250), Point(115, 315), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(195, 215), Point(250, 285), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(285, 240), Point(350, 310), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(430, 235), Point(485, 305), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(560, 250), Point(620, 315), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(675, 250), Point(735, 315), Scalar( 0, 0, 255 ), 2);

  imwrite( "annotated.jpg", frame );

  return 0;
}
