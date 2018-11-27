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

  rectangle(frame, Point(460, 220), Point(555, 320), Scalar( 0, 0, 255 ), 2);
  rectangle(frame, Point(730, 190), Point(830, 295), Scalar( 0, 0, 255 ), 2);

  imwrite( "annotated.jpg", frame );

  return 0;
}
