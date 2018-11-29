#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void threshold(cv::Mat &image);

int main() {

  Mat image;
  image = imread( "magnitude.jpg", 1);

  Mat thresholdImg = image;

  threshold(thresholdImg);
}

void threshold(cv::Mat &image) {
  cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);


  for (int y = 0; y < image.rows; y++) {
    for (int x = 0; x < image.cols; x++) {

      int val = 0;
      val = image.at<uchar>(y, x);

      // image.at<double>(y, x) = val;

      std::cout << val;
      if (val > 150) image.at<uchar>(y, x) = (uchar)255;
      else image.at<uchar>(y, x) = (uchar)0;
    }
  }

  // std::cout << image;
  // cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);

  imwrite("thresholded.jpg", image);
}
