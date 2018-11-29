#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void convolutionDX(
	cv::Mat &input,
	int size,
	cv::Mat &output
);

void convolutionDY(
	cv::Mat &input,
	int size,
	cv::Mat &output
);

void getMagnitude(
  cv::Mat &dfdx,
  cv::Mat &dfdy,
	cv::Mat &output
);

void getDirection(
  cv::Mat &dfdx,
  cv::Mat &dfdy,
	cv::Mat &output
);

void getThresholdedImg(
	cv::Mat &input,
	cv::Mat &output
);

int main() {

  Mat image;
  image = imread( "dart.bmp", 1 );

  namedWindow( "Original Image", CV_WINDOW_AUTOSIZE );
  imshow( "Original Image", image );

  cvtColor( image, image, CV_BGR2GRAY );

  Mat dfdx;
  dfdx.create(image.size(), CV_64F);

  Mat dfdy;
  dfdy.create(image.size(), CV_64F);

  Mat gradientMagnitude;
  gradientMagnitude.create(image.size(), CV_64F);

  Mat gradientDirection;
	gradientDirection.create(image.size(), CV_64F);

	Mat thresholdedImg;
	thresholdedImg.create(image.size(), CV_64F);

  convolutionDX(image, 3, dfdx);
  convolutionDY(image, 3, dfdy);

	getMagnitude(dfdx, dfdy, gradientMagnitude);
	getDirection(dfdx, dfdy, gradientDirection);

	getThresholdedImg(gradientMagnitude, thresholdedImg);

  return 0;
}

void convolutionDX(cv::Mat &input, int size, cv::Mat &output) {

  // Create convolution kernel
  cv :: Mat kernel(size, size, CV_64F);

  int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

  // Set kernel values
	for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
	  for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
      if ((n + kernelRadiusY) == 0) {
        kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = (double)-1 / (size * size);
      }
      else if ((n + kernelRadiusY) == 1) {
        kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = 0;
      }
      else {
        kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = (double)1 / (size * size);
      }
			kernel.at<double>(1, 0) = (double)-2 / (size * size);
			kernel.at<double>(1, 2) = (double) 2 / (size * size);
    }
  }



  // Create padded version of input
  cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

  // Time to do convolution
  for (int i = 0; i < input.rows; i++) {
     for (int j = 0; j < input.cols; j++) {

       double sum = 0.0;

       for(int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
         for(int n = -kernelRadiusY; n <= kernelRadiusY; n++) {

           // find the correct indices we are using
 					int imagex = i + m + kernelRadiusX;
 					int imagey = j + n + kernelRadiusY;
 					int kernelx = m + kernelRadiusX;
 					int kernely = n + kernelRadiusY;

 					// get the values from the padded image and the kernel
 					int imageval = (int) paddedInput.at<uchar>( imagex, imagey );
 					double kernalval = kernel.at<double>( kernelx, kernely );

 					// do the multiplication
 					sum += imageval * kernalval;
         }
       }
       output.at<double>(i, j) = sum;
    }
  }

	Mat img;
	img.create(input.size(), CV_64F);
	// Normalise to avoid out of range and negative values
	cv::normalize(output, img, 0, 255, cv::NORM_MINMAX);

  //Save thresholded image
  imwrite("dfdx.jpg", img);
}

void convolutionDY(cv::Mat &input, int size, cv::Mat &output) {

  // Create convolution kernel
  cv :: Mat kernel(size, size, CV_64F);

  int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

  // Set kernel values
	for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
	  for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
      if ((m + kernelRadiusX) == 0) {
        kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = (double)-1 / (size * size);
      }
      else if ((m + kernelRadiusX) == 1) {
        kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = 0;
      }
      else {
        kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = (double)1 / (size * size);
      }
			kernel.at<double>(0, 1) = (double)-2 / (size * size);
			kernel.at<double>(2, 1) = (double) 2 / (size * size);
    }
  }

  // Create padded version of input
  cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

  // Time to do convolution
  for (int i = 0; i < input.rows; i++) {
     for (int j = 0; j < input.cols; j++) {

       double sum = 0.0;

       for(int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
         for(int n = -kernelRadiusY; n <= kernelRadiusY; n++) {

           // find the correct indices we are using
 					int imagex = i + m + kernelRadiusX;
 					int imagey = j + n + kernelRadiusY;
 					int kernelx = m + kernelRadiusX;
 					int kernely = n + kernelRadiusY;

 					// get the values from the padded image and the kernel
 					int imageval = (int) paddedInput.at<uchar>( imagex, imagey );
 					double kernalval = kernel.at<double>( kernelx, kernely );

 					// do the multiplication
 					sum += imageval * kernalval;
         }
       }
       output.at<double>(i, j) = sum;
    }
  }

	Mat img;
	img.create(input.size(), CV_64F);
	// Normalise to avoid out of range and negative values
	cv::normalize(output, img, 0, 255, cv::NORM_MINMAX);

  //Save thresholded image
  imwrite("dfdy.jpg", img);
}

void getMagnitude(cv::Mat &dfdx, cv::Mat &dfdy,	cv::Mat &output) {
  for (int y = 0; y < output.rows; y++) {
    for (int x = 0; x < output.cols; x++) {

      double dxVal = 0.0;
      double dyVal = 0.0;
      double magnitudeVal = 0.0;

      dxVal = dfdx.at<double>(y, x);
      dyVal = dfdy.at<double>(y, x);

			// Calculate magnitude
      magnitudeVal = sqrt(pow(dxVal, 2) + pow(dyVal, 2));

			output.at<double>(y, x) = magnitudeVal;
    }
  }

	Mat img;
	img.create(dfdx.size(), CV_64F);

	cv::normalize(output, img, 0, 255, cv::NORM_MINMAX);

  imwrite("magnitude.jpg", img);
}

void getDirection(cv::Mat &dfdx, cv::Mat &dfdy,	cv::Mat &output) {
	for (int y = 0; y < output.rows; y++) {
    for (int x = 0; x < output.cols; x++) {

			double dxVal = 0.0;
      double dyVal = 0.0;
      double gradientVal = 0.0;

      dxVal = dfdx.at<double>(y, x);
      dyVal = dfdy.at<double>(y, x);

			// Calculate direction
      gradientVal = atan2(dyVal, dxVal);

			output.at<double>(y, x) = gradientVal;
    }
  }

	Mat img;
	img.create(dfdx.size(), CV_64F);

	cv::normalize(output, img, 0, 255, cv::NORM_MINMAX);

	imwrite("direction.jpg", img);
}

void getThresholdedImg(cv::Mat &input, cv::Mat &output) {
	Mat img;
	img.create(input.size(), CV_64F);

	cv::normalize(input, img, 0, 255, cv::NORM_MINMAX);

	for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {

      double val = 0;
      val = img.at<double>(y, x);


      if (val > 80) output.at<double>(y, x) = 255.0;
      else output.at<double>(y, x) = 0.0;
    }
  }

  imwrite("thresholded.jpg", output);
}
