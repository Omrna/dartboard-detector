#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

#define PI 3.14159265

using namespace cv;

std::vector<double> rhoValues;
std::vector<double> thetaValues;

void convolution(	Mat &input,	int size,	int direction,	Mat kernel,	Mat &output );

void getMagnitude( Mat &dfdx, Mat &dfdy, Mat &output );

void getDirection( Mat &dfdx, Mat &dfdy, Mat &output );

void getThresholdedMag(	Mat &input,	Mat &output );

void getHoughSpace(	Mat &thresholdedMag,Mat &gradientDirection,	int threshold, int width,	int height,	Mat &output );

void drawFoundLines( Mat &image, int width, int height );

int main(int argc, const char** argv) {

	const char* imgName = argv[1];

  Mat image;
  image = imread(imgName, 1 );

  // namedWindow( "Original Image", CV_WINDOW_AUTOSIZE );
  // imshow( "Original Image", image );

  cvtColor( image, image, CV_BGR2GRAY );

  Mat dfdx;
  dfdx.create(image.size(), CV_64F);

  Mat dfdy;
  dfdy.create(image.size(), CV_64F);

	Mat dxKernel = (Mat_<double>(3,3) << -1, 0, 1,
																			 -2, 0, 2,
																			 -1, 0, 1);

  Mat dyKernel = (Mat_<double>(3,3) << -1,-2,-1,
																		    0, 0, 0,
																		    1, 2, 1);

  Mat gradientMagnitude;
  gradientMagnitude.create(image.size(), CV_64F);

  Mat gradientDirection;
	gradientDirection.create(image.size(), CV_64F);

	Mat thresholdedMag;
	thresholdedMag.create(image.size(), CV_64F);

	Mat houghSpace;

	Mat foundLines = imread( imgName, 1 );

  convolution(image, 3, 0, dxKernel, dfdx);
  convolution(image, 3, 1, dyKernel, dfdy);

	getMagnitude(dfdx, dfdy, gradientMagnitude);
	getDirection(dfdx, dfdy, gradientDirection);

	getThresholdedMag(gradientMagnitude, thresholdedMag);

	getHoughSpace(thresholdedMag, gradientDirection, 240, image.cols, image.rows, houghSpace);

	drawFoundLines(foundLines, image.cols, image.rows);

  return 0;
}

void convolution(Mat &input, int size, int direction, Mat kernel, Mat &output) {

  int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

  // Create padded version of input
  Mat paddedInput;
	copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		BORDER_REPLICATE );


	// Gaussian blur before finding derivation
	GaussianBlur(paddedInput, paddedInput, Size(3,3), 0, 0, BORDER_DEFAULT);


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
	normalize(output, img, 0, 255, NORM_MINMAX);

  //Save thresholded image
	if (direction == 0) imwrite("output/dfdx.jpg", img);
	else imwrite("output/dfdy.jpg", img);
}

void getMagnitude(Mat &dfdx, Mat &dfdy,	Mat &output) {
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

	normalize(output, img, 0, 255, NORM_MINMAX);

  imwrite("output/magnitude.jpg", img);
}

void getDirection(Mat &dfdx, Mat &dfdy,	Mat &output) {
	for (int y = 0; y < output.rows; y++) {
    for (int x = 0; x < output.cols; x++) {

			double dxVal = 0.0;
      double dyVal = 0.0;
      double gradientVal = 0.0;

      dxVal = dfdx.at<double>(y, x);
      dyVal = dfdy.at<double>(y, x);

			// Calculate direction
			if (dxVal != 0 && dyVal != 0) gradientVal = atan2(dyVal, dxVal);
			else gradientVal = (double) atan(0);

			output.at<double>(y, x) = gradientVal;
    }
  }

	Mat img;
	img.create(dfdx.size(), CV_64F);

	normalize(output, img, 0, 255, NORM_MINMAX);

	imwrite("output/direction.jpg", img);
}

void getThresholdedMag(Mat &input, Mat &output) {
	Mat img;
	img.create(input.size(), CV_64F);

	normalize(input, img, 0, 255, NORM_MINMAX);

	for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {

      double val = 0;
      val = img.at<double>(y, x);


      if (val > 100) output.at<double>(y, x) = 255.0;
      else output.at<double>(y, x) = 0.0;
    }
  }

  imwrite("output/thresholded.jpg", output);
}

void getHoughSpace( Mat &thresholdedMag, Mat &gradientDirection, int threshold, int width, int height, Mat &houghSpace) {
	//double maxDist = sqrt(pow(width, 2) + pow(height, 2)) / 2;


	double rho = 0.0;
	double radians = 0.0;
	double directionTheta = 0.0;
	double directionVal = 0.0;
	int angleRange = 20;

	// houghSpace.create(round(maxDist), 180, CV_64F);

	houghSpace.create(2*(width + height), 360, CV_64F);

	for (int y = 0; y < thresholdedMag.rows; y++) {
		for (int x = 0; x < thresholdedMag.cols; x++) {
			if (thresholdedMag.at<double>(y, x) > 250) {

				directionVal = gradientDirection.at<double>(y, x);
				if (directionVal > 0) directionTheta = (directionVal * (180/PI));
				else directionTheta = 360 + (directionVal * (180/PI));

				directionTheta = round(directionTheta);

				for (int theta = directionTheta - angleRange; theta < directionTheta + angleRange; theta += 1) {
				// for (int theta = 0; theta < 360; theta++){
					radians = theta * (PI/ 180);

					rho = (x * cos(radians)) + (y * sin(radians)) + width + height;

					houghSpace.at<double>( rho , theta )++;
				}
			}
		}
	}

	// Mat img;
	// img.create(houghSpace.size(), CV_64F);

	imwrite("output/unThresholdedHoughSpace.jpg", houghSpace);



	//normalize(houghSpace, img, 0, 255, NORM_MINMAX);
	double min, max;
	cv::minMaxLoc(houghSpace, &min, &max);
	double houghSpaceThreshold = min + ((max - min)/2);

	//std::cout << max << " and " << min << '\n';

	// Thresholding Hough space
	for (int y = 0; y < houghSpace.rows; y++) {
		for (int x = 0; x < houghSpace.cols; x++) {

			double val = 0.0;
      val = houghSpace.at<double>(y, x);

			if (val > houghSpaceThreshold){
				rhoValues.push_back(y);
				thetaValues.push_back(x);
				houghSpace.at<double>(y, x) = 255;
				// std::cout<< rhoValues.size() << " + " << thetaValues.size() << "\n";
			}

      else houghSpace.at<double>(y, x) = 0.0;
		}
	}

	imwrite("output/houghSpace.jpg", houghSpace);
}

void drawFoundLines( Mat &image, int width, int height ){
	int centreX = width / 2;
	int centreY = height /2;

	for (int i = 0; i < rhoValues.size(); i++) {

		Point point1, point2;
		double theta = thetaValues[i];
		double rho = rhoValues[i];

		double radians = theta * (PI/ 180);

		//std::cout << rho << "and" << radians << '\n';

		double a = cos(radians);
		double b = sin(radians);
		double x0 = a * (rho - width - height);
		double y0 = b * (rho - width - height);

		point1.x = cvRound(x0 + 1000*(-b));
		point1.y = cvRound(y0 + 1000*(a));
		point2.x = cvRound(x0 - 1000*(-b));
		point2.y = cvRound(y0 - 1000*(a));

		line(image, point1, point2,  Scalar( 0, 255, 0 ), 2);
	}

	imwrite("output/foundLines.jpg", image);
}
