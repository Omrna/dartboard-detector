/* Compile on personal Linux:
g++ dartboardDetector.cpp /usr/lib/libopencv_core.so.2.4
/usr/lib/libopencv_highgui.so.2.4 -lopencv_imgproc -lopencv_objdetect */

// Header Inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

/** Initialise Classes **/
class lineData {
  public:
    double m, c;
    Point point1, point2;
    lineData(double m, double c, Point point1, Point point2) : m(m), c(c), point1(point1), point2(point2) {}
};

/** Function Headers */
void detectVJ( Mat frame );
void displayVJ( Mat frame );
float F1Test( int facesDetected, const char* imgName, Mat frame );
void removeOverlaps();

void convolution(	Mat &input,	int size,	int direction,	Mat kernel,	Mat &output );
void getMagnitude( Mat &dfdx, Mat &dfdy, Mat &output );
void getDirection( Mat &dfdx, Mat &dfdy, Mat &output );
void getThresholdedMag(	Mat &input,	Mat &output );
void getHoughSpace(	Mat &thresholdedMag, Mat &gradientDirection, int threshold, int width,	int height,	Mat &output,
                    std::vector<double> &rhoValues, std::vector<double> &thetaValues);
void extractLines( Mat &frame, Mat &croppedImg, std::vector<double> &rhoValues, std::vector<double> &thetaValues, vector<lineData> &lines );
int lineDetector( Mat &frame, Mat &croppedImg, Rect rectangle );
int findCircles( Mat &frame );

void analyseBoxes( Mat &frame );

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

std::vector<Rect> detectedDartboardsVJ;
std::vector<Rect> detectedDartboardsFinal;
std::vector<Rect> trueDartboards;
//std::vector<line> v;

/* Functions */

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


      if (val > 200) output.at<double>(y, x) = 255.0;
      else output.at<double>(y, x) = 0.0;
    }
  }

  imwrite("output/thresholded.jpg", output);
}

void getHoughSpace( Mat &thresholdedMag, Mat &gradientDirection, int threshold, int width, int height, Mat &houghSpace, std::vector<double> &rhoValues, std::vector<double> &thetaValues) {
	//double maxDist = sqrt(pow(width, 2) + pow(height, 2)) / 2;

	double rho = 0.0;
	double radians = 0.0;
	double directionTheta = 0.0;
	double directionVal = 0.0;
	int angleRange = 1;

	// houghSpace.create(round(maxDist), 180, CV_64F);

	houghSpace.create(2*(width + height), 360, CV_64F);

	for (int y = 0; y < thresholdedMag.rows; y++) {
		for (int x = 0; x < thresholdedMag.cols; x++) {
			if (thresholdedMag.at<double>(y, x) > 250) {

				directionVal = gradientDirection.at<double>(y, x);
				if (directionVal > 0) directionTheta = (directionVal * (180/PI));
				else directionTheta = 360 + (directionVal * (180/PI));

				directionTheta = round(directionTheta);

				for (int theta = directionTheta - angleRange; theta < directionTheta + angleRange + 1; theta += 1) {
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

	normalize(houghSpace, houghSpace, 0, 255, NORM_MINMAX);

	double min, max;
	cv::minMaxLoc(houghSpace, &min, &max);
	// double houghSpaceThreshold = min + ((max - min)/2);

	// std::cout << max << " and " << min << '\n';

	// Thresholding Hough space
	for (int y = 0; y < houghSpace.rows; y++) {
		for (int x = 0; x < houghSpace.cols; x++) {

			double val = 0.0;
      val = houghSpace.at<double>(y, x);

			if (val > 150){
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

void extractLines( Mat &frame, Mat &croppedImg, Rect rectangle, std::vector<double> &rhoValues, std::vector<double> &thetaValues, vector<lineData> &lines ){

  int frameWidth = frame.cols;
  int frameHeight = frame.rows;

  int cropWidth = croppedImg.cols;
  int cropHeight = croppedImg.rows;

  Mat image = frame;

	for (int i = 0; i < rhoValues.size(); i++) {

		Point point1, point2;
		double theta = thetaValues[i];
		double rho = rhoValues[i];
    double m, c;
		double radians = theta * (PI/ 180);

    if (thetaValues.size() < 10000) {
      // Using y = mx + c
      // y = (p/sin(theta)) - x(cos(theta)/sin(theta))
      m = cos(radians) / sin(radians);
      c = (rho - cropWidth - cropHeight)/sin(radians);

      // When x = rectangle.x and y = rectangle.y + height
      point1.x = cvRound(rectangle.x);
      point1.y = cvRound(rectangle.y) + cvRound(c);
      // When x = end of image
      point2.x = cvRound(rectangle.x + rectangle.width);
      point2.y = cvRound(rectangle.y) + cvRound(c - (rectangle.width * m));

      clipLine(rectangle, point1, point2);

      lineData currentLine (m, c, point1, point2);

      lines.push_back(currentLine);

  		line(image, point1, point2,  Scalar( 0, 0, 255 ), 1);
    }
	}
  // std::cout << lines[1] << '\n';

	// imwrite("output/foundLines.jpg", image);
}

int lineDetector( Mat &frame, Mat &croppedImg, Rect rectangle ) {

  Mat image = croppedImg;

  Mat grayImage;

  cvtColor( image, grayImage, CV_BGR2GRAY );
  // equalizeHist( grayImage, grayImage );

  GaussianBlur(grayImage, grayImage, Size(3,3), 0, 0, BORDER_DEFAULT);
  // GaussianBlur(grayImage, grayImage, Size(3,3), 0, 0, BORDER_DEFAULT);

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

  std::vector<double> rhoValues;
  std::vector<double> thetaValues;

  vector<lineData> lines;

  convolution(grayImage, 3, 0, dxKernel, dfdx);
  convolution(grayImage, 3, 1, dyKernel, dfdy);

	getMagnitude(dfdx, dfdy, gradientMagnitude);
	getDirection(dfdx, dfdy, gradientDirection);

	getThresholdedMag(gradientMagnitude, thresholdedMag);

	getHoughSpace(thresholdedMag, gradientDirection, 240, image.cols, image.rows, houghSpace, rhoValues, thetaValues);

	extractLines(frame, image, rectangle, rhoValues, thetaValues, lines);

  return rhoValues.size();
}

int findCircles( Mat &frame ) {

  Mat image = frame;
  Mat grayImage;

  cvtColor( image, grayImage, CV_BGR2GRAY );
  equalizeHist( grayImage, grayImage );

  GaussianBlur(grayImage, grayImage, Size(3,3), 0, 0, BORDER_DEFAULT);

  Mat dfdx;
  dfdx.create(grayImage.size(), CV_64F);

  Mat dfdy;
  dfdy.create(grayImage.size(), CV_64F);

	Mat dxKernel = (Mat_<double>(3,3) << -1, 0, 1,
																			 -2, 0, 2,
																			 -1, 0, 1);

  Mat dyKernel = (Mat_<double>(3,3) << -1,-2,-1,
																		    0, 0, 0,
																		    1, 2, 1);

  Mat gradientMagnitude;
  gradientMagnitude.create(grayImage.size(), CV_64F);

  Mat gradientDirection;
	gradientDirection.create(grayImage.size(), CV_64F);

	Mat thresholdedMag;
	thresholdedMag.create(grayImage.size(), CV_64F);

	Mat houghSpace;

	// Mat foundLines = image;

  std::vector<double> rhoValues;
  std::vector<double> thetaValues;

  convolution(grayImage, 3, 0, dxKernel, dfdx);
  convolution(grayImage, 3, 1, dyKernel, dfdy);

	getMagnitude(dfdx, dfdy, gradientMagnitude);
	getDirection(dfdx, dfdy, gradientDirection);

	getThresholdedMag(gradientMagnitude, thresholdedMag);

  return 0;
}

int main( int argc, const char** argv ){

	const char* imgName = argv[1];

  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat VJOutput = frame.clone();

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces with Viola Jones
	detectVJ( frame );

  // 4. Draw Viola Jones boxes
  displayVJ( VJOutput );

  // 5. Save image with Viola Jones detections
  imwrite( "output/detectedVJ.jpg", VJOutput );

  // 6. Loop through cropped boxes to detect lines
  analyseBoxes( frame );

  // 7. Show and save Viola Jones boundaries on line detections
  displayVJ( frame );
  imwrite("output/foundLines.jpg", frame);


	// ADDED: 8. Perform F1 test
	float f1score = F1Test(detectedDartboardsVJ.size(), imgName, frame);

	// 6. Save Result Image
	// imwrite( "output/detected.jpg", frame );

	return 0;
}

float F1Test( int facesDetected, const char* imgName, Mat frame ){
	int validFaces = 0;

	// Manipulate string to get correct CSV file name
	string fileExtension = "points.csv";
	string current_line;

	std::string imgNameString(imgName);

	string::size_type i = imgNameString.rfind('.', imgNameString.length());
  if (i != string::npos) {
		imgNameString.replace(i, fileExtension.length(), fileExtension);
  }

	const char *c = imgNameString.c_str();
	ifstream inputFile(c);

	// Break if no CSV file found
	if (inputFile.peek() == std::ifstream::traits_type::eof()) {
		std::cout << "No CSV file found. F1 score cannot be calculated" << '\n';
		return 0.0;
	}

	// Go through CSV file line by line
	while(getline(inputFile, current_line)){

		// Array of values for each rectangle
		std::vector<int> values;

		std::stringstream convertor(current_line);
		std::string token; // somewhere to put the comma separated value

		// Insert each value into values array
		while (std::getline(convertor, token, ',')) {
			values.push_back(std::atoi(token.c_str()));
		}

		// Populate array with ground truth rectangles
		trueDartboards.push_back(Rect(values[0], values[1], values[2], values[3]));
	}

	int truePositives = 0;
	int falsePositives = 0;

	// Compare each detected face to every ground truth face
	for (int i = 0; i < detectedDartboardsVJ.size(); i++) {
		for (int j = 0; j < trueDartboards.size(); j++) {
			// Get intersection and check matching area percentage
			Rect intersection = detectedDartboardsVJ[i] & trueDartboards[j];
			float intersectionArea = intersection.area();

			// If there is an intersection, check percentage of intersection area
			// to detection area
			if (intersectionArea > 0) {
				float matchPercentage = (intersectionArea / trueDartboards[j].area()) * 100;

				// If threshold reached, increment true positives
				if (matchPercentage > 60){
					truePositives++;
					break;
				}
				if (j == (trueDartboards.size() - 1)) falsePositives++;
			}
			// If loop reaches end without reaching intersection threshold, it is
			// a false negative
			else {
				if (j == (trueDartboards.size() - 1)) falsePositives++;
			}
		}
	}

	std::cout << "true positives: " << truePositives << ", false positives: " << falsePositives << "\n";

	// Time for F1 test
	// Precision = TP / (TP + FP)
	// Recall = TPR (True Positive Rate)
	// F1 = 2((PRE * REC)/(PRE + REC))

	float precision = (float)truePositives / ((float)truePositives + (float)falsePositives);
	float recall = (float)truePositives / (float)trueDartboards.size();

	float f1 = 2 * ((precision * recall)/(precision + recall));

	std::cout << "f1 score: " << f1 << "\n";

	return f1;
}

void detectVJ( Mat frame ){
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, detectedDartboardsVJ, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	std::cout << "dartboards detected with Viola Jones: " << detectedDartboardsVJ.size() << std::endl;

}

void displayVJ( Mat frame ){
  // Draw box around faces found
	for( int i = 0; i < detectedDartboardsVJ.size(); i++ )
	{
		rectangle(frame, Point(detectedDartboardsVJ[i].x, detectedDartboardsVJ[i].y), Point(detectedDartboardsVJ[i].x + detectedDartboardsVJ[i].width, detectedDartboardsVJ[i].y + detectedDartboardsVJ[i].height), Scalar( 0, 255, 0 ), 2);
	}
}

void analyseBoxes( Mat &frame ){
  Mat frameForCrop = frame.clone();
  Mat croppedImg;
  int linesFound = 0;

  for (int i = 0; i < detectedDartboardsVJ.size(); i ++){
    croppedImg = frameForCrop(detectedDartboardsVJ[i]);
    linesFound = lineDetector(frame, croppedImg, detectedDartboardsVJ[i]);
  }

}
