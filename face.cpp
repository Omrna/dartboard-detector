/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;



/** Function Headers */
void detectAndDisplay( Mat frame );
void F1Test( int facesDetected, const char* imgName );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
std::vector<Rect> faces;

/** @function main */
int main( int argc, const char** argv )
{

	const char* imgName = argv[1];

  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// ADDED: 4. Perform F1 test
	F1Test(faces.size(), imgName);

	// 5. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

void F1Test( int facesDetected, const char* imgName ){
	int truePositives = 0;

	string fileExtension = "points.csv";
	string current_line;

	std::string imgNameString(imgName);

	string::size_type i = imgNameString.rfind('.', imgNameString.length());
  if (i != string::npos) {
		imgNameString.replace(i, fileExtension.length(), fileExtension);
  }

	std::cout << imgNameString << " \n";

	const char *c = imgNameString.c_str();

	ifstream inputFile(c);

	while(getline(inputFile, current_line)){
   ++truePositives;
	}

	std::cout << truePositives << "\n";

	// float F1Test =
	// Use rectangle intersection Function
	// Precision = TP / (TP + FP)
	// Recall = TPR
	// F1 = 2((PRE * REC)/(PRE + REC))

}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << "faces detected: " << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
