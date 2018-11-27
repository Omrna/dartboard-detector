/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
float F1Test( int facesDetected, const char* imgName, Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
std::vector<Rect> detectedFaces;
std::vector<Rect> trueFaces;

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
	float f1score = F1Test(detectedFaces.size(), imgName, frame);

	// 5. Save Result Image
	imwrite( "detected.jpg", frame );

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
		trueFaces.push_back(Rect(values[0], values[1], values[2], values[3]));
	}

	int truePositives = 0;
	int falsePositives = 0;

	// Compare each detected face to every ground truth face
	for (int i = 0; i < detectedFaces.size(); i++) {
		for (int j = 0; j < trueFaces.size(); j++) {
			// Get intersection and check matching area percentage
			Rect intersection = detectedFaces[i] & trueFaces[j];
			float intersectionArea = intersection.area();

			// If there is an intersection, check percentage of intersection area
			// to detection area
			if (intersectionArea > 0) {
				float matchPercentage = (intersectionArea / trueFaces[j].area()) * 100;

				// If threshold reached, increment true positives
				if (matchPercentage > 60){
					truePositives++;
					break;
				}
				if (j == (trueFaces.size() - 1)) falsePositives++;
			}
			// If loop reaches end without reaching intersection threshold, it is
			// a false negative
			else {
				if (j == (trueFaces.size() - 1)) falsePositives++;
			}
		}
	}

	std::cout << "true positives: " << truePositives << ", false positives: " << falsePositives << "\n";

	// Time for F1 test
	// Precision = TP / (TP + FP)
	// Recall = TPR (True Positive Rate)
	// F1 = 2((PRE * REC)/(PRE + REC))

	float precision = (float)truePositives / ((float)truePositives + (float)falsePositives);
	float recall = (float)truePositives / (float)trueFaces.size();

	float f1 = 2 * ((precision * recall)/(precision + recall));

	std::cout << "f1 score: " << f1 << "\n";

	return f1;
}

void detectAndDisplay( Mat frame )
{
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, detectedFaces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	std::cout << "faces detected: " << detectedFaces.size() << std::endl;

  // 4. Draw box around faces found
	for( int i = 0; i < detectedFaces.size(); i++ )
	{
		rectangle(frame, Point(detectedFaces[i].x, detectedFaces[i].y), Point(detectedFaces[i].x + detectedFaces[i].width, detectedFaces[i].y + detectedFaces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
