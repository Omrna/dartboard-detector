#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, const char** argv) {

  Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  const char* imgName = argv[1];

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

    std::cout << values[0];
		// Populate array with ground truth rectangles
    rectangle(frame, Point(values[0], values[1]), Point(values[0] + values[2], values[1] + values[3]), Scalar( 0, 0, 255 ), 2);
	}

  imwrite( "annotated.jpg", frame );

  return 0;
}
