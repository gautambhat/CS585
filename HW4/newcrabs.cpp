
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

Rect boundrec1;
Rect boundrec2;


void findContour(Mat& imgThreshold);

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 25;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = { 0, 0 };
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);


//int to string helper function
string intToString(int number){

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}

bool searchForMovement(Mat thresholdImage, Mat &cameraFeed){
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);// retrieves external contours
	Moments moment;
	Point2f centroid;
	for (auto& contour : contours)
	{
		if (contourArea(contour) > 20 )
		{
			moment = moments(contour, false);
			centroid = Point2f(moment.m10 / moment.m00, moment.m01 / moment.m00);
			circle(cameraFeed, centroid, 20, Scalar(0, 255, 0), 2);
			std::cout << " The centroid of the crab is " << centroid.x << " " << centroid.y << std::endl;
		}
	}
	//if contours vector is not empty, we have found some objects
	if (contours.size() > 0)
	{
		objectDetected = true;
	}

	else {
		objectDetected = false;
		return false;
	}

	if (objectDetected){


		//the largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is the object we are looking for.
		vector< vector<Point> > largestContourVec;
		largestContourVec.push_back(contours.at(contours.size() - 1));
		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(largestContourVec.at(0));
		int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
		int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

		//update the objects positions by changing the 'theObject' array values
		theObject[0] = xpos, theObject[1] = ypos;

		//make some temp x and y variables so we dont have to type out so much
		int x = theObject[0];
		int y = theObject[1];

		//draw some crosshairs around the object
		circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
		line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
		line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
		line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
		line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);

		//write the position of the object to the screen
		putText(cameraFeed, "Tracking object at (" + intToString(x) + "," + intToString(y) + ")", Point(x, y), 1, 1, Scalar(255, 0, 0), 2);
	}
	return true;
}

int main(){

	VideoCapture cap;
	cap.open("C://Users//aalva//Downloads//GP010190.MP4");

	//some boolean variables for added functionality
	bool objectDetected = false;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = false;
	bool trackingEnabled = false;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing
	Mat frame1, frame2;
	//their grayscale images (needed for absdiff() function)
	Mat grayImage1, grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//video capture object.

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 31;

	int iLowV = 112;
	int iHighV = 255;

	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	int iLastX = -1;
	int iLastY = -1;

	//Capture a temporary image from the camera
	Mat imgTmp;
	cap.read(imgTmp);
	imshow("First image", imgTmp);
	waitKey(0);

	Mat imgHSV;

	cvtColor(imgTmp, imgHSV, COLOR_BGR2HSV);

	Mat imgThresholded;

	//Create a black image with the size as the camera output
	Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);

	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
	int erode_erosion_size = 5;
	Mat erode_element = getStructuringElement(MORPH_RECT,
		Size(2 * erode_erosion_size + 1, 2 * erode_erosion_size + 1),
		Point(erode_erosion_size, erode_erosion_size));

	erode(imgThresholded, imgThresholded, erode_element);
	erode(imgThresholded, imgThresholded, erode_element);
	erode(imgThresholded, imgThresholded, erode_element);

	int dilate_erosion_size = 6;
	Mat dilate_element = getStructuringElement(MORPH_RECT,
		Size(2 * dilate_erosion_size + 1, 2 * dilate_erosion_size + 1),
		Point(dilate_erosion_size, dilate_erosion_size));
	dilate(imgThresholded, imgThresholded, dilate_element);
	dilate(imgThresholded, imgThresholded, dilate_element);
	dilate(imgThresholded, imgThresholded, dilate_element);
	dilate(imgThresholded, imgThresholded, dilate_element);
	//imshow("Thresholded", imgThresholded);
	//imshow("Original", imgTmp);
	findContour(imgThresholded);
	cv::Mat leftTank;
	cv::Mat rightTank;
	rightTank = cv::Mat(imgTmp, boundrec2).clone();
	leftTank = cv::Mat(imgTmp, boundrec1).clone();
	imshow("left Tank", leftTank);
	imshow("right Tank", rightTank);

	Mat baseLeftFrame, baseRightFrame;
	baseLeftFrame = imread("leftTank0.jpg");
	baseRightFrame = imread("rightTank0.jpg");
	waitKey(0);

	while (1){

		//we can loop the video by re-opening the capture every time the video reaches its last frame
		//C://Users//gautam//Downloads//CS585_lab6_solution//GP010191.MP4
		//C://Users//gautam//Downloads//motionTrackingTutorial//bouncingBall.avi

		if (!cap.isOpened()){
			cout << "ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}

		//check if the video has reach its last frame.
		//we add '-1' because we are reading two frames from the video at a time.
		//if this is not included, we get a memory error!
		while (cap.get(CV_CAP_PROP_POS_FRAMES)<cap.get(CV_CAP_PROP_FRAME_COUNT) - 1){

			//Mat imgOriginal;

			//bool bSuccess = cap.read(imgOriginal); // read a new frame from video

			//if (!bSuccess) //if not success, break loop
			//{
			//cout << "Cannot read a frame from video stream" << endl;
			//break;
			//}
			//rightTank = cv::Mat(imgOriginal, boundrec1).clone();
			//leftTank = cv::Mat(imgOriginal, boundrec2).clone();
			//imshow("leftTank", leftTank);
			//imshow("rightTank", rightTank);

			/*int ms = cap.get(CV_CAP_PROP_POS_MSEC);
			int x = ms / 1000;
			int seconds = x % 60;
			x /= 60;
			int minutes = x % 60;
			x /= 60;
			cout << minutes << ":" << seconds << endl;*/
			//read first frame
			cap.read(frame1);
			frame1 = cv::Mat(frame1, boundrec2).clone();
			//convert frame1 to gray scale for frame differencing
			cv::cvtColor(frame1, grayImage1, COLOR_BGR2GRAY);
			cv::cvtColor(baseRightFrame, grayImage2, COLOR_BGR2GRAY);
			//copy second frame
			//cap.read(frame2);
			//frame2 = cv::Mat(frame2, boundrec2).clone();
			//convert frame2 to gray scale for frame differencing
			//cv::cvtColor(frame2, grayImage2, COLOR_BGR2GRAY);
			//perform frame differencing with the sequential images. This will output an "intensity image"
			//do not confuse this with a threshold image, we will need to perform thresholding afterwards.
			cv::absdiff(grayImage1, grayImage2, differenceImage);
			//threshold intensity image at a given sensitivity value
			cv::threshold(differenceImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);
			if (debugMode == true){
				//show the difference image and threshold image
				cv::imshow("Difference Image", differenceImage);
				cv::imshow("Threshold Image", thresholdImage);
			}
			else{
				//if not in debug mode, destroy the windows so we don't see them anymore
				cv::destroyWindow("Difference Image");
				cv::destroyWindow("Threshold Image");
			}
			//blur the image to get rid of the noise. This will output an intensity image
			cv::blur(thresholdImage, thresholdImage, cv::Size(BLUR_SIZE, BLUR_SIZE));
			//threshold again to obtain binary image from blur output
			cv::threshold(thresholdImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);
			if (debugMode == true){
				//show the threshold image after it's been "blurred"

				imshow("Final Threshold Image", thresholdImage);

			}
			else {
				//if not in debug mode, destroy the windows so we don't see them anymore
				cv::destroyWindow("Final Threshold Image");
			}

			//if tracking enabled, search for contours in our thresholded image
			if (trackingEnabled){

				if (searchForMovement(thresholdImage, frame1)){
					int ms = cap.get(CV_CAP_PROP_POS_MSEC);
					int x = ms / 1000;
					int seconds = x % 60;
					x /= 60;
					int minutes = x % 60;
					x /= 60;
					cout << minutes << ":" << seconds << endl;
				}
			}

			//show our captured frame
			imshow("Frame1", frame1);
			//check to see if a button has been pressed.
			//this 10ms delay is necessary for proper operation of this program
			//if removed, frames will not have enough time to referesh and a blank 
			//image will appear.
			switch (waitKey(30)){

			case 27: //'esc' key has been pressed, exit program.
				return 0;
			case 116: //'t' has been pressed. this will toggle tracking
				trackingEnabled = !trackingEnabled;
				if (trackingEnabled == false) cout << "Tracking disabled." << endl;
				else cout << "Tracking enabled." << endl;
				break;
			case 100: //'d' has been pressed. this will debug mode
				debugMode = !debugMode;
				if (debugMode == false) cout << "Debug mode disabled." << endl;
				else cout << "Debug mode enabled." << endl;
				break;
			case 112: //'p' has been pressed. this will pause/resume the code.
				pause = !pause;
				if (pause == true){
					cout << "Code paused, press 'p' again to resume" << endl;
					while (pause == true){
						//stay in this loop until 
						switch (waitKey()){
							//a switch statement inside a switch statement? Mind blown.
						case 112:
							//change pause back to false
							pause = false;
							cout << "Code Resumed" << endl;
							break;
						}
					}
				}
			}
		}
		//release the capture before re-opening and looping again.
		cap.release();
	}

	return 0;

}


void findContour(Mat& imgThreshold)
{
	// We will add the contour detection here
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgThreshold, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat contour_output = Mat::zeros(imgThreshold.size(), CV_8UC3);
	cout << "The number of contours detected is: " << contours.size() << endl;
	std::vector<int> areaVec;
	std::vector<int> widthVec;
	for (int i = 0; i < contours.size(); ++i)
	{
		vector<Point> cPoints = contours[i];
		int minXPoint = cPoints.at(0).x;
		int maxXPoint = cPoints.at(0).x;
		for (auto& point : cPoints)
		{
			if (point.x < minXPoint)
				minXPoint = point.x;
			else if (point.x > maxXPoint)
				maxXPoint = point.x;
		}
		int width = maxXPoint - minXPoint;
		std::cout << "width" << "\t" << width << std::endl;
		//drawContours(contour_output, contours, i, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
		//imshow("contour_output", contour_output);
		//waitKey(0);

		widthVec.push_back(width);
	}

	std::vector<int> y(widthVec.size());
	std::size_t n(0);
	std::generate(std::begin(y), std::end(y), [&]{ return n++; });

	std::sort(std::begin(y),
		std::end(y),
		[&](double i1, double i2) { return widthVec[i1] > widthVec[i2]; });

	for (auto v : y)
		std::cout << v << ' ';
	//Mat contour_output = Mat::zeros(imgThreshold.size(), CV_8UC3);

	boundrec1 = boundingRect(contours[y[0]]);
	boundrec2 = boundingRect(contours[y[1]]);
	rectangle(contour_output, boundrec1, Scalar(0, 255, 0), 1, 8, 0);
	rectangle(contour_output, boundrec2, Scalar(255, 0, 0), 1, 8, 0);

	drawContours(contour_output, contours, y[0], Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
	drawContours(contour_output, contours, y[1], Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
	//imshow("contour_output", contour_output);
}