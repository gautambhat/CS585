/*	CS585_Lab2.cpp
*	CS585 Image and Video Computing Fall 2014
*	Lab 2
*	--------------
*	This program introduces the following concepts:
*		a) Reading a stream of images from a webcamera, and displaying the video
*		b) Skin color detection
*		c) Background differencing
*		d) Visualizing motion history
*	--------------
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//function declarations

/**
Function that returns the maximum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/

bool FindTemplate(Mat Img_Scene_Bgr, Mat Img_Template_Bgr, Point &Point_TemplateLocation)
{
	// `Img_Scene_Bgr` and `Img_Template_Bgr` are the reference and template image
	cv::Mat Img_Result_Float(Img_Scene_Bgr.rows - Img_Template_Bgr.rows + 1, Img_Scene_Bgr.cols - Img_Template_Bgr.cols + 1, CV_32FC1);
	cv::matchTemplate(Img_Scene_Bgr, Img_Template_Bgr, Img_Result_Float, CV_TM_CCORR_NORMED);
	//normalize(Img_Result_Float, Img_Result_Float, 0, 1, NORM_MINMAX, -1, Mat());

	double minval, maxval, threshold = 0.8;
	cv::Point minloc, maxloc;
	cv::minMaxLoc(Img_Result_Float, &minval, &maxval, &minloc, &maxloc);
	cout << maxval << "\n";
	if (maxval >= threshold && maxloc.x)
	{
		Point_TemplateLocation = maxloc;
		return true;	
	}
	else
	{
		return false;
	}
}


int myMax(int a, int b, int c);

/**
Function that returns the minimum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMin(int a, int b, int c);

/**
Function that detects whether a pixel belongs to the skin based on RGB values
@param src The source color image
@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
*/
void mySkinDetect(Mat& src, Mat& dst);

/**
Function that does frame differencing between the current frame and the previous frame
@param src The current color image
@param prev The previous color image
@param dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
and previous image are not the same
*/
void transformer(Mat& t){
	
	Size size(200,200);
	resize(t, t, size);
	for (int i = 0; i < t.rows; i++){
		for (int j = 0; j < t.cols; j++){
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = t.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)){
				
				intensity[0] = 255;
				intensity[1] = 255;
				intensity[2] = 255;
				t.at<Vec3b>(i, j) = intensity;
			}
			else{
				intensity[0] = 0;
				intensity[1] = 0;
				intensity[2] = 0;
				t.at<Vec3b>(i, j) = intensity;
			}
		}
	}

}

/**
Function that accumulates the frame differences for a certain number of pairs of frames
@param mh Vector of frame difference images
@param dst The destination grayscale image to store the accumulation of the frame difference images
*/

int best = 1;
Mat result1, result2, result3;
Mat templ1 = imread("C://Users//gautam//Downloads//chill.jpg");
Mat templ2 = imread("C://Users//gautam//Downloads//paper.jpg");
Mat templ3 = imread("C://Users//gautam//Downloads//scissors.jpg");
int main()
{
	transformer(templ3);
	transformer(templ2);
	transformer(templ1);
	imshow("template", templ1);
	imshow("template", templ2);
	imshow("template", templ3);

	//----------------
	//a) Reading a stream of images from a webcamera, and displaying the video
	//----------------
	// For more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
	// open the video camera no. 0
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}
	//create a window called "MyVideoFrame0"
	//namedWindow("MyVideo0", WINDOW_AUTOSIZE);
	Mat frame0;

	// read a new frame from video
	bool bSuccess0 = cap.read(frame0);

	//if not successful, break loop
	if (!bSuccess0)
	{
		cout << "Cannot read a frame from video stream" << endl;
	}


	while (1)
	{
		// read a new frame from video
		Mat frame;
		bool bSuccess = cap.read(frame);

		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		// destination frame
		Mat frameDest;
		frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1
		//----------------
		//----------------
		mySkinDetect(frame, frameDest);
		imshow("Skin", frameDest);



		
		frame0 = frame;
		//wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

	}
	cap.release();
	return 0;
}



//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.

	vector<Mat> List_Template_Img;
	List_Template_Img.push_back(templ1);//Otherwise Get some folder & add the Files in it
	List_Template_Img.push_back(templ2);
	List_Template_Img.push_back(templ3);

	vector<Mat> resultMat;

	int match_method = 2;
	int result_cols = src.cols - templ1.cols + 1;
	int result_rows = src.rows - templ1.rows + 1;

	result1.create(result_rows, result_cols, CV_32FC1);
	result2.create(result_rows, result_cols, CV_32FC1);
	result3.create(result_rows, result_cols, CV_32FC1);

	resultMat.push_back(result1);
	resultMat.push_back(result2);
	resultMat.push_back(result3);

	//Mat templ = Mat::zeros(templat.rows, templat.cols, CV_8UC1);


	src.copyTo(dst);
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)){
				intensity[0] = 255;
				intensity[1] = 255;
				intensity[2] = 255;
				dst.at<Vec3b>(i, j) = intensity;
			}
			else{
				intensity[0] = 0;
				intensity[1] = 0;
				intensity[2] = 0;
				dst.at<Vec3b>(i, j) = intensity;
			}
		}
	}



	best = 0;
	Point  Point_TemplateLocation;
	for (int i = 0; i < List_Template_Img.size(); i++)
	{
	if (!FindTemplate(dst, List_Template_Img[i], Point_TemplateLocation))
	{
	cout << "No Match Found";
	}
	else
	{
		best = i;
		break;
	}

	}

	if (best == 0)
		putText(src, format("Call me!"), Point(Point_TemplateLocation.x + List_Template_Img[best].cols / 4, Point_TemplateLocation.y + List_Template_Img[best].rows / 2), 1, 1, Scalar(255, 0, 0), 1, -1);
	else if (best == 1)
		putText(src, format("Paper!"), Point(Point_TemplateLocation.x + List_Template_Img[best].cols / 4, Point_TemplateLocation.y + List_Template_Img[best].rows / 2), 1, 1, Scalar(255, 0, 0), 1, -1);
	else
		putText(src, format("Scissors!"), Point(Point_TemplateLocation.x + List_Template_Img[best].cols / 4, Point_TemplateLocation.y + List_Template_Img[best].rows / 2), 1, 1, Scalar(255, 0, 0), 1, -1);

	rectangle(src, Point_TemplateLocation, Point(Point_TemplateLocation.x + List_Template_Img[best].cols, Point_TemplateLocation.y + List_Template_Img[best].rows), Scalar(0, 0, 255), 2, 8, 0);
	
	imshow("result", resultMat[best]);
	imshow("source", src);

}



//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}
