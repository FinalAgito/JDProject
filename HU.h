#pragma once
#include <iostream>
#include <fstream> 
#include <sstream>
#include <icrsint.h>
#include<opencv.hpp>
#include<highgui\highgui.hpp>
#include<imgproc\imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2\core\core.hpp>

using namespace cv;
using namespace std;
class HU
{
public:
	HU(void);
	~HU(void);
	double MatchingHu(float *data1, float *data2);
 static int &  ExcuteHu(Mat img, void * param);
 int &ExcuteFFT(Mat img,void * param);
 int & EllipticFourierDescriptor(vector<Point> &contour,void *param); 
 float MatchingFFT(void *data1, void *data2);
float  CalDiffHog(void *data1, void *data2);
 int & ExcuteHog(Mat img, void * param, vector<float>&descriptors);
 void getFiles( string path, string exd, vector<string>& files );
 void HU::ExcuteHog2(Mat img, vector<float>&descriptors);
};


