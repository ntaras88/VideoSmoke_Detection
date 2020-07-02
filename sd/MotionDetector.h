#pragma once

#include "stdafx.h"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"


#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;

class MotionDetector
{
public:
	MotionDetector(void);
	~MotionDetector(void);

private:
	uchar*	backgroundFrame;
	uchar*	currentFrame;
	uchar*	currentFrameDilatated;

	int		counter;

	bool	calculateMotionLevel;
	int		width;	// image width
	int		height;	// image height
	int		pixelsChanged;
public:
	int nThreshold;
public:
	bool	initialized;
	bool	getMotionLevelCalculation();
	void	setMotionLevelCalculation(bool value);
	double	getMotionLevel();
	void	Reset();
	Mat	ProcessFrame(Mat* image, Rect& rect );
private:
	void PreprocessInputImage( Mat* data, Rect& rect, uchar* buf );
	void PostprocessInputImage( Mat* data, Rect& rect, uchar* buf, Mat *motionMask);
	int resol;
};

