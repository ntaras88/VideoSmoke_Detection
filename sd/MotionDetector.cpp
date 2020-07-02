#include "stdafx.h"
#include "MotionDetector.h"

MotionDetector::MotionDetector(void)
{
	backgroundFrame = NULL;
	currentFrame = NULL;
	currentFrameDilatated = NULL;
	counter = 0;
	resol = 4;
	initialized = false;
}


MotionDetector::~MotionDetector(void)
{
	if (backgroundFrame != NULL) delete[] backgroundFrame;
	if (currentFrame != NULL) delete[] currentFrame;
	if (currentFrameDilatated != NULL)delete[] currentFrameDilatated;
	backgroundFrame = NULL;
	currentFrame = NULL;
	currentFrameDilatated = NULL;
	counter = 0;
}

	// Motion level calculation - calculate or not motion level
bool MotionDetector::getMotionLevelCalculation()
{
	return calculateMotionLevel;
}

void MotionDetector::setMotionLevelCalculation(bool value)
{
	calculateMotionLevel = value;
}

// Motion level - amount of changes in percents
double MotionDetector::getMotionLevel()
{
	return (double) pixelsChanged / ( width * height );
}

// Reset detector to initial state
void MotionDetector::Reset( )
{
	if(backgroundFrame != NULL) delete[] backgroundFrame;
	if(currentFrame != NULL) delete[] currentFrame;
	if(currentFrameDilatated != NULL)delete[] currentFrameDilatated;
	backgroundFrame = NULL;
	currentFrame = NULL;
	currentFrameDilatated = NULL;
	counter = 0;
}

// Process new frame
Mat MotionDetector::ProcessFrame(Mat* frame, Rect& rect )
{
	Mat motionMask = Mat::zeros(rect.height, rect.width, CV_8UC1);
	try{
		
	// get image dimension
		width	= rect.width;
		height	= rect.height;
		int fW = ( ( ( width - 1 ) / resol ) + 1 );
		int fH = ( ( ( height - 1 ) / resol ) + 1 );
		int len = fW * fH;

		if (initialized == false)
		{
			Reset();
			// alloc memory for a backgound image and for current image
		
			backgroundFrame = new uchar[len];
			currentFrame = new uchar[len];
			currentFrameDilatated = new uchar[len];

			// create initial backgroung image
			PreprocessInputImage(frame, rect, backgroundFrame );
			initialized = true;
			return motionMask;
		}

		// preprocess input image
		PreprocessInputImage(frame, rect, currentFrame );

		if ( ++counter == 8 )
		{
			counter = 0;

			// move background towards current frame
			for ( int i = 0; i < len; i++ )
			{
				int t = currentFrame[i] - backgroundFrame[i];
				if ( t > 0 )
					backgroundFrame[i] += 1;
				else if ( t < 0 )
					backgroundFrame[i] -= 1;
			}
		}

		// difference and thresholding
		pixelsChanged = 0;
		for ( int i = 0; i < len; i++ )
		{
			int t = currentFrame[i] - backgroundFrame[i];
			if ( t < 0 )
				t = -t;

			if ( t >= nThreshold )
			{
				pixelsChanged++;
				currentFrame[i] = 255;
			}
			else
			{
				currentFrame[i] = 0;
			}
		}
		if ( calculateMotionLevel )
			pixelsChanged *= pow(resol , 2);
		else
			pixelsChanged = 0;

		// dilatation analogue for borders extending
		// it can be skipped
		for ( int i = 0; i < fH; i++ )
		{
			for ( int j = 0; j < fW; j++ )
			{
				int k = i * fW + j;
				int v = currentFrame[k];

				// left pixels
				if ( j > 0 )
				{
					v += currentFrame[k - 1];

					if ( i > 0 )
					{
						v += currentFrame[k - fW - 1];
					}
					if ( i < fH - 1 )
					{
						v += currentFrame[k + fW - 1];
					}
				}
				// right pixels
				if ( j < fW - 1 )
				{
					v += currentFrame[k + 1];

					if ( i > 0 )
					{
						v += currentFrame[k - fW + 1];
					}
					if ( i < fH - 1 )
					{
						v += currentFrame[k + fW + 1];
					}
				}
				// top pixel
				if ( i > 0 )
				{
					v += currentFrame[k - fW];
				}
				// right pixel
				if ( i < fH - 1 )
				{
					v += currentFrame[k + fW];
				}
				currentFrameDilatated[k] = (v != 0) ?  255 : 0;
			}
		}
		
		// postprocess the input image
		PostprocessInputImage(frame, rect, currentFrameDilatated, &motionMask);
		
	}catch(Exception) {

	}
	return motionMask;
}

// Preprocess input image
void MotionDetector::PreprocessInputImage( Mat* data, Rect& rect, uchar* buf )
{
	int len = (int)( ( rect.width - 1 ) / resol ) + 1;
	int rem = ( ( rect.width - 1 ) % resol ) + 1;
	int tmp[500];// = new int[500];
	int i, j, t1, t2, k = 0;

	try{			
		for ( int y = rect.y; y < (rect.y + rect.height);)
		{
			// collect pixels
			for(int m = 0;m < len;m++) {
				tmp[m] = 0;
			}
			for ( i = 0; ( i < resol ) && ( y < (rect.y + rect.height) ); i++, y++ ){
				for ( int x = rect.x; x < (rect.x + rect.width); x++ )
				{
					// grayscale value using BT709
					tmp[(int) ( (x - rect.x) / resol )] += (int)( 0.2125f * data->at<Vec4b>(y, x).val[2] + 0.7154f * data->at<Vec4b>(y, x).val[1] + 0.0721f * data->at<Vec4b>(y, x).val[0] );
				}
			}
			// get average values
			t1 = i * resol;
			t2 = i * rem;

			for ( j = 0; j < len - 1; j++, k++ ) {
				if(t1 == 0) {
					buf[k] = 0;
				}else{
					buf[k] = ( tmp[j] / t1 );
				}
			}
			if(t2 == 0) buf[k++] = 0; else buf[k++] = ( tmp[j] / t2 );
		}
	}catch(Exception) {
	
	}
	//delete[] tmp;
	//free((void *)tmp);
}

// Postprocess input image
void MotionDetector::PostprocessInputImage( Mat* data, Rect& rect, uchar* buf, Mat *motionMask)
{
	int len = (int)( ( rect.width - 1 ) / resol ) + 1;
	int lenWM1 = len - 1;
	int lenHM1 = (int)( ( rect.height - 1 ) / resol);
	for(int i = 0;i < lenHM1;i++) {
		for(int j = 0;j < lenWM1;j++) {
			motionMask->at<uchar>(i, j) = 0;
		}
	}
	int rem = ( ( rect.width - 1 ) % resol ) + 1;

	int pi = 0, j = 0, k = 0;
		// for each line
	for ( int y = rect.y; y < (rect.y + rect.height); y++ )
	{
		pi = (y - rect.y) / resol;

		// for each pixel
		for ( int x = rect.x; x < (rect.x + rect.width); x++ )
		{
			j = (x - rect.x) / resol;	
			k = pi * len + j;

			// check if we need to highlight moving object
			if (buf[k] == 255)
			{
				// check for border
				if (
					( ( x % resol == 0 ) && ( ( j == 0 ) || ( buf[k - 1] == 0 ) ) ) ||
					( ( x % resol == resol - 1 ) && ( ( j == lenWM1 ) || ( buf[k + 1] == 0 ) ) ) ||
					( ( y % resol == 0 ) && ( ( pi == 0 ) || ( buf[k - len] == 0 ) ) ) ||
					( ( y % resol == resol - 1 ) && ( ( pi == lenHM1 ) || ( buf[k + len] == 0 ) ) )
					)
				{
					//data->at<Vec3b>(y, x) = Vec3b(0, 0, 255);
				}
				if(( x % resol == 0 ) && ( ( j == 0 ) || ( buf[k - 1] == 0 ) ) ) {
					motionMask->at<uchar>(y, x) = motionMask->at<uchar>(y, x) + 3;
				}
				if(( x % resol == resol - 1 ) && ( ( j == lenWM1 ) || ( buf[k + 1] == 0 ) ) ) {
					motionMask->at<uchar>(y, x) = motionMask->at<uchar>(y, x) + 12;
				}
				if(( y % resol == 0 ) && ( ( pi == 0 ) || ( buf[k - len] == 0 ) ) ) {
					motionMask->at<uchar>(y, x) = motionMask->at<uchar>(y, x) + 48;
				}
				if( ( y % resol == resol - 1 ) && ( ( pi == lenHM1 ) || ( buf[k + len] == 0 ) ) ) {
					motionMask->at<uchar>(y, x) = motionMask->at<uchar>(y, x) + 192;
				}
			}
		}
	}
}