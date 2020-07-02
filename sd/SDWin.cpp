// SDWin.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "SDWin.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <process.h>

using namespace cv;

#define MAX_CAMERA_NUMBER	64
#define MAX_LOADSTRING 100
#define NONE 0  // no filter
#define HARD 1  // hard shrinkage
#define SOFT 2  // soft shrinkage
#define GARROT 3  // garrot filter

#define BUFFER_LEN_MAX		64
#define WIDTH_RESIZED		184//736
#define HEIGHT_RESIZED		96//384
#define MOTION_TRIGER		30
#define SMOKE_TRIGER		2
#define DETECTION_TRIGGER	1
#define MOTION_EXTEND		3
#define MX_TRIGGER			2
#define MOTION_DELAY		90
#define SMOKE_DELAY			90
#define WAVELET_TRIGGER		0.7
#define WAVELET_S_TRIGGER	0.5
#define BLUR_AMOUNT			2
#define WAVE_BLUR_AMOUNT	2
#define WAVE_S_BLUR_AMOUNT	2
#define FRAME_PER_SECOND	25.0f
#define BG_STABLE_TRIGGER	200
#define BG_STABLE_BLOCK		8
#define MAX_STABLE_TIME		150000

std::vector<vector<Point>> blobs;
// Global Variables:
HINSTANCE hInst;								// current instance
TCHAR szTitle[MAX_LOADSTRING];					// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];			// the main window class name
typedef struct _DETECT_PROFILE_{
	int		camRows = 4;
	int		camColumns = 4;
	BOOL	fullScreeenFlag = FALSE;
	BOOL	camProperties[MAX_CAMERA_NUMBER];
	int		bgDetectTime = 1500;
}DETECT_PROFILE, *PDETECT_PROFILE;

int blockSize = 2;
DETECT_PROFILE	profile;

bool startFalg = false;
bool stopFlag = false;
bool bShowMotionFlag = false;

int fCount = 0;
int quadFilterRate = 0;
int bufferLength = 0;
int bufferLenStartVal = 5;
int nWidthResized = 0;
int nHeightResized = 0;
int nDetectionTrigger = 0;
int nMotionExtend = 0;
int nMXTrigger = 0;

CRITICAL_SECTION		crit;
HCURSOR hCursor = NULL;
HANDLE detectThread = NULL;
HWND hTopMostWindow = NULL;
// Forward declarations of functions included in this code module:
LRESULT CALLBACK	MonitorDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
BOOL	GetDesktopImage(Mat *origin);

static void cvHaarWavelet(Mat &src, Mat &dst, int NIter)
{
	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;

	for (int k = 0; k<NIter; k++)
	{
		for (int y = 0; y<(height >> (k + 1)); y++)
		{
			for (int x = 0; x<(width >> (k + 1)); x++)
			{
				c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x) = c;

				dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x + (width >> (k + 1))) = dh;

				dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x) = dv;

				dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
			}
		}
		dst.copyTo(src);
	}
}

float sgn(float x)
{
	float res = 0;
	if (x == 0)
	{
		res = 0;
	}
	if (x > 0)
	{
		res = 1;
	}
	if (x < 0)
	{
		res = -1;
	}
	return res;
}
//--------------------------------
// Soft shrinkage
//--------------------------------
float soft_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = sgn(d)*(fabs(d) - T);
	}
	else
	{
		res = 0;
	}

	return res;
}
//--------------------------------
// Hard shrinkage
//--------------------------------
float hard_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = d;
	}
	else
	{
		res = 0;
	}

	return res;
}
//--------------------------------
// Garrot shrinkage
//--------------------------------
float Garrot_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = d - ((T*T) / d);
	}
	else
	{
		res = 0;
	}

	return res;
}

static void cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50)
{
	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;
	//--------------------------------
	// NIter - number of iterations 
	//--------------------------------
	for (int k = NIter; k>0; k--)
	{
		for (int y = 0; y<(height >> k); y++)
		{
			for (int x = 0; x<(width >> k); x++)
			{
				c = src.at<float>(y, x);
				dh = src.at<float>(y, x + (width >> k));
				dv = src.at<float>(y + (height >> k), x);
				dd = src.at<float>(y + (height >> k), x + (width >> k));

				// (shrinkage)
				switch (SHRINKAGE_TYPE)
				{
				case HARD:
					dh = hard_shrink(dh, SHRINKAGE_T);
					dv = hard_shrink(dv, SHRINKAGE_T);
					dd = hard_shrink(dd, SHRINKAGE_T);
					break;
				case SOFT:
					dh = soft_shrink(dh, SHRINKAGE_T);
					dv = soft_shrink(dv, SHRINKAGE_T);
					dd = soft_shrink(dd, SHRINKAGE_T);
					break;
				case GARROT:
					dh = Garrot_shrink(dh, SHRINKAGE_T);
					dv = Garrot_shrink(dv, SHRINKAGE_T);
					dd = Garrot_shrink(dd, SHRINKAGE_T);
					break;
				}

				//-------------------
				dst.at<float>(y * 2, x * 2) = 0.5*(c + dh + dv + dd);
				dst.at<float>(y * 2, x * 2 + 1) = 0.5*(c - dh + dv - dd);
				dst.at<float>(y * 2 + 1, x * 2) = 0.5*(c + dh - dv - dd);
				dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5*(c - dh - dv + dd);
			}
		}
		Mat C = src(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		Mat D = dst(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		D.copyTo(C);
	}
}

void labelBlobs(const cv::Mat &binary)
{
	blobs.clear();

	// Using labels from 2+ for each blob
	cv::Mat label_image;
	binary.convertTo(label_image, CV_32FC1);

	int label_count = 2; // starts at 2 because 0,1 are used already

	for (int y = 0; y < binary.rows; y++) {
		for (int x = 0; x < binary.cols; x++) {
			if ((int)label_image.at<float>(y, x) != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x, y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);

			vector<Point>  blob;

			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if ((int)label_image.at<float>(i, j) != label_count) {
						continue;
					}

					blob.push_back(cv::Point(j, i));
				}
			}

			blobs.push_back(blob);
			label_count++;
		}
	}
}

void MakeIntegral(Mat *origin, Mat *integral) {
	Mat extend;

	if (origin->type() == CV_8UC3) {
		extend = Mat::zeros(origin->rows + 2, origin->cols + 2, CV_8UC3);
		for (int i = 0; i< origin->rows; i++) {
			Vec3b *ptrOrigin = (Vec3b *)origin->ptr(i);
			Vec3b *ptrExtend = (Vec3b *)extend.ptr(i + 1);
			for (int j = 0; j < origin->cols; j++) {
				ptrExtend[j + 1] = ptrOrigin[j];
			}
		}
		for (int i = 1; i < extend.rows; i++) {
			Vec3b *ptrExtend = (Vec3b *)extend.ptr(i);
			Vec3f *ptrIntg1 = (Vec3f *)integral->ptr(i - 1);
			Vec3f *ptrIntg2 = (Vec3f *)integral->ptr(i);
			Vec3f s = Vec3f(0.0, 0.0, 0.0);
			for (int j = 1; j < extend.cols; j++) {

				s.val[0] += (float)(ptrExtend[j].val[0]);
				s.val[1] += (float)(ptrExtend[j].val[1]);
				s.val[2] += (float)(ptrExtend[j].val[2]);
				ptrIntg2[j].val[0] = ptrIntg1[j].val[0] + s.val[0];
				ptrIntg2[j].val[1] = ptrIntg1[j].val[1] + s.val[1];
				ptrIntg2[j].val[2] = ptrIntg1[j].val[2] + s.val[2];
			}
		}
	}
	else if (origin->type() == CV_8UC1)
	{
		extend = Mat::zeros(origin->rows + 2, origin->cols + 2, CV_8UC1);
		for (int i = 0; i< origin->rows; i++) {
			uchar *ptrOrigin = (uchar *)origin->ptr(i);
			uchar *ptrExtend = (uchar *)extend.ptr(i + 1);
			for (int j = 0; j < origin->cols; j++) {
				ptrExtend[j + 1] = ptrOrigin[j];
			}
		}
		for (int i = 1; i < extend.rows; i++) {
			float s = 0;
			uchar *ptrExtend = (uchar *)extend.ptr(i);
			float *ptrIntg1 = (float *)integral->ptr(i - 1);
			float *ptrIntg2 = (float *)integral->ptr(i);
			for (int j = 1; j < extend.cols; j++) {
				s += (float)(ptrExtend[j]);
				ptrIntg2[j] = ptrIntg1[j] + s;
			}
		}
	}
	else if (origin->type() == CV_32FC1)
	{
		extend = Mat::zeros(origin->rows + 2, origin->cols + 2, CV_32FC1);
		for (int i = 0; i< origin->rows; i++) {
			float *ptrOrigin = (float *)origin->ptr(i);
			float *ptrExtend = (float *)extend.ptr(i + 1);
			for (int j = 0; j < origin->cols; j++) {
				ptrExtend[j + 1] = ptrOrigin[j];
			}
		}
		for (int i = 1; i < extend.rows; i++) {
			float s = 0;
			float *ptrExtend = (float *)extend.ptr(i);
			float *ptrIntg1 = (float *)integral->ptr(i - 1);
			float *ptrIntg2 = (float *)integral->ptr(i);
			for (int j = 1; j < extend.cols; j++) {
				s += (float)(ptrExtend[j]);
				ptrIntg2[j] = ptrIntg1[j] + s;
			}
		}
	}
	extend.release();
}

void MakeEnergy(Mat *src, Mat *dst) {
	for (int i = 0; i < src->rows / 2; i++) {
		float *ptr1 = (float *)src->ptr(i);
		float *ptr2 = (float *)src->ptr(i + src->rows / 2);
		float *ptr = (float *)dst->ptr(i);
		for (int j = 0; j < src->cols / 2; j++) {
			ptr[j] = ptr1[j + src->cols / 2] * ptr1[j + src->cols / 2] + ptr2[j] * ptr2[j] + ptr2[j + src->cols / 2] * ptr2[j + src->cols / 2];
		}
	}
}

void cleanup() {
	stopFlag = true;
}

BOOL ReadProfile() {
	FILE *pFile = fopen("C:\\Windows\\CamProfile.bin", "rb");
	if (pFile == NULL) return FALSE;
	if (fread(&profile, sizeof(DETECT_PROFILE), 1, pFile) == 0) return FALSE;
	fclose(pFile);
	return TRUE;
}

BOOL WriteProfile() {
	FILE *pFile = fopen("C:\\Windows\\CamProfile.bin", "wb");
	if (pFile == NULL) return FALSE;
	if (fwrite(&profile, sizeof(DETECT_PROFILE), 1, pFile) == 0) return FALSE;
	fclose(pFile);
	return TRUE;
}

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPTSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
	profile.bgDetectTime = FRAME_PER_SECOND * 1.0f;
	profile.camColumns = 4;
	profile.camRows = 4;
	for (int i = 0; i < MAX_CAMERA_NUMBER; i++) {
		profile.camProperties[i] = FALSE;
	}
	profile.fullScreeenFlag = FALSE;
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);
	hInst = hInstance;
	InitializeCriticalSection(&crit);
	DialogBoxA(hInst, MAKEINTRESOURCEA(IDD_MONITOR), 0, (DLGPROC)MonitorDlgProc);
	atexit(cleanup);
 	// TODO: Place code here.
	DeleteCriticalSection(&crit);
	return 0;
}

void ToMat(HBITMAP hBmp, Mat *img)
{
	BITMAP bmp;
	GetObject(hBmp, sizeof(bmp), &bmp);

	int nChannels = bmp.bmBitsPixel == 1 ? 1 : bmp.bmBitsPixel / 8;
	if (nChannels > 4) return;
	int depth = bmp.bmBitsPixel == 1 ? IPL_DEPTH_1U : IPL_DEPTH_8U;

	if (nChannels == 3)
		*img = Mat(bmp.bmHeight, bmp.bmWidth, CV_8UC3);
	else if (nChannels == 4)
		*img = Mat(bmp.bmHeight, bmp.bmWidth, CV_8UC4);
	else if (nChannels == 1)
		*img = Mat(bmp.bmHeight, bmp.bmWidth, CV_8UC1);

	BYTE * pBuffer = new BYTE[bmp.bmHeight * bmp.bmWidth * nChannels];

	GetBitmapBits(hBmp, bmp.bmHeight * bmp.bmWidth * nChannels, pBuffer);
	memcpy(img->data, pBuffer, bmp.bmHeight * bmp.bmWidth * nChannels);
	delete pBuffer;
}

void ShowResultImage(HWND hWnd, Mat *frame) {
	HDC hdc = GetDC(GetDlgItem(hWnd, IDC_SCREEN));
	HDC hdcMem = CreateCompatibleDC(hdc);
	HGDIOBJ oldBitmap;
	HBITMAP hBmp = CreateBitmap(frame->cols, frame->rows, 1, 8 * frame->channels(), frame->data);
	RECT rect;
	GetClientRect(GetDlgItem(hWnd, IDC_SCREEN), &rect);

	SetStretchBltMode(hdc, HALFTONE);

	oldBitmap = SelectObject(hdcMem, hBmp);

	StretchBlt(hdc, 0, 0, rect.right - rect.left, rect.bottom - rect.top, hdcMem, 0, 0, frame->cols, frame->rows, SRCCOPY);

	SelectObject(hdcMem, oldBitmap);
	DeleteDC(hdcMem);
	ReleaseDC(hWnd, hdc);
	DeleteDC(hdc);
	DeleteObject(hBmp);
}

BOOL GetDesktopImage(Mat *frame) {
	HDC hdcTmp = CreateCompatibleDC(NULL);
	HDC hdcDesk;
	HBITMAP hBitmap;
	RECT	rect;

	if (profile.fullScreeenFlag) hTopMostWindow = GetDesktopWindow();
	if (hTopMostWindow == NULL) return FALSE;
	hdcDesk = GetWindowDC(hTopMostWindow);
	if (hdcDesk == NULL) return FALSE;
	GetClientRect(hTopMostWindow, &rect);
	hBitmap = CreateBitmap(nWidthResized, nHeightResized, 1, 32, NULL);

	SetStretchBltMode(hdcTmp, HALFTONE);
	SelectObject(hdcTmp, hBitmap);
	StretchBlt(hdcTmp, 0, 0, nWidthResized, nHeightResized, hdcDesk, 0, 0,
		rect.right - rect.left, rect.bottom - rect.top, SRCCOPY);
	ToMat(hBitmap, frame);
	ReleaseDC(hTopMostWindow, hdcDesk);
	DeleteDC(hdcDesk);
	DeleteDC(hdcTmp);
	DeleteObject(hBitmap);
	
	return TRUE;
}

void ConvertGray2BGR(Mat *src, Mat*dst)
{
	for (int i = 0; i < src->rows; i++) {
		uchar *ptrSrc = (uchar *)src->ptr(i);
		Vec4b *ptrDst = (Vec4b *)dst->ptr(i);
		for (int j = 0; j < src->cols; j++) {
			ptrDst[j] = Vec4b(ptrSrc[j], ptrSrc[j], ptrSrc[j], 0);
		}
	}
}

void ConvertFloat2BGR(Mat *src, Mat*dst)
{
	*src = *src * 255.0f / 1000.0f;
	for (int i = 0; i < src->rows; i++) {
		float *ptrSrc = (float *)src->ptr(i);
		Vec4b *ptrDst = (Vec4b *)dst->ptr(i);
		for (int j = 0; j < src->cols; j++) {
			ptrDst[j] = Vec4b(ptrSrc[j], ptrSrc[j], ptrSrc[j], 0);
		}
	}
}

void Convert324(Mat *src, Mat*dst)
{
	for (int i = 0; i < src->rows; i++) {
		Vec3b *ptrSrc = (Vec3b *)src->ptr(i);
		Vec4b *ptrDst = (Vec4b *)dst->ptr(i);
		for (int j = 0; j < src->cols; j++) {
			ptrDst[j] = Vec4b(ptrSrc[j].val[0], ptrSrc[j].val[1], ptrSrc[j].val[2], 0);
		}
	}
}

void Convert2Float(Mat *src, Mat*dst)
{
	for (int i = 0; i < src->rows; i++) {
		uchar *ptrSrc = (uchar *)src->ptr(i);
		float *ptrDst = (float *)dst->ptr(i);
		for (int j = 0; j < src->cols; j++) {
			ptrDst[j] = (float)ptrSrc[j];
		}
	}
}

void DetectSmoke(void *threadArg) {
	EnterCriticalSection(&crit);
	detectThread = GetCurrentThread();
	SetThreadPriority(detectThread, THREAD_PRIORITY_HIGHEST);
	nWidthResized = WIDTH_RESIZED * profile.camColumns;
	nHeightResized = HEIGHT_RESIZED * profile.camRows;
	char buf[256] = { 0 };
	float fps = 0;
	fCount = 0;
	Mat hist = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize * BUFFER_LEN_MAX, CV_8UC1);
	Mat smokeRgn = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
	Mat intImg = Mat::zeros(nHeightResized + 2, nWidthResized + 2, CV_32FC1);
	Mat backGround = Mat::zeros(nHeightResized, nWidthResized, CV_8UC1);
	Mat slowBG = Mat::zeros(nHeightResized, nWidthResized, CV_8UC1);
	Mat stableness = Mat::zeros(nHeightResized / BG_STABLE_BLOCK, nWidthResized / BG_STABLE_BLOCK, CV_32FC1);
	Mat maxStableness = Mat::zeros(nHeightResized / BG_STABLE_BLOCK, nWidthResized / BG_STABLE_BLOCK, CV_32FC1);
	Mat stableFlag = Mat::zeros(nHeightResized / BG_STABLE_BLOCK, nWidthResized / BG_STABLE_BLOCK, CV_8UC1);
	Mat wBG = Mat::ones(nHeightResized / 2, nWidthResized / 2, CV_32FC1);
	Mat wFG = Mat::ones(nHeightResized / 2, nWidthResized / 2, CV_32FC1);
	Mat wBGIntg = Mat::ones(nHeightResized / 2 + 2, nWidthResized / 2 + 2, CV_32FC1);
	Mat wFGIntg = Mat::ones(nHeightResized / 2 + 2, nWidthResized / 2 + 2, CV_32FC1);
	Mat wDF = Mat::ones(nHeightResized / blockSize, nWidthResized / blockSize, CV_32FC1);
	Mat wDFSlow = Mat::ones(nHeightResized / blockSize, nWidthResized / blockSize, CV_32FC1);
	Mat motionBlob = Mat::ones(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
	Mat motionIntg = Mat::ones(nHeightResized / blockSize + 2, nWidthResized / blockSize + 2, CV_32FC1);
	Mat motionMask = Mat::ones(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
	Mat rgn[2];
	Mat smoke[2];
	Mat SD;
	VideoCapture source("D:\\Backups\\TestVideos\\a9.avi");
	for (int i = 0; i < 2; i++) {
		rgn[i] = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
		smoke[i] = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
	}
	stopFlag = false;
	startFalg = true;
	int ticks = 30;
	int Sticks = 0;
	float delay = 1; 
	HWND hWnd = (HWND)threadArg;
	int tickCount = 0;
	float totalFPS = 0;
	float avgFPS = FRAME_PER_SECOND;

	while(stopFlag == false) {
		Sticks = GetTickCount();
		
		Mat origin;
		Mat gray;
		Mat blurImg;
		Mat fBG; Mat fFG;
		
		Mat tWBG = Mat::zeros(nHeightResized, nWidthResized, CV_32FC1);
		Mat tWFG = Mat::zeros(nHeightResized, nWidthResized, CV_32FC1);
		HWND fg = GetForegroundWindow();
		if (fg != hWnd && fg != NULL) {
			if (fg != hTopMostWindow && profile.fullScreeenFlag == false) {
				fCount = 0;
				startFalg = true;
				hTopMostWindow = fg;
				hist.release();
				smokeRgn.release();
				stableness.release();
				maxStableness.release();
				for (int i = 0; i < 2; i++) {
					rgn[i].release();
					smoke[i].release();
					rgn[i] = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
					smoke[i] = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
				}
				hist = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize * BUFFER_LEN_MAX, CV_8UC1);
				smokeRgn = Mat::zeros(nHeightResized / blockSize, nWidthResized / blockSize, CV_8UC1);
				stableness = Mat::zeros(nHeightResized / BG_STABLE_BLOCK, nWidthResized / BG_STABLE_BLOCK, CV_32FC1);
				maxStableness = Mat::zeros(nHeightResized / BG_STABLE_BLOCK, nWidthResized / BG_STABLE_BLOCK, CV_32FC1);
				stableFlag = Mat::zeros(nHeightResized / BG_STABLE_BLOCK, nWidthResized / BG_STABLE_BLOCK, CV_8UC1);
			}
		}
		if(!GetDesktopImage(&origin)) goto _final;
		if (origin.empty()) goto _final;
		resize(origin, origin, Size(nWidthResized, nHeightResized));
		float sX = origin.cols / profile.camColumns;
		float sY = origin.rows / profile.camRows;
		for (int i = 0; i < profile.camRows; i++) {
			for (int j = 0; j < profile.camColumns; j++) {
				rectangle(origin, Rect(j * sX, i * sY, sX, sY), Scalar(255, 255, 0), 3);
			}
		}
		cvtColor(origin, gray, CV_BGR2GRAY);

		blur(gray, blurImg, Size(BLUR_AMOUNT, BLUR_AMOUNT));
		if (startFalg) {
			//blurImg.copyTo(backGround);
			//blurImg.copyTo(slowBG);
			startFalg = false;
		}
		for (int i = 0; i < blurImg.rows; i++)
		{
			uchar *ptrBlur = (uchar *)blurImg.ptr(i);
			uchar *ptrBG = (uchar *)backGround.ptr(i);
			uchar *ptrMotion = (uchar*)motionMask.ptr(i / blockSize);

			for (int j = 0; j < blurImg.cols; j++)
			{
				if (ptrBG[j] > ptrBlur[j]) {
					ptrBG[j]--;
				}
				else if (ptrBG[j] < ptrBlur[j])
				{
					ptrBG[j]++;
				}
				if (abs((int)ptrBG[j] - (int)ptrBlur[j]) >= 5)
				{
					ptrMotion[j / blockSize] = 1;
				}
				else
				{
					ptrMotion[j / blockSize] = 0;
				}
			}
		}

		MakeIntegral(&motionMask, &motionIntg);
		for (int i = 0; i < motionMask.rows; i += BG_STABLE_BLOCK / blockSize) {
			float *ptrmIntg1 = (float *)motionIntg.ptr(i);
			float *ptrmIntg2 = (float *)motionIntg.ptr(i + (BG_STABLE_BLOCK / blockSize));
			float *ptrStbl = (float *)stableness.ptr(i / (BG_STABLE_BLOCK / blockSize));
			float *ptrMStbl = (float *)maxStableness.ptr(i / (BG_STABLE_BLOCK / blockSize));
			uchar *ptrStbFlag = (uchar *)stableFlag.ptr(i / (BG_STABLE_BLOCK / blockSize));
			for (int j = 0; j < motionMask.cols; j += BG_STABLE_BLOCK / blockSize) {
				float a = ptrmIntg1[j] + ptrmIntg2[j + BG_STABLE_BLOCK / blockSize] - ptrmIntg1[j + BG_STABLE_BLOCK / blockSize] - ptrmIntg2[j];
				if (a > 0) {
					ptrStbl[j / (BG_STABLE_BLOCK / blockSize)] = 0;
				}
				else{
					if (ptrStbl[j / (BG_STABLE_BLOCK / blockSize)] < MAX_STABLE_TIME) ptrStbl[j / (BG_STABLE_BLOCK / blockSize)]++;
				}
				if (ptrStbl[j / (BG_STABLE_BLOCK / blockSize)] >= ptrMStbl[j / (BG_STABLE_BLOCK / blockSize)]) {
					ptrMStbl[j / (BG_STABLE_BLOCK / blockSize)] = ptrStbl[j / (BG_STABLE_BLOCK / blockSize)];
				}
				
				if (ptrMStbl[j / (BG_STABLE_BLOCK / blockSize)] < profile.bgDetectTime) {
					rectangle(origin, Rect(j * blockSize + 1, i * blockSize + 1, BG_STABLE_BLOCK - 2, BG_STABLE_BLOCK - 2), Scalar(255, 0, 0));
				}
				else
				{
					if (ptrStbFlag[j / (BG_STABLE_BLOCK / blockSize)] == 0)
					{
						for (int x = 0; x < BG_STABLE_BLOCK; x++) {
							uchar *ptrBlur = (uchar *)blurImg.ptr(i * blockSize + x);
							uchar *ptrSBG = (uchar *)slowBG.ptr(i * blockSize + x);
							for (int y = 0; y < BG_STABLE_BLOCK; y++) {
								ptrSBG[j * blockSize + y] = ptrBlur[j * blockSize + y];
							}
						}
						ptrStbFlag[j / (BG_STABLE_BLOCK / blockSize)] = 1;
					}
				}
			}
		}
		
		fCount++;
		if (fCount > avgFPS * 2.0f) {
			for (int i = 0; i < blurImg.rows; i++)
			{
				uchar *ptrBlur = (uchar *)blurImg.ptr(i);
				uchar *ptrBG = (uchar *)slowBG.ptr(i);
				for (int j = 0; j < blurImg.cols; j++)
				{
					if (ptrBG[j] > ptrBlur[j]) {
						ptrBG[j]--;
					}
					else if (ptrBG[j] < ptrBlur[j])
					{
						ptrBG[j]++;
					}
				}
			}
			fCount = 0;
		}
		backGround.convertTo(fBG, CV_32FC1);
		blurImg.convertTo(fFG, CV_32FC1);
		cvHaarWavelet(fBG, tWBG, 1);
		cvHaarWavelet(fFG, tWFG, 1);

		MakeEnergy(&tWBG, &wBG);
		MakeEnergy(&tWFG, &wFG);

		//fBG.release(); tWBG.release(); fFG.release(); tWFG.release();
		MakeIntegral(&wBG, &wBGIntg);
		MakeIntegral(&wFG, &wFGIntg);
		for (int i = 0; i < wBG.rows; i += blockSize / 2) {
			int r2 = i + WAVE_BLUR_AMOUNT < wBG.rows ? i + WAVE_BLUR_AMOUNT : wBG.rows - 1;
			int r1 = i - WAVE_BLUR_AMOUNT >= 0 ? i - WAVE_BLUR_AMOUNT : 0;
			float *ptrDF = (float *)wDF.ptr(i / (blockSize / 2));
			float *ptrBG1 = (float *)wBGIntg.ptr(r1);
			float *ptrFG1 = (float *)wFGIntg.ptr(r1);
			float *ptrBG2 = (float *)wBGIntg.ptr(r2);
			float *ptrFG2 = (float *)wFGIntg.ptr(r2);

			for (int j = 0; j < wBG.cols; j += blockSize / 2) {
				int c1 = j - WAVE_BLUR_AMOUNT >= 0 ? j - WAVE_BLUR_AMOUNT : 0;
				int c2 = j + WAVE_BLUR_AMOUNT < wBG.cols ? j + WAVE_BLUR_AMOUNT : wBG.cols - 1;
				float sum1 = ptrBG1[c1] + ptrBG2[c2] - ptrBG1[c2] - ptrBG2[c1];
				float sum2 = ptrFG1[c1] + ptrFG2[c2] - ptrFG1[c2] - ptrFG2[c1];

				if (sum1 != 0)
					ptrDF[j / (blockSize / 2)] = sum2 / sum1;
				else
					ptrDF[j / (blockSize / 2)] = 1;
			}
		}

		slowBG.convertTo(fBG, CV_32FC1);
		cvHaarWavelet(fBG, tWBG, 1);
		MakeEnergy(&tWBG, &wBG);
		MakeIntegral(&wBG, &wBGIntg);
		for (int i = 0; i < wBG.rows; i += blockSize / 2) {
			int r2 = i + WAVE_S_BLUR_AMOUNT < wBG.rows ? i + WAVE_S_BLUR_AMOUNT : wBG.rows - 1;
			int r1 = i - WAVE_S_BLUR_AMOUNT >= 0 ? i - WAVE_S_BLUR_AMOUNT : 0;
			float *ptrDF = (float *)wDFSlow.ptr(i / (blockSize / 2));
			float *ptrBG1 = (float *)wBGIntg.ptr(r1);
			float *ptrFG1 = (float *)wFGIntg.ptr(r1);
			float *ptrBG2 = (float *)wBGIntg.ptr(r2);
			float *ptrFG2 = (float *)wFGIntg.ptr(r2);

			for (int j = 0; j < wBG.cols; j += blockSize / 2) {
				int c1 = j - WAVE_S_BLUR_AMOUNT >= 0 ? j - WAVE_S_BLUR_AMOUNT : 0;
				int c2 = j + WAVE_S_BLUR_AMOUNT < wBG.cols ? j + WAVE_S_BLUR_AMOUNT : wBG.cols - 1;
				float sum1 = ptrBG1[c1] + ptrBG2[c2] - ptrBG1[c2] - ptrBG2[c1];
				float sum2 = ptrFG1[c1] + ptrFG2[c2] - ptrFG1[c2] - ptrFG2[c1];

				if (sum1 != 0)
					ptrDF[j / (blockSize / 2)] = sum2 / sum1;
				else
					ptrDF[j / (blockSize / 2)] = 1;
			}
		}
		fBG.release(); tWBG.release(); fFG.release(); tWFG.release();

		blurImg.release();
		MakeIntegral(&gray, &intImg);
		quadFilterRate = pow(2, 1);

		for (int i = 0; i < gray.rows; i += blockSize) {
			uchar *ptrHist = (uchar *)hist.ptr(i / blockSize);
			float *ptrIntg1 = (float *)intImg.ptr(i);
			float *ptrIntg2 = (float *)intImg.ptr(i + quadFilterRate);

			for (int j = 0; j < gray.cols; j += blockSize) {
				for (int n = 0; n < BUFFER_LEN_MAX - 1; n++) {
					ptrHist[j / blockSize * BUFFER_LEN_MAX + n] = ptrHist[j / blockSize * BUFFER_LEN_MAX + n + 1];
				}
				float h = (ptrIntg1[j] + ptrIntg2[j + quadFilterRate]
					- ptrIntg1[j + quadFilterRate] - ptrIntg2[j]) / (quadFilterRate * quadFilterRate);
				ptrHist[j / blockSize * BUFFER_LEN_MAX + BUFFER_LEN_MAX - 1] = (uchar)h;
			}
		}
		for (int bfc = bufferLenStartVal; bfc <= bufferLenStartVal + 1; bfc++) {
			int scale = bfc - bufferLenStartVal + 1;
			bufferLength = (int)((float)scale * avgFPS * 32.0f / FRAME_PER_SECOND) > BUFFER_LEN_MAX ? BUFFER_LEN_MAX : (int)((float)scale * avgFPS * 32.0f / FRAME_PER_SECOND);
			int delta = 6 * bufferLength / pow(2, bfc);
			for (int i = 0; i < gray.rows; i += blockSize) {
				uchar *ptrHist = (uchar *)hist.ptr(i / blockSize);
				uchar *ptrRgn = (uchar *)rgn[bfc - bufferLenStartVal].ptr(i / blockSize);
				float *ptrWdf = (float *)wDF.ptr(i / blockSize);
				float *ptrWdfSlow = (float *)wDFSlow.ptr(i / blockSize);

				for (int j = 0; j < gray.cols; j += blockSize) {
					int rj = j / blockSize * BUFFER_LEN_MAX;
					float minV = 255;
					float maxV = 0;
					for (int n = (BUFFER_LEN_MAX - bufferLength); n < BUFFER_LEN_MAX; n++) {
						if (n == 0) {
							minV = ptrHist[rj + n];
							maxV = ptrHist[rj + n];
						}
						else{
							if (minV > ptrHist[rj + n]) minV = ptrHist[rj + n];
							if (maxV < ptrHist[rj + n]) maxV = ptrHist[rj + n];
						}
					}
					if (maxV - minV < MOTION_TRIGER / 3) {
						continue;
					}
					bool smokeFlag = false;
					bool motionFlag = false;
					int m = (j / blockSize) * BUFFER_LEN_MAX + BUFFER_LEN_MAX - bufferLength;
					int motionCount = 0;
					int smokeCount1 = 0;
					int smokeCount2 = 0;
					int noiseCount1 = 0;
					int noiseCount2 = 0;

					for (int n = 0; n < bufferLength - 3; n++) {
						if (abs(ptrHist[m + n] - ptrHist[m + n + 1]) >= MOTION_TRIGER
							|| abs(ptrHist[m + n] - ptrHist[m + n + 2]) >= MOTION_TRIGER
							|| abs(ptrHist[m + n] - ptrHist[m + n + 3]) >= MOTION_TRIGER) {
							motionCount++;
							motionFlag = true;
							ptrRgn[j / blockSize] = MOTION_DELAY;
							break;
						}
					}
					if (motionFlag) continue;
					
					for (int n = 0; n < bufferLength - delta; n++) {
						int d = (int)ptrHist[m + n + delta] - (int)ptrHist[m + n];
						if (d >= SMOKE_TRIGER) {
							smokeCount2++;
						}
						d = (int)ptrHist[m + n] - (int)ptrHist[m + n + delta];
						if (d >= SMOKE_TRIGER) {
							smokeCount1++;
						}
					}
					if (ptrWdf[j / blockSize] < WAVELET_TRIGGER && ptrWdfSlow[j / blockSize] < WAVELET_S_TRIGGER)
					{
						if (smokeCount2 > bufferLength / 3)
						{
							if (ptrRgn[j / blockSize] == 0) ptrRgn[j / blockSize] = 1;
						}
					}
				}
			}
		}

		for (int i = 0; i < smokeRgn.rows; i++) {
			uchar *ptrSmokeRgn = (uchar *)smokeRgn.ptr(i);
			uchar *ptrMotion = (uchar*)motionBlob.ptr(i);
			uchar *ptrMask = (uchar*)motionMask.ptr(i);
			uchar *ptrRgn[2];
			ptrRgn[0] = (uchar *)(rgn[0].ptr(i));
			ptrRgn[1] = (uchar *)(rgn[1].ptr(i));
			for (int j = 0; j < smokeRgn.cols; j++) {
				
				if (ptrRgn[0][j] > 1) {
					ptrRgn[0][j]--;
					if (ptrRgn[0][j] == 1) ptrRgn[0][j] = 0;
				}
				if (ptrRgn[1][j] > 1) {
					ptrRgn[1][j]--;
					if (ptrRgn[1][j] == 1) ptrRgn[1][j] = 0;
				}
				
				if ((ptrRgn[0][j] > 1 || ptrRgn[1][j] > 1) && ptrMask[j] == 1)  {
					ptrMotion[j] = 1;
				}
				else
				{
					ptrMotion[j] = 0;
				}
				if (ptrMotion[j] == 1 && bShowMotionFlag == true)
					rectangle(origin, Rect(j * blockSize, i * blockSize, blockSize, blockSize), Scalar(0, 255, 0));
			}
		}
		MakeIntegral(&motionBlob, &motionIntg);
		float smokeCount = 0;
		for (int n = 0; n < 2; n++) {
			for (int i = 0; i < rgn[n].rows; i++) {
				uchar *ptrRgn = (uchar *)rgn[n].ptr(i);
				uchar *ptrSmoke = (uchar *)smoke[n].ptr(i);
				uchar *ptrSmokeRgn = (uchar *)smokeRgn.ptr(i);
				float *ptrMstbl = (float *)maxStableness.ptr(i * blockSize / BG_STABLE_BLOCK);

				int r1 = i - MOTION_EXTEND >= 0 ? i - MOTION_EXTEND : 0;
				int r2 = i + MOTION_EXTEND < smokeRgn.rows ? i + MOTION_EXTEND : smokeRgn.rows - 1;
				float *ptr1 = (float *)motionIntg.ptr(r1);
				float *ptr2 = (float *)motionIntg.ptr(r2);

				for (int j = 0; j < rgn[n].cols; j++) {
					int c1 = j - MOTION_EXTEND >= 0 ? j - MOTION_EXTEND : 0;
					int c2 = j + MOTION_EXTEND < smokeRgn.cols ? j + MOTION_EXTEND : smokeRgn.cols - 1;
					float mr = ptr1[c1] + ptr2[c2] - ptr1[c2] - ptr2[c1];
					if (mr > MX_TRIGGER) {
						if (ptrRgn[j] == 1 || ptrSmoke[j] > 0 || ptrSmokeRgn[j] == 1)
						{
							ptrRgn[j] = 0; ptrSmoke[j] = 0; ptrSmokeRgn[j] = 0;
						}
					}
					else
					{
						if (ptrRgn[j] == 1 && ptrSmoke[j] == 0) {
							ptrSmoke[j] = SMOKE_DELAY;
						}
						if (ptrSmoke[j] > 0) {
							ptrSmoke[j]--;
							if (ptrSmoke[j] == 0) ptrRgn[j] = 0;
						}
						if (ptrSmoke[j] > 0) {
							ptrSmokeRgn[j] = 1;
						}
						else{
							ptrSmokeRgn[j] = 0;
						}
					}
					if (ptrMstbl[j * blockSize / BG_STABLE_BLOCK] < profile.bgDetectTime) {
						ptrRgn[j] = 0; ptrSmoke[j] = 0; ptrSmokeRgn[j] = 0;
					}
				}
			}
		}

		
		labelBlobs(smokeRgn);
		for (int i = 0; i < blobs.size(); i++) {
			if (blobs[i].size() < DETECTION_TRIGGER) {
				for (int j = 0; j < blobs[i].size(); j++) {
					smokeRgn.at<uchar>(blobs[i][j].y, blobs[i][j].x) = 0;
				}
			}
			else{
				for (int j = 0; j < blobs[i].size(); j++) {
					rectangle(origin, Rect(blobs[i][j].x * blockSize, blobs[i][j].y * blockSize, blockSize, blockSize), Scalar(0, 0, 255));
				}
			}
		}

		MakeIntegral(&smokeRgn, &motionIntg);
		sX = sX / blockSize;
		sY = sY / blockSize;

		int nCamNo = 0;
		for (int i = 0; i < profile.camRows; i++) {
			int r1 = i * sY;
			int r2 = i * sY + sY < motionIntg.rows ? i * sY + sY : motionIntg.rows - 1;
			float *ptr1 = (float *)motionIntg.ptr(r1);
			float *ptr2 = (float *)motionIntg.ptr(r2);
			for (int j = 0; j < profile.camColumns; j++) {
				int c1 = j * sX;
				int c2 = j * sX + sX < motionIntg.cols ? j * sX + sX : motionIntg.cols - 1;
				nCamNo++;

				smokeCount = ptr1[c1] + ptr2[c2] - ptr1[c2] - ptr2[c1];
				if (profile.camProperties[nCamNo - 1] == TRUE)
					smokeCount = sY * (float)blockSize - smokeCount * sY * (float)blockSize / 30.0f >= 0 ? sY  * (float)blockSize - smokeCount * sY * (float)blockSize / 30.0f : 0;
				else
					smokeCount = sY * (float)blockSize - smokeCount * sY * (float)blockSize / 100.0f >= 0 ? sY  * (float)blockSize - smokeCount * sY * (float)blockSize / 100.0f : 0;

				for (int k = smokeCount; k < sY * blockSize; k++) {
					rectangle(origin, Rect(c2 * blockSize - sX / 10 - 2, r1 * blockSize + k - 3, sX / 10, 1), Scalar(255 - (sY * blockSize - k) * 255 / (sY * blockSize), 0, 50 + (sY * blockSize - k) * 200 / (sY * blockSize)));
				}
			}
		}
		//SD = Mat(origin.rows, origin.cols, CV_8UC4);
		//ConvertFloat2BGR(&wDFSlow, &SD);
		//Convert324(&origin, &SD);
		ShowResultImage(hWnd, &origin);
		//SD.release();
		origin.release();
		gray.release();
		
	_final:
		ticks = GetTickCount() - Sticks;
		delay = 1000.0f / FRAME_PER_SECOND - (float)ticks < 0 ? 0 : 1000.0f / FRAME_PER_SECOND - (float)ticks;
		if (ticks < 1)
			fps = FRAME_PER_SECOND;
		else
			fps = 1000.0f / (float)(ticks + delay);
		tickCount++;
		totalFPS += fps;
		if (tickCount > avgFPS) {
			avgFPS = totalFPS / tickCount;
			tickCount = 0;
			totalFPS = 0;
		}
		sprintf(buf, "FPS : %f    Delay : %f     Ticks : %d", avgFPS, delay, ticks);
		SetDlgItemTextA(hWnd, IDC_FPS, buf);
		//delay = delay / 3;
		Sleep(delay);
		//Sleep(1);

	}
	hist.release();
	for (int i = 0; i < 2; i++) {
		smoke[i].release();
		rgn[i].release();
	}
	smokeRgn.release();
	backGround.release();
	stableness.release();
	maxStableness.release();
	stableFlag.release();
	wDF.release();
	wBG.release();
	wFG.release();
	wBGIntg.release();
	wFGIntg.release();
	intImg.release();
	motionBlob.release();
	motionIntg.release();
	motionMask.release();
	slowBG.release();
	wDFSlow.release();
	LeaveCriticalSection(&crit);
	_endthread();
}

void WaitCursor()
{
	static HCURSOR hcWait;
	if (hcWait == NULL)
		hcWait = LoadCursor(NULL, IDC_WAIT);
	SetCursor(hcWait);
	hCursor = hcWait;
}

void NormalCursor()
{
	static HCURSOR hcArrow;
	if (hcArrow == NULL)
		hcArrow = LoadCursor(NULL, IDC_ARROW);
	SetCursor(hcArrow);
	hCursor = NULL;
}

LRESULT CALLBACK ProfileDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	static int selCam = -1;
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	HDC hdc;

	switch (message)
	{
	case WM_INITDIALOG:
		{
			char buf[256];
			sprintf(buf, "%d", profile.camRows);
			SetDlgItemTextA(hWnd, IDC_ROWS, buf);
			sprintf(buf, "%d", profile.camColumns);
			SetDlgItemTextA(hWnd, IDC_COLUMNS, buf);
			sprintf(buf, "%f", (float)profile.bgDetectTime / (FRAME_PER_SECOND * 60.0f));
			SetDlgItemTextA(hWnd, IDC_BG_DETECT_TIME, buf);
			if (profile.fullScreeenFlag) {
				SendMessage(GetDlgItem(hWnd, IDC_DESKTOP), BM_SETCHECK, BST_CHECKED, 0);
				SendMessage(GetDlgItem(hWnd, IDC_WINDOW), BM_SETCHECK, BST_UNCHECKED, 0);
			}
			else{
				SendMessage(GetDlgItem(hWnd, IDC_DESKTOP), BM_SETCHECK, BST_UNCHECKED, 0);
				SendMessage(GetDlgItem(hWnd, IDC_WINDOW), BM_SETCHECK, BST_CHECKED, 0);
			}
			SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_RESETCONTENT, 0, 0);
			SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_RESETCONTENT, 0, 0);

			int camNo = 0;
			for (int i = 0; i < profile.camColumns * profile.camRows; i++) {
				camNo++;
				sprintf(buf, "Camera%d", camNo);
				int index = SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_ADDSTRING, 0, (LPARAM)buf);
				SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_SETITEMDATA, index, (LPARAM)i);
			}
			SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_SETCURSEL, 0, (LPARAM)0);
			int index = SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_ADDSTRING, 0, (LPARAM)"Indoor Short-Sight Camera");
			SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_SETITEMDATA, index, (LPARAM)0);
			index = SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_ADDSTRING, 0, (LPARAM)"Outdoor Far-Sight Camera");
			SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_SETITEMDATA, index, (LPARAM)1);
			if (profile.camProperties[0] == FALSE)
				SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_SETCURSEL, 0, (LPARAM)0);
			else
				SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_SETCURSEL, 1, (LPARAM)0);
			selCam = 0;
		}
		return 1;
	case WM_COMMAND:
		wmId = LOWORD(wParam);
		wmEvent = HIWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		case IDC_SET:
			{
				char buf[256];
				int val1 = 0;
				int val2 = 0;

				GetDlgItemTextA(hWnd, IDC_ROWS, buf, 255);
				val1 = atoi(buf);
				if (val1 < 0 || val1 > 64) {
					MessageBoxA(hWnd, "Incorrect Camera Rows!", "Warnning", MB_OK);
					break;
				}
				GetDlgItemTextA(hWnd, IDC_COLUMNS, buf, 255);
				val2 = atoi(buf);
				if (val2 < 0 || val2 > 64) {
					MessageBoxA(hWnd, "Incorrect Camera Columns!", "Warnning", MB_OK);
					break;
				}
				if (val1 * val2 < 0 || val1 * val2 > 64) {
					MessageBoxA(hWnd, "Total Cameras must be between 0 and 64!", "Warnning", MB_OK);
					break;
				}
				SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_RESETCONTENT, 0, 0);

				int camNo = 0;
				for (int i = 0; i < val1 * val2; i++) {
					camNo++;
					sprintf(buf, "Camera%d", camNo);
					int index = SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_ADDSTRING, 0, (LPARAM)buf);
					SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_SETITEMDATA, index, (LPARAM)i);
				}
				SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_SETCURSEL, 0, (LPARAM)0);
			}
			break;
		case IDCANCEL:
			EndDialog(hWnd, 0);
			break;
		case IDOK:
			{
				char buf[256];
				int val = 0;
				GetDlgItemTextA(hWnd, IDC_ROWS, buf, 255);
				val = atoi(buf);
				if (val < 0 || val > 64) {
					MessageBoxA(hWnd, "Incorrect Camera Rows!", "Warnning", MB_OK);
					break;
				}
				profile.camRows = val;
				GetDlgItemTextA(hWnd, IDC_COLUMNS, buf, 255);
				val = atoi(buf);
				if (val < 0 || val > 64) {
					MessageBoxA(hWnd, "Incorrect Camera Columns!", "Warnning", MB_OK);
					break;
				}
				profile.camColumns = val;
				if (profile.camColumns * profile.camRows < 0 || profile.camColumns * profile.camRows > 64) {
					MessageBoxA(hWnd, "Total Cameras must be between 0 and 64!", "Warnning", MB_OK);
					break;
				}
				GetDlgItemTextA(hWnd, IDC_BG_DETECT_TIME, buf, 255);
				float fval = atof(buf);
				if (fval <= 0 || fval > 30) {
					MessageBoxA(hWnd, "Detection Start Time must be between 0 and 30 minutes!", "Warnning", MB_OK);
					break;
				}
				profile.bgDetectTime =(int)(FRAME_PER_SECOND * 60.0f * fval);
				if (SendMessage(GetDlgItem(hWnd, IDC_DESKTOP), BM_GETCHECK, 0, 0) == BST_CHECKED){
					profile.fullScreeenFlag = TRUE;
				}
				if (SendMessage(GetDlgItem(hWnd, IDC_WINDOW), BM_GETCHECK, 0, 0) == BST_CHECKED){
					profile.fullScreeenFlag = FALSE;
					hTopMostWindow = NULL;
				}
			}
			EndDialog(hWnd, 1);
			break;
		case IDC_CAM_NO:
			selCam = SendMessage(GetDlgItem(hWnd, IDC_CAM_NO), CB_GETCURSEL, 0, (LPARAM)0);
			break;
		case IDC_CAM_PTY:
			if (SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_GETCURSEL, 0, (LPARAM)0) == 0)
				profile.camProperties[selCam] = FALSE;
			else if (SendMessage(GetDlgItem(hWnd, IDC_CAM_PTY), CB_GETCURSEL, 0, (LPARAM)0) == 1)
				profile.camProperties[selCam] = TRUE;
			break;
		default:
			break;
		}
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		return 1;
	default:
		return 0;
	}
	return 0;
}

LRESULT CALLBACK MonitorDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	HDC hdc;

	switch (message)
	{
	case WM_INITDIALOG:
		{
			EnableWindow(GetDlgItem(hWnd, IDC_STOP), FALSE);
		}
		return 1;
	case WM_COMMAND:
		wmId = LOWORD(wParam);
		wmEvent = HIWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		case IDC_MOTION_VIEW:
			if (SendMessage(GetDlgItem(hWnd, IDC_MOTION_VIEW), BM_GETCHECK, 0, 0) == BST_CHECKED)
				bShowMotionFlag = true;
			else
				bShowMotionFlag = false;
			break;
		case IDC_SETTING:
			DialogBoxA(hInst, MAKEINTRESOURCEA(IDD_SETTINGS), hWnd, (DLGPROC)ProfileDlgProc);
			break;
		case IDCANCEL:
			if (detectThread != NULL) {
				WaitCursor();
				stopFlag = true;
				WaitForSingleObject(detectThread, 100);
				detectThread = NULL;
				NormalCursor();
			}
			exit(0);
			break;
		case IDC_START_DETECT:
			WaitCursor();
			EnableWindow(GetDlgItem(hWnd, IDC_START_DETECT), FALSE);
			EnableWindow(GetDlgItem(hWnd, IDC_SETTING), FALSE);
			EnableWindow(GetDlgItem(hWnd, IDC_STOP), TRUE);
			_beginthread(DetectSmoke, 0, hWnd);
			NormalCursor();
			break;
		case IDC_STOP:
			WaitCursor();
			EnableWindow(GetDlgItem(hWnd, IDC_STOP), FALSE);
			WaitForSingleObject(detectThread, 100);
			detectThread = NULL;
			cleanup();
			EnableWindow(GetDlgItem(hWnd, IDC_START_DETECT), TRUE);
			EnableWindow(GetDlgItem(hWnd, IDC_SETTING), TRUE);
			NormalCursor();
			break;
		default:
			break;
		}
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		return 1;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 1;
	default:
		return 0;
	}
	return 0;
}

