// HandTracking.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <cstdlib>



#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

// from example
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;

#define VIDEO_FILE	"video.avi"
#define VIDEO_FORMAT	CV_FOURCC('M','J','P','G')
#define NUM_FINGERS	5
#define NUM_DEFECTS	8


	int iWidth;
	int iHeight;

	int iRows;
	int iCols;



	double fWidth;
	double fHeight;

	double ratio;

	Mat mask;


class htCtx {
public:
	VideoCapture capture;
	VideoWriter writer;


	Mat image;		 /* Input image */
	Mat thr_image;   /* After filtering and thresholding */
	Mat temp_image1; /* Temporary image (1 channel) */
	Mat temp_image3; /* Temporary image (3 channels) */
	Mat kernel;
	
	vector<Point> contour;
	vector<Point> hull;


	CvPoint		hand_center;

	//CvPoint		*fingers;	/* Detected fingers positions */
	//CvPoint		*defects;	/* Convexity defects depth points */

	vector<CvPoint>	fingers;	/* Detected fingers positions */
	vector<CvPoint> defects;	/* Convexity defects depth points */


	MemStorage	hull_st;
	MemStorage	contour_st;
	MemStorage	temp_st;
	MemStorage	defects_st;

	int		num_fingers;
	int		hand_radius;
	int		num_defects;

	void init();
	void initCapture();
	void initWindows();
	void filter_and_threshold();
	void find_contour();
	void find_convex_hull();
	void find_fingers();
	void display();

private:


};

void htCtx::init(){

	/*
	ctx->thr_image = cvCreateImage(cvGetSize(ctx->image), 8, 1);
	ctx->temp_image1 = cvCreateImage(cvGetSize(ctx->image), 8, 1);
	ctx->temp_image3 = cvCreateImage(cvGetSize(ctx->image), 8, 3);
	ctx->kernel = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_RECT, NULL);
	*/


	//cvCreateStructuringElementEx(

	thr_image.create(image.size(),CV_8UC1);
	temp_image1.create(image.size(),CV_8UC1);
	temp_image3.create(image.size(),CV_8UC3);
	kernel.create(Size(9,9),CV_8UC1);
	
	/*ctx->fingers = new CvPoint[NUM_FINGERS];
	ctx->defects = new CvPoint[NUM_DEFECTS];*/

	fingers.resize(NUM_FINGERS);
	defects.resize(NUM_DEFECTS);

}

void htCtx::initCapture(){
	capture = VideoCapture(0);

	if (!capture.isOpened()) {
		printf("Error initializing capture\n");
		return;
	}
	
	Mat frame;
	capture >> frame;
	image = new IplImage(frame.clone());
}

void htCtx::initWindows(){
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	namedWindow("thresholded", CV_WINDOW_AUTOSIZE);
	moveWindow("output", 50, 50);
	moveWindow("thresholded", 700, 50);

}

void htCtx::filter_and_threshold(){
	//cvSmooth(ctx->image, ctx->temp_image3, CV_GAUSSIAN, 11, 11, 0, 0);
	/* Remove some impulsive noise */
	//cvSmooth(ctx->temp_image3, ctx->temp_image3, CV_MEDIAN, 11, 11, 0, 0);

	GaussianBlur(image, temp_image3, Size(11,11), 0, 0);
	medianBlur(temp_image3, temp_image3, 11);
	

	//cvCvtColor(ctx->temp_image3, ctx->temp_image3, CV_BGR2HSV);
	cvtColor(temp_image3,temp_image3, CV_BGR2HSV);

	/*cvInRangeS(ctx->temp_image3,
		   cvScalar(0, 0, 160, 0),
		   cvScalar(255, 400, 300, 255),
		   ctx->thr_image);*/

	inRange(temp_image3, Scalar(0,0,160,0), Scalar(255,400,300,255),thr_image);

	/*
	cvMorphologyEx(ctx->thr_image, ctx->thr_image, NULL, ctx->kernel,
		       CV_MOP_OPEN, 1);
	*/

	morphologyEx(thr_image, thr_image, CV_MOP_OPEN, kernel);

	//morphologyEx(thr_image, thr_image,  

}

void htCtx::find_contour(){
	double area, max_area = 0.0;
	int thresh = 100;
	int max_thresh = 255;


	Mat canny_output;
	vector<Vec4i> hierarchy;

	vector<vector<Point>> contours;
	vector<Point> contour_;
	vector<Point> tmp;

	//thr_image.copyTo(temp_image1,NULL);
	cvtColor( temp_image3, temp_image1, CV_BGR2GRAY );

	Canny( temp_image1, canny_output, thresh, thresh*2, 3 );

	//findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL,  CV_CHAIN_APPROX_SIMPLE, Point(0,0));
	findContours(temp_image1, contours, hierarchy, CV_RETR_EXTERNAL,  CV_CHAIN_APPROX_SIMPLE, Point(0,0));
	
	/* cvFindContours modifies input image, so make a copy */
	
	/* Select contour having greatest area */



	/*for (vector<vector<Point>>::iterator cnt = contours.begin(); cnt != contours.end(); ++cnt){
		area = abs(contourArea((InputArray)cnt));
		if (area > max_area) {
			max_area = area;
			contour_ = (vector<Point>)cnt;
		}
	}*/

	for (int i=0; i < contours.size(); i++){
		area = abs(contourArea(contours[i]));
		if (area > max_area) {
			max_area = area;
			contour_ = contours[i];
		}
	}

	if (!contour_.empty()) {
		/*contour = cvApproxPoly(contour, sizeof(CvContour),
				       ctx->contour_st, CV_POLY_APPROX_DP, 2,
				       1);*/

		approxPolyDP(contour_, contour_ , 2, false);

		contour = contour_;

	}
}

void htCtx::find_convex_hull(){

	//CvSeq *defects;
	vector<Vec4i> defects;

	//CvConvexityDefect *defect_array;
	vector<Vec4i> defect_array;


	int i;
	int x = 0, y = 0;
	int dist = 0;


	if (!contour.empty()) return;

	//ctx->hull = cvConvexHull2(ctx->contour, ctx->hull_st, CV_CLOCKWISE, 0);

	convexHull(contour, hull, true);

	if (!hull.empty()) {

		/* Get convexity defects of contour w.r.t. the convex hull */
		/*defects = cvConvexityDefects(ctx->contour, ctx->hull,
					     ctx->defects_st);*/

		convexityDefects(contour,hull, defects);

		//if (defects && defects->total) {
		if (!defects.empty()) {

			/*defect_array = calloc(defects->total,
					      sizeof(CvConvexityDefect));*/

			//defect_array = new CvConvexityDefect[defects->total];

			
				
			//cvCvtSeqToArray(defects, defect_array, CV_WHOLE_SEQ);

			defect_array = defects;


			/* Average depth points to get hand center */
			/*for (i = 0; i < defects->total && i < NUM_DEFECTS; i++) {
				x += defect_array[i].depth_point->x;
				y += defect_array[i].depth_point->y;

				ctx->defects[i] = cvPoint(defect_array[i].depth_point->x,
							  defect_array[i].depth_point->y);
			}*/

			for (i = 0; i < defects.size() && i < NUM_DEFECTS; i++) {

				x += hull[defect_array[i][2]].x;
				y += hull[defect_array[i][2]].y;
			
				
			}

			x /= defects.size();
			y /= defects.size();


			num_defects = defects.size();
			hand_center = cvPoint(x, y);

			/* Compute hand radius as mean of distances of
			   defects' depth point to hand center */

			for (i = 0; i < defects.size(); i++) {

				/*int d = (x - defect_array[i].depth_point->x) *
					(x - defect_array[i].depth_point->x) +
					(y - defect_array[i].depth_point->y) *
					(y - defect_array[i].depth_point->y);*/

				int d = (x - hull[defect_array[i][2]].x) *
						(x - hull[defect_array[i][2]].x) +
						(y - hull[defect_array[i][2]].y) *
						(y - hull[defect_array[i][2]].y);

				dist += sqrt(d);
			}

			hand_radius = dist / defects.size();
			defect_array.clear();

		}
	}

}

void htCtx::find_fingers(){
	int n;
	int i;
	vector<Point> points;
	Point max_point;
	int dist1 = 0, dist2 = 0;
	int finger_distance[NUM_FINGERS + 1];

	num_fingers = 0;

	if (contour.empty() || hull.empty())
		return;

	n = contour.size();
	//points = calloc(n, sizeof(CvPoint));
	points.resize(n); // ?
	points = contour;

	//cvCvtSeqToArray(ctx->contour, points, CV_WHOLE_SEQ);

	for (i = 0; i < n; i++) {
		int dist;
		int cx = hand_center.x;
		int cy = hand_center.y;

		dist = (cx - points[i].x) * (cx - points[i].x) +
			   (cy - points[i].y) * (cy - points[i].y);

		if (dist < dist1 && dist1 > dist2 && max_point.x != 0
			&& max_point.y < image.rows - 10) {

			finger_distance[num_fingers] = dist;
			fingers[num_fingers++] = max_point;
			if (num_fingers >= NUM_FINGERS + 1)
				break;
		}

		dist2 = dist1;
		dist1 = dist;
		max_point = points[i];
	}

}

void htCtx::display()
{
	int i;

	if (num_fingers == NUM_FINGERS) {

//#if defined(SHOW_HAND_CONTOUR)
//		cvDrawContours(ctx->image, ctx->contour,
//			       CV_RGB(0,0,255), CV_RGB(0,255,0),
//			       0, 1, CV_AA, cvPoint(0,0));
//#endif


		//cvCircle(ctx->image, ctx->hand_center, 5, CV_RGB(255, 0, 255), 1, CV_AA, 0);
		//cvCircle(ctx->image, ctx->hand_center, ctx->hand_radius, CV_RGB(255, 0, 0), 1, CV_AA, 0);

		circle(image, hand_center, hand_radius, Scalar(255,0,255), 1); 

		for (i = 0; i < num_fingers; i++) {

			//cvCircle(ctx->image, ctx->fingers[i], 10, CV_RGB(0, 255, 0), 3, CV_AA, 0);
			//cvLine(ctx->image, ctx->hand_center, ctx->fingers[i], CV_RGB(255,255,0), 1, CV_AA, 0);

			circle(image, fingers[i], 10, Scalar(0,255,0), 3);
			line(image, hand_center, fingers[i], Scalar(255,255,0), 1);

		}

		for (i = 0; i < num_defects; i++) {

			//cvCircle(ctx->image, ctx->defects[i], 2, CV_RGB(200, 200, 200), 2, CV_AA, 0);
			circle(image, defects[i], 2, Scalar(200,200,200), 2 );
		}
	}

	imshow("output", image);
	imshow("thresholded", thr_image);
}


int _tmain_ht(int argc, _TCHAR* argv[])
{
	//struct ctx ctx = { };
	htCtx ctx;
	ctx.init();
	ctx.initCapture();
	ctx.initWindows();
	
	int key;

	for (;;){
		
		//ctx.capture >> ctx.image;

		ctx.image = imread("../../contourtest.png");

		ctx.filter_and_threshold();
		ctx.find_contour();
		ctx.find_convex_hull();
		ctx.find_fingers();
		ctx.display();

		imshow("output", ctx.image);

		if ((key = waitKey(5)) == (char)("q") || 
			(key = waitKey(5)) == (char)("é")) break;
	}

	return 0;
}

int _tmain_contours(int argc, _TCHAR* argv[]){

	double thresh = 100;

	Mat image = imread("../../contourtest.png");
	Mat image_grayscale;
	Mat canny_output;
	Mat image_contours;
	vector<Vec4i> hierarchy;

	vector<vector<Point>> contours;
	vector<Point> contour;
	vector<Point> tmp;
	namedWindow("input");
	namedWindow("contours");

	contours.resize(0);

	cvtColor( image, image_grayscale, CV_BGR2GRAY );
	Canny( image_grayscale, canny_output, thresh, thresh*2, 3 );

	findContours(canny_output, contours, CV_RETR_EXTERNAL, 
		CV_CHAIN_APPROX_NONE, Point(0,0));
	//findContours(image_grayscale, contours, hierarchy, CV_RETR_EXTERNAL,  CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	
	image_contours.create(image.size(), CV_8UC3);
	if (!contours.empty()) {
		
		//approxPolyDP(contour, contour, 2, false);
		//drawContours(image_contours, contours, -1, Scalar(200,200,20), 1,8,hierarchy); 


	}

	//vector<double> area;
	double area;

	int cnt = contours.size();
	//for (int i=0; i < contours.size()-1; i++){
	//	drawContours(image_contours, contours, i, Scalar(i*50 + 0 ,i*40 + 80, 250 - i*70), 1,8); 

	//	area = contourArea(contours[i]);
	//	//area.push_back(abs(contou rArea(contours[i])));
	//}
	int n = 0;
	//contour = contours[n];
	//approxPolyDP(contours[n], contours[n], 2, false);
	drawContours(image_contours, contours, -1, Scalar(250,0,0), 1,8); 

	//area = contourArea(contour);
	//drawContours(image_contours, contours, 0, Scalar(250,0,0), 1,8); 


	imshow("input", image_grayscale);
	imshow("contours", image_contours);

	waitKey();
	return 0;
}


Point2f nextStep(Point2f p, Point2f & v, int ch = 0){

	double f_min = 0.1f;
	//double f_l = sqrt(abs(p.x - fWidth/2.0)/ (fWidth/2.0));// * abs(p.x - fWidth/2.0)/ (fWidth/2.0) + f_min;
	
	double l = 5;
	double f_l = abs(p.x - fWidth/2.0)/ (fWidth/2.0) + f_min;
	Point2f res, v_res;

	//l = rand()/RAND_MAX * 0.01 + 0.01;
	int W = mask.size().width;
	int H = mask.size().height;
	double f = mask.data[(int)(p.y * (double)W/fWidth) * mask.step1() + (int)(p.x * (double)H/fHeight)*mask.channels() + ch]/256.0; 


	double d_fi = 90 * (1-f);
	double fi = (rand()/(double)RAND_MAX  - 0.5)* CV_PI/180.0 * d_fi;

	v_res.x =  cos(fi)*v.x + sin(fi)*v.y;
	v_res.y = -sin(fi)*v.x + cos(fi)*v.y;

	//v_res.x = 0.01;
	//v_res.y = 0.01;


	//res = p + (v_res)*l*f_l;

	



	res = p + (v_res)* (pow(f, 2.0) + 0.1);


	if (res.x > fWidth) res.x -= fWidth;
	if (res.y > fHeight) res.y -= fHeight;

	if (res.x <0) res.x += fWidth;
	if (res.y <0) res.y += fHeight;


	v = v_res;
	return res;

}

int _tmain(int argc, _TCHAR* argv[]){

	namedWindow("image dst");
	namedWindow("mask");

	//Mat image = imread("");
	

	Mat image;
	Mat image_dst;

	//mask = imread("D:/pic/mask.bmp");
	mask = imread("D:/pic/213744310.jpg");


	//cvtColor(mask, mask, CV_BGR2GRAY);
	imshow("mask", mask);
	image.create(mask.size()*1,CV_8UC1);

	//image = mask.clone();
	


	const int N = 200;
	vector<Point2f> pfR;
	vector<Point2f> vfR;
	vector<Point2i> piR;

	vector<Point2f> pfG;
	vector<Point2f> vfG;
	vector<Point2i> piG;

	vector<Point2f> pfB;
	vector<Point2f> vfB;
	vector<Point2i> piB;

	iWidth = image.size().width;
	iHeight = image.size().height;

	iRows = image.rows;
	iCols = image.cols;

	//image_dst.create(iRows, iCols, CV_8UC1);
	image_dst.create(iRows, iCols, CV_8UC3);



	fWidth  = 1.0*(double)iWidth/(double)iHeight;
	fHeight = 1.0;

	ratio = (double)iHeight/fHeight;
	

	Point2f pnt, vlc;

	double V = 0.005;
	for (int i=0; i<N; i++){
		//    ------------------ R  -----------------
		pnt.x = rand()*fWidth/RAND_MAX;
		pnt.y = rand()*fHeight/RAND_MAX;

		//vlc.x = rand()/RAND_MAX * V;
		//vlc.y = rand()/RAND_MAX * V;
		double d_fi = 360;
		double fi = (rand()/(double)RAND_MAX - 0.5 )* CV_PI/180.0 * d_fi;

		vlc.x = cos(fi)*V;
		vlc.y = sin(fi)*V;

		vfR.push_back(vlc);
		pfR.push_back(pnt);
		piR.push_back(Point2i(int(pfR[i].x * ratio), int(pfR[i].y * ratio)));



		//    ------------------ G  -----------------
		pnt.x = rand()*fWidth/RAND_MAX;
		pnt.y = rand()*fHeight/RAND_MAX;

		//vlc.x = rand()/RAND_MAX * V;
		//vlc.y = rand()/RAND_MAX * V;
		d_fi = 360;
		fi = (rand()/(double)RAND_MAX - 0.5 )* CV_PI/180.0 * d_fi;

		vlc.x = cos(fi)*V;
		vlc.y = sin(fi)*V;

		vfG.push_back(vlc);
		pfG.push_back(pnt);
		piG.push_back(Point2i(int(pfG[i].x * ratio), int(pfG[i].y * ratio)));


		//    ------------------ B  -----------------

		pnt.x = rand()*fWidth/RAND_MAX;
		pnt.y = rand()*fHeight/RAND_MAX;

		//vlc.x = rand()/RAND_MAX * V;
		//vlc.y = rand()/RAND_MAX * V;
		d_fi = 360;
		fi = (rand()/(double)RAND_MAX - 0.5 )* CV_PI/180.0 * d_fi;

		vlc.x = cos(fi)*V;
		vlc.y = sin(fi)*V;

		vfB.push_back(vlc);
		pfB.push_back(pnt);
		piB.push_back(Point2i(int(pfB[i].x * ratio), int(pfB[i].y * ratio)));
		//circle(image_dst, pi[i], 2, Scalar(200,0,0), 1, 8, 0);
		
	}

	//imshow("image dst", image_dst);
	//waitKey();

	int key;
	image_dst = 0;
	for (;;){
		
		//image_dst = 0;
		GaussianBlur(image_dst, image_dst, Size(3,3), 1);

		for (int i=0; i<N; i++){
			pfR[i] = nextStep(pfR[i], vfR[i], 2);
			piR[i] = Point2i(int(pfR[i].x * ratio), int(pfR[i].y * ratio));

			pfG[i] = nextStep(pfG[i], vfG[i], 1);
			piG[i] = Point2i(int(pfG[i].x * ratio), int(pfG[i].y * ratio));

			pfB[i] = nextStep(pfB[i], vfB[i], 0);
			piB[i] = Point2i(int(pfB[i].x * ratio), int(pfB[i].y * ratio));

			image_dst.data[ piR[i].y * image_dst.step1() + piR[i].x * image_dst.channels() + 0] = 220;
			image_dst.data[ piG[i].y * image_dst.step1() + piG[i].x * image_dst.channels() + 1] = 220;
			image_dst.data[ piB[i].y * image_dst.step1() + piB[i].x * image_dst.channels() + 2] = 220;
		}
		
		imshow("image dst", image_dst);

		int delay = 10;

		key = waitKey(delay);
		if (key == (char)("q") || 
			key == (char)("é"))
			break;
	}


	return 0;
}




int _tmain_(int argc, _TCHAR* argv[]){

	namedWindow("image dst");

	//Mat image = imread("");
	Mat image;
	Mat image_dst;

	image.create(500, 500,CV_8UC1);


	iWidth = image.size().width;
	iHeight = image.size().height;

	iRows = image.rows;
	iCols = image.cols;

	image_dst.create(iRows, iCols, CV_8UC1);


	fWidth  = 1.0*(double)iWidth/(double)iHeight;
	fHeight = 1.0;

	//ratio = (double)iHeight/fHeight;
	ratio = (double)fHeight/iHeight;
	

	double t = 0;

	//imshow("image dst", image_dst);
	//waitKey();

	int key;
	double dt = 0.3;
	double kx = 40.0;
	double ky = 1;
	for (;;){
		
		t +=dt;
		//image_dst = 0;

		for (int col = 0; col < iWidth; col++)
			for (int row = 0; row < iHeight; row++)
			{
				double x0 = ratio * (col - iWidth/2);
				double y0 = ratio * (row - iHeight/2);

				double x1 = ratio * (col - iWidth/3);
				double y1 = ratio * (row - iHeight/3);


				double r0 = sqrt(x0*x0 + y0*y0);
				double fi0 = atan2(x0,y0);

				double r1 = sqrt(x1*x1 + y1*y1);
				double fi1 = atan2(x1, y1);


				//double res = cos(x * kx)*sin(y * ky + t);
				//double res = cos(r * kx)*sin(fi * ky);
				double res0 = cos(fi0 * ky - t + r0*kx) * sin(r0 * kx * 0.76 + CV_PI/2.0);
				double res1 = cos(fi1 * ky - t + r1*kx) * sin(r1 * kx * 0.76 + CV_PI/2.0);
				double res = res0 * res1;

				//double res = cos(r * kx + fi * ky - t);
				//double res = cos(r * kx + fi * ky*t);

				image_dst.data[ row * image_dst.step1() + col * image_dst.channels()] = (int)(res * 128) + 128;
			}
		
		
		imshow("image dst", image_dst);

		int delay = 1;
		key = waitKey(delay);

		printf("   %f", t);
		if (key == (char)("q") || 
			key == (char)("é"))
			break;
	}

	return 0;

}
//opencv_core243d.lib;opencv_imgproc243d.lib;opencv_highgui243d.lib;opencv_ml243d.lib;opencv_video243d.lib;opencv_features2d243d.lib;opencv_calib3d243d.lib;opencv_objdetect243d.lib;opencv_contrib243d.lib;opencv_legacy243d.lib;opencv_flann243d.lib;opencv_gpu243d.lib;opencv_nonfree243d.lib;%(AdditionalDependencies)

//opencv_core242d.lib;opencv_imgproc242d.lib;opencv_highgui242d.lib;opencv_ml242d.lib;opencv_video242d.lib;opencv_features2d242d.lib;opencv_calib3d242d.lib;opencv_objdetect242d.lib;opencv_contrib242d.lib;opencv_legacy242d.lib;opencv_flann242d.lib;opencv_gpu242d.lib;opencv_nonfree242d.lib;%(AdditionalDependencies)
