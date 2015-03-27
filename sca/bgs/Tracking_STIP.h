#ifndef Tracking_STIP_H
#define Tracking_STIP_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//Color library
const static double Color[20][3] = {{200/255.,0,0},{0,0,200./255.},{0,200./255.,0},{200/255.,200/255.,0},{0,200/255.,200/255.},{200/255.,0,200/255.},
{0,80/255.,255/255.},{220/255.,150/255.,200/255.},{0,250/255.,100/255.},{100/255.,0,250/255.},{100/255.,250/255.,0},
{240/255.,240/255.,100/255.},{180/255.,250/255.,20/255.},{160/255.,190/255.,20/255.},{25/255.,255/255.,155/255.},{180/255.,130/255.,70/255.},
{32/255.,165/255.,220/255.},{120/255.,150/255.,255/255.},{125/255.,25/255.,25/255.},{105/255.,180/255.,180/255.}};


class Tracking_STIP{
public:

	//-------------Main Functions-------------
	Tracking_STIP(cv::Mat iniFr);

	~Tracking_STIP();

	void getRegions();

	void matchRegions();

	//-----------Auxiliar Functions-----------
	void Bwlabel(cv::Mat src, cv::Mat &des);

	void Erosion(cv::Mat src, cv::Mat des);

	void Imagesc(cv::Mat src, int siz, char* Plotname);

	void Save_bbox();

	cv::Mat Mul3(cv::Mat A3, cv::Mat B1);

	//---------------Variables----------------
	cv::Mat Im_RGB;		//Current RGB image
	cv::Mat Im_LBSP;	//Current LBSP image
	cv::Mat Fmask;		//Current foreground mask
	cv::Mat Rmask;		//Current foreground mask with isolated regions
	int Nroi;			//Number of isolated regions in current foreground mask
	int fr_idx;			//Frame index
	char vidname[50];	//Video name

	//---------------Parameters---------------
	double pminArea;
	
private:

	//---------------Variables----------------
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Point pt;
	int minArea;
	std::vector<std::vector<int>> Xcroi;	//X coordinates of each isolated region 
	std::vector<std::vector<int>> Ycroi;	//Y coordinates of each isolated region 
	std::vector<std::vector<int>> Bbox;		//Current detected bounding boxes
	std::vector<std::vector<int>> Bbox_states;		//States of detected bounding boxes
};
#endif