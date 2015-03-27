#include <iostream>
#include "ActionHOGUtils.h"
#include "ActionHOGLibs.h"
#include "opencv2\nonfree\nonfree.hpp"


using std::cout;
using std::cerr;


ActionHOG::ActionHOG() {
	det = "SURF";
	
	chan = "IMG_MHI_OPT";
	imgflag = mhiflag = optflag = true;
	
	vis = true;

	imgngs = 4; imgnbs = 8;
	mhings = 4; mhinbs = 8;
	optngs = 4; optnbs = 8;

	imgHOGDims = imgngs*imgngs*imgnbs;
	mhiHOGDims = mhings*mhings*mhinbs;
	optHOGDims = optngs*optngs*optnbs;
	
	fp = NULL;

	fr_p = 0; //initialize procesed frames
}



ActionHOG::~ActionHOG() {
	vid.release();
}




int ActionHOG::comp() {

	// number of detected points
	int count = 0;
	// time counter at starting point
	double stimer = (double)getTickCount();
	
		if (src.channels() == 3)
			cvtColor(src, cur, CV_RGB2GRAY);
		else
			src.copyTo(cur);

		if (pre.empty())
			cur.copyTo(pre);

		vector<KeyPoint> srcKeys;
		detKeys(pre, srcKeys);

		// if no point detected in a certain frame
		if (srcKeys.size() == 0) {
			cur.copyTo(pre);
			getMotionHistoryImage(fr_p, pre, cur, mhi8U, mhi32F);

		}
		
		// update MHI
		getMotionHistoryImage(fr_p, pre, cur, mhi8U, mhi32F);

		// filter interest points by MHI and OPT
		dstKeys.clear();
		//filterKeysByMotion(srcKeys, dstKeys, mhi8U, pre, cur, src);

		dstKeys = srcKeys;

		// filter by BGS mask
		filter_Fmask();

	    // update the total number
		count += (int)dstKeysf.size();
		
		// compute HOG for image channel
		if (imgflag)
			getImageHOG(pre, dstKeysf, imgHOG);

		// compute HOG for MHI channel
		if (mhiflag)
			getMotionHistoryImageHOG(mhi8U, dstKeysf, mhiHOG);

		// compute HOG for OPT channel
		if (optflag)
			getOpticalFlowHOG(pre, cur, dstKeysf, optHOG);

		// update preceding frame
		cur.copyTo(pre);

		// increase number of processed frames
		fr_p++;

		//Copy color STIP descriptors to an array according each ROI

		vector<float> STIP;
		vector<vector<float>> STIPs;
		
		STIPs_roi.clear();

		for (int i=0; i<Nbbox;i++){//loop through ROIs
			STIPs.clear();
			for (int k=0; k<kpts_roi.size();k++){//loop through Keypoints
				
				//check if the Keypoint belongs to the current ROI
				if(kpts_roi[k]==i+1){
					if (imgflag) {
						const float *pimg = imgHOG.ptr<float>(k);
						STIP.clear();
						for (int j = 0; j < imgHOGDims; ++j)
							STIP.push_back(pimg[j]);	//store values of a single STIP
					}
					STIPs.push_back(STIP);	//store all the STIPs of a single ROI
				}
			}
			STIPs_roi.push_back(STIPs); //store STIPs of each ROI
		}

	return 1;
}


void ActionHOG::filter_Fmask(){	
	Mat imgDstKeys = src.clone();

	dstKeysf.clear();
	kpts_roi.clear();

	for(int i=0; i<dstKeys.size(); i++){
		int v = (int)Rmask.at<uchar>(int(dstKeys[i].pt.y),int(dstKeys[i].pt.x));

		//circle(imgDstKeys, dstKeys[i].pt, (int)dstKeys[i].size/2, Scalar(0, 0, 255), 2);
		if(v!=0){
			dstKeysf.push_back(dstKeys[i]);
			kpts_roi.push_back(v);
		}
	}

	// visualize keys
	if (vis) {
		
		int nDstKeys = dstKeysf.size();

		for (int i = 0; i < nDstKeys; ++i) {
			// draw circle
			circle(imgDstKeys, dstKeysf[i].pt, (int)dstKeysf[i].size/2, Scalar(0, 255, 0), 2);
		}

		imshow("keys", imgDstKeys);
	}
}


int ActionHOG::detKeys(const Mat &img, vector<KeyPoint> &keys) {
	// SURF detector with no orientation normalization
	if (det == "SURF") {
		double hessThresh = 400.0;  // por defecto 400;
		int nOctaves = 3;  // por defecto 3
		int nLayers = 2;
		bool extended = true;
		bool upright = true;  // por defecto true
		SURF detector(hessThresh, nOctaves, nLayers, extended, upright);
		Mat mask = Mat::ones(img.rows, img.cols, CV_8UC1);
		detector(img, mask, keys);
		
		
	} else {
		cerr << det << " is not an available detector!\n";
		system("pause");
		exit(0);
	}

	return 1;	
}


int ActionHOG::getMotionHistoryImage(int idx, const Mat &pre, const Mat &cur, Mat &mhi8U, Mat &mhi32F) {
	if (idx == 0) 
		mhi32F = Mat::zeros(pre.rows, pre.cols, CV_32FC1);

	Mat diff = abs(cur - pre);

	threshold(diff, diff, MHI_DIFF_THRESH, 1, THRESH_BINARY); 

	double timestamp = (double)idx;
	updateMotionHistory(diff, mhi32F, timestamp, MHI_DURATION);
	

	if (timestamp < MHI_DURATION)
		mhi32F.convertTo(mhi8U, CV_8UC1, 255.0/MHI_DURATION, (timestamp - MHI_DURATION)*255.0/MHI_DURATION);
	else
		mhi32F.convertTo(mhi8U, CV_8UC1, 255.0/MHI_DURATION, (MHI_DURATION - timestamp)*255.0/MHI_DURATION);

	Mat mei(mhi8U.rows, mhi8U.cols, CV_8UC1);
	threshold(mhi8U, mei, MHI_MORPH_THRESH, 255, THRESH_BINARY);
	erode(mei, mei, Mat(), Point(-1, -1), 1);
	dilate(mei, mei, Mat(), Point(-1, -1), 6);

	Mat temp;
	mhi8U.copyTo(temp, mei);
	mhi8U = temp;

	if (vis) {
		imshow("MHI", mhi8U);
		waitKey(fdur);
	}

	return 1;
}


int ActionHOG::filterKeysByMotion(const vector<KeyPoint> &srcKeys, vector<KeyPoint> &dstKeys, const Mat &mhi,
								  const Mat &pre, const Mat &cur, const Mat &src) {
	// filter keys by MHI
	Mat mei(mhi.rows, mhi.cols, CV_8UC1);
	threshold(mhi, mei, MHI_MIN_THRESH, 255, THRESH_BINARY);

	int nSrcKeys = (int)srcKeys.size();
	vector<KeyPoint> tempKeys;

	for (int i = 0; i < nSrcKeys; ++i) {
		int ix = (int)(srcKeys[i].pt.x + 0.5);
		int iy = (int)(srcKeys[i].pt.y + 0.5);
		
		if (mei.at<uchar>(iy, ix) == 0)
			//continue;
		
		tempKeys.push_back(srcKeys[i]);
	}

	// filter keys by OPT
	int nTempKeys = tempKeys.size();
	vector<Point2f> preKeys(nTempKeys);

	for (int i = 0; i < nTempKeys; ++i)
		preKeys[i] = tempKeys[i].pt;

	// compute optical flows of those keys preserved after MHI filtering
	vector<float> err;
	vector<uchar> status;
	vector<Point2f> curKeys;

	// only for visualization
	vector<Point2f> optPre;
	vector<Point2f> optCur;


	if (nTempKeys > 0)
		calcOpticalFlowPyrLK(pre, cur, preKeys, curKeys, status, err);
	
	for (int i = 0; i < nTempKeys; ++i) {
		if (status[i] == 0 || err[i] > 50)
			continue;

		Point2f p1 = Point2f(preKeys[i].x, preKeys[i].y);
		Point2f p2 = Point2f(curKeys[i].x, curKeys[i].y);
		double dist = std::sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));

		if (dist < OPT_MIN_THRESH || dist > OPT_MAX_THRESH)
			continue;

		// keep optical flow vector for visualization
		optPre.push_back(p1);
		optCur.push_back(p2);

		// preserve keys with sufficient motions
		dstKeys.push_back(tempKeys[i]);
	}

	return 1;
}


int ActionHOG::getImageHOG(const Mat &img, const vector<KeyPoint> &keys, Mat &hog) {
	Mat gradx, grady;
	getGradients(img, gradx, grady);

	int nKeys = keys.size();
	hog.create(nKeys, imgngs*imgngs*imgnbs, CV_32FC1);

	for (int i = 0; i < nKeys; ++i) {
		Mat desc;
		getHOGatKey(keys[i], gradx, grady, imgngs, imgnbs, desc);
		desc.copyTo(hog.row(i));
	}

	return 1;
}


int ActionHOG::getMotionHistoryImageHOG(const Mat &mhi, const vector<KeyPoint> &keys, Mat &hog) {
	Mat gradx, grady;
	getGradients(mhi, gradx, grady);

	int nKeys = keys.size();
	hog.create(nKeys, mhings*mhings*mhinbs, CV_32FC1);

	for (int i = 0; i < nKeys; ++i) {
		Mat desc;
		getHOGatKey(keys[i], gradx, grady, mhings, mhinbs, desc);
		desc.copyTo(hog.row(i));
	}

	return 1;
}

#ifdef OPT_PATCH
int ActionHOG::getOpticalFlowHOG(const Mat &pre, const Mat &cur, const vector<KeyPoint> &keys, Mat &hog) {
	int nRows = pre.rows;
	int nCols = pre.cols;

	int nKeys = keys.size();
	hog.create(nKeys, optngs*optngs*optnbs, CV_32FC1);

	for (int i = 0; i < nKeys; ++i) {
		int x = (int)(keys[i].pt.x + 0.5f);
		int y = (int)(keys[i].pt.y + 0.5f);
		int rad = (int)(keys[i].size / 2.0f + 0.5f);

		int x1 = x - rad; x1 = (x1 < 0) ? 0 : x1;
		int x2 = x + rad; x2 = (x2 > nCols) ? nCols : x2;
		int y1 = y - rad; y1 = (y1 < 0) ? 0 : y1;
		int y2 = y + rad; y2 = (y2 > nRows) ? nRows : y2;

		Mat preROI(pre, Range(y1, y2), Range(x1, x2));
		Mat curROI(cur, Range(y1, y2), Range(x1, x2));

		double scale  = 0.5;
		int levels = 1;
		int winsize = 5;
		int iterations = 10;
		int polyn = 5;
		double polysigma = 1.1;

		Mat opt;
		calcOpticalFlowFarneback(preROI, curROI, opt, scale, levels, winsize, iterations, polyn, polysigma, OPTFLOW_FARNEBACK_GAUSSIAN);

		Mat velx(opt.rows, opt.cols, CV_32FC1);
		Mat vely(opt.rows, opt.cols, CV_32FC1);

		for (int ir = 0; ir < opt.rows; ++ir) {
			float *ptro = opt.ptr<float>(ir);
			float *ptrx = velx.ptr<float>(ir);
			float *ptry = vely.ptr<float>(ir);

			for (int ic = 0; ic < opt.cols; ++ic) {
				ptrx[ic] = ptro[2*ic+0];
				ptry[ic] = ptro[2*ic+1];
			}
		}

		Mat desc;
		getHOGatPatch(velx, vely, optngs, optnbs, desc);
		desc.copyTo(hog.row(i));
	}	

	return 1;
}
#else
int ActionHOG::getOpticalFlowHOG(const Mat &pre, const Mat &cur, const vector<KeyPoint> &keys, Mat &hog) {
	int nRows = pre.rows;
	int nCols = pre.cols;
	int nKeys = keys.size();
	hog.create(nKeys, optngs*optngs*optnbs, CV_32FC1);

	double scale  = 0.5;
	int levels = 1;
	int winsize = 3;
	int iterations = 10;
	int polyn = 5;
	double polysigma = 1.1;

	Mat opt;
	calcOpticalFlowFarneback(pre, cur, opt, scale, levels, winsize, iterations,
							 polyn, polysigma, OPTFLOW_FARNEBACK_GAUSSIAN);

	Mat velx = Mat::zeros(opt.rows, opt.cols, CV_32FC1);
	Mat vely = Mat::zeros(opt.rows, opt.cols, CV_32FC1);

	for (int ir = 0; ir < opt.rows; ++ir) {
		float *ptro = opt.ptr<float>(ir);
		float *ptrx = velx.ptr<float>(ir);
		float *ptry = vely.ptr<float>(ir);

		for (int ic = 0; ic < opt.cols; ++ic) {
			float vx = ptro[2*ic+0];
			float vy = ptro[2*ic+1];
			float dist = vx*vx + vy*vy;

			if (dist > 1) {
				ptrx[ic] = vx;
				ptry[ic] = vy;
			}
		}
	}

	for (int i = 0; i < nKeys; ++i) {
		Mat desc;
		getHOGatKey(keys[i], velx, vely, optngs, optnbs, desc);
		desc.copyTo(hog.row(i));
	}

	return 1;
}
#endif


int ActionHOG::writeKeyDesc(const vector<KeyPoint> &keys) {
	int nKeys = keys.size();

	char save_path[250], fr[5], bbox[5];

	itoa(fr_idx,fr,10);

	double d1, d2, d3, d4;

	for (int i=0; i<Nbbox;i++){ //loop through detected ROI

		itoa(i,bbox,10);

		//make path to save LBSP images
		strcpy(save_path,"C:\\Users\\Santiago\\Desktop\\People_Dbase\\STIP\\");
		strcat(save_path,vidname);
		strcat(save_path,"_");
		strcat(save_path,fr);
		strcat(save_path,"_");
		strcat(save_path,bbox);
		strcat(save_path,".txt");

		//open file
		fp = fopen(save_path,"w+");

		for (int k=0; k<kpts_roi.size();k++){	//loop through STIP descriptors
			
			//check if the descriptor belongs to the current ROI
			if(kpts_roi[k]==i+1){

				//write stip coordinates and ratio
				fprintf(fp, "%f %f %f ", keys[k].pt.x, keys[k].pt.y, keys[k].size);

				//write distances to "interest points"
				d1 = sqrt((keys[k].pt.x + 219)*(keys[k].pt.x + 219) +(keys[k].pt.y + 159)*(keys[k].pt.y + 159) );
				d2 = sqrt((keys[k].pt.x + 20)*(keys[k].pt.x + 20) + (keys[k].pt.y + 205)*(keys[k].pt.y + 205) );
				d3 = sqrt((keys[k].pt.x + 75)*(keys[k].pt.x + 75) +(keys[k].pt.y + 182)*(keys[k].pt.y + 182) );
				d4 = sqrt((keys[k].pt.x + 350)*(keys[k].pt.x + 350) +(keys[k].pt.y + 127)*(keys[k].pt.y + 127) );
				fprintf(fp,"%f %f %f %f ", d1, d2, d3, d4);

				if (imgflag) {
					const float *pimg = imgHOG.ptr<float>(k);
					for (int j = 0; j < imgHOGDims; ++j)
						fprintf(fp, "%f ", pimg[j]);
				}

				if (mhiflag) {
					const float *pmhi = mhiHOG.ptr<float>(k);
					for (int j = 0; j < mhiHOGDims; ++j)
						fprintf(fp, "%f ", pmhi[j]);
				}

				if (optflag) {
					const float *popt = optHOG.ptr<float>(k);
					for (int j = 0; j < optHOGDims; ++j)
						fprintf(fp, "%f ", popt[j]);
				}
				fprintf(fp,"\n");
			}
		}
		fclose(fp);
	}
	return 1;
}