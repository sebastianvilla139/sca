/*
	HAGTA: Human Activities Ground Truth Analyzer.
	This software takes a path containing frames of a video, then plots reference ground truth and optionally test ground truth.
	Software preprocesses information in the following way:

	1. Get number of frames from ground truth and validate that number of frames in path is the same.
	2. Read each frame in path and according to frame id, search using XPath on XML.
	3. If there is a result from XPath query, paint a bounding box from XML info.
	4. If theres is a ground truth test file, search with XPath and do steps 2 and 3.

	Prerequisites:
	-> This software processes a ground truth for a frame sequence. The frames must be on a directory, enumerated with the name of the frame
	and without requiring any mask for the filename, natural sorting will be used as the order mecanism to process frames. All images must 
	be with the same image format and file extension.

	-> Requires:
		* OpenCV.
		* Boost.
		* XSD Codesynthesis.
		* Xerces.
		* XQilla.
*/

// OpenCV includes
#include <cv.h>
#include <highgui.h>

// Standard includes
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

// Boost includes
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

// Xerces includes
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMDocumentType.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMNodeIterator.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

// XQilla includes
#include <xqilla/xqilla-dom3.hpp>

// Util includes
#include "util.h"

using namespace cv;
using namespace std;
using namespace boost::filesystem;
using namespace xercesc;
//XERCES_CPP_NAMESPACE_USE
			
using std::cerr;

#define MAX_XPATH_EXPRESSION_SIZE 300
#define MAX_FRAME_INDEX_FILENAME_SIZE 10
#define MAX_FRAME_NUMBER_STRING_SIZE 10
#define MAX_ROI_LABEL_STRING_SIZE 100
#define MAX_ROI_FRAME_FILENAME_SIZE 10

const string frame_mask = "image%05d.png";
const string frame_extension = ".png";
char frame_index_filename[MAX_FRAME_INDEX_FILENAME_SIZE] = {'\0'};
char frame_expression[MAX_XPATH_EXPRESSION_SIZE] = {'\0'};
char frame_number_string[MAX_FRAME_NUMBER_STRING_SIZE] = {'\0'};
char roi_label_string[MAX_ROI_LABEL_STRING_SIZE] = {'\0'};
char roi_frame_filename[MAX_ROI_FRAME_FILENAME_SIZE] = {'\0'};

int extract_frame_number_from_mask(string filename)
{	
	string frame_number = filename.substr(5, 5);
	return atoi(frame_number.c_str());
}

void set_labels(cv::Mat& im, cv::Rect r, std::string label_top, std::string label_middle, std::string label_bottom, int roi_counter, Scalar color)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 2;
    int baseline = 0;
	int top_offset = 10;
	int bottom_offset = 10;

	cv::Size text_top = cv::getTextSize(label_middle, fontface, scale, thickness, &baseline);
    cv::Point pt_top(r.x + (r.width-text_top.width)/2, r.y - top_offset);    
    cv::putText(im, label_top, pt_top, fontface, scale, color, thickness, 8);

	cv::Size text_middle = cv::getTextSize(label_middle, fontface, scale, thickness, &baseline);
    cv::Point pt_middle(r.x + (r.width-text_middle.width)/2, r.y + (r.height+text_middle.height)/2);    
    cv::putText(im, label_middle, pt_middle, fontface, scale, color, thickness, 8);

	cv::Size text_bottom = cv::getTextSize(label_middle, fontface, scale, thickness, &baseline);
    cv::Point pt_bottom(r.x + (r.width-text_bottom.width)/2, r.y + (r.height - bottom_offset));    
    cv::putText(im, label_bottom, pt_bottom, fontface, scale, color, thickness, 8);
}

int main(int argc, char *argv[])
{
	if (argc < 8)
	{
		cout << currentDateTime()  << "Usage: humanActivitiesGTA.exe gt_xml_path frames_path generate_index_flag(1=true, 0=false, default=0) draw_frames_flag(1=true, 0=false, default=1) generate_roi_files(1=true, 0=false, default=0) generate_roi_videos(1=true, 0=false, default=0) compute_stip_videos(1=true, 0=false, default=0)\n";
		return 1;
	}
	
	path gt_xml_path (argv[1]);
	path frames_path (argv[2]);

	try {
		
		XMLPlatformUtils::Initialize();		
		XQillaPlatformUtils::initialize();

	} catch (const XMLException& eXerces) {
		cerr << "Error during Xerces-C initialisation.\n"
			<< "Xerces exception message: "
			<< UTF8(eXerces.getMessage()) << endl;
		return 1;
	}

	try
	{
		if (!exists(gt_xml_path))
		{
			cout << currentDateTime()  << "File: " << gt_xml_path << " doesn't exist!" << endl;
			return 1;
		}

		if (!exists(frames_path))
		{
			cout << currentDateTime()  << "File: " << frames_path << " doesn't exist!" << endl;
			return 1;
		}

		if (!is_regular_file(gt_xml_path)) {
			cout << currentDateTime()  << "gt_xml_path is NOT a regular file!" << endl;
			return 1;
		} else {
			//cout << currentDateTime()  << "gt_xml_path " << " size is " << file_size(gt_xml_path) << endl;
		}		

		if (!is_directory(frames_path)) {
			cout << currentDateTime()  << "frames_path is NOT a directory!" << endl;
			return 1;
		} else {

			cout << currentDateTime()  << "Processing " << frames_path << " ..." << endl;

			DOMImplementation* xqillaImplementation = DOMImplementationRegistry::getDOMImplementation(X("XPath2 3.0"));

			DOMLSParser* xmlParser = xqillaImplementation->createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);

			DOMConfiguration* dc_parser = xmlParser->getDomConfig();

			dc_parser->setParameter(XMLUni::fgDOMNamespaces, true);
			dc_parser->setParameter(XMLUni::fgXercesSchema, true);
			dc_parser->setParameter(XMLUni::fgDOMValidate, true);

			DOMDocument* document = xmlParser->parseURI(argv[1]);

			if(document == 0) {
				cerr << "Document not found: " << argv[1] << endl;
				return 1;
			}    

			//const DOMXPathNSResolver* resolver = document->createNSResolver(document->getDocumentElement());
			//XQillaNSResolver* xqillaResolver = (XQillaNSResolver*)resolver;   
			//xqillaResolver->addNamespaceBinding(X("xs"), X("http://www.w3.org/2001/XMLSchema"));		
			//xqillaResolver->addNamespaceBinding(X("fn"), X("http://www.w3.org/2005/xpath-functions"));		

			if (strcmp(argv[3], "1") == 0) {
			
				// TODO Calculate frame indexes...
			}

			if (strcmp(argv[4], "1") == 0 || strcmp(argv[5], "1") == 0) {

				// TODO Calculate rois...				
			}

			if (strcmp(argv[6], "1") == 0 || strcmp(argv[7], "1") == 0) {
				typedef vector<path> vec;				

				string frame_index_dir_path = frames_path.string() + "\\frame_index";

				path frame_index_dir(frame_index_dir_path);
																	
				int first_frame_index_number = 1;

				vec v;

				copy(directory_iterator(frame_index_dir), directory_iterator(), back_inserter(v));

				sort(v.begin(), v.end(), numericCompare);

				for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
				{										
					string first_file = it->filename().string();

					//cout << " first_file: " << first_file << endl;

					unsigned found = first_file.find_last_of(".");

					first_frame_index_number = atoi(first_file.substr(0, found).c_str());

					cout << " first_frame_index_number: " << first_frame_index_number << endl;

					break;
				}

				// Generate roi videos from roi frames
				std::ostringstream roi_dir_path;
				roi_dir_path << frames_path.string() << "\\roi" << "\0";
				//cout << currentDateTime()  << "roi_dir_path: " << roi_dir_path.str() << endl;	

				path roi_dir (roi_dir_path.str());

				try
				{
					if (exists(roi_dir))
					{						
						if (is_directory(roi_dir))
						{
							//cout << currentDateTime()  << roi_dir << " is a directory containing:\n";							

							vec roi_path_vector;

							copy(directory_iterator(roi_dir), directory_iterator(), back_inserter(roi_path_vector));							
							
							for (vec::const_iterator it_roi(roi_path_vector.begin()), it_roi_end(roi_path_vector.end()); it_roi != it_roi_end; ++it_roi)
							{
								//cout << currentDateTime()  << "   " << *it_roi << '\n';								

								vec person_path_vector;

								path person_dir (*it_roi);

								//cout << currentDateTime()  << person_dir << " is a directory containing:\n";

								copy(directory_iterator(person_dir), directory_iterator(), back_inserter(person_path_vector));

								if (strcmp(argv[6], "1") == 0) {
									for (vec::const_iterator it_person(person_path_vector.begin()), it_person_end(person_path_vector.end()); it_person != it_person_end; ++it_person)
									{
										//cout << currentDateTime()  << "   " << *it_person << '\n';									

										path profile_dir (*it_person);

										// Generate videos according to standard structure of directories.

										// Full rois video
										vec full_roi_path_vector;
										
										path full_video_frames_pathname(profile_dir.string() + "\\frames\\full\\");										

										string full_video_dir_pathname = profile_dir.string() + "\\video.avi";										

										//cout << currentDateTime()  << "full_video_dir_pathname: " << full_video_dir_pathname << endl;

										VideoWriter full_video_out(full_video_dir_pathname, CV_FOURCC('M','J','P','G'), 30, Size(512, 384));									

										copy(directory_iterator(full_video_frames_pathname), directory_iterator(), back_inserter(full_roi_path_vector));

										for (vec::const_iterator it_full_frame(full_roi_path_vector.begin()), it_full_frame_end(full_roi_path_vector.end()); it_full_frame != it_full_frame_end; ++it_full_frame)
										{
											//cout << currentDateTime()  << "   " << *it_full_frame << '\n';

											path frame_filename_path (*it_full_frame);										

											Mat a;
											//cout << currentDateTime()  << "Reading " <<  frame_filename_path.string() << endl;
											a = imread(frame_filename_path.string());

											if(!a.data)
											{
												//cout << currentDateTime()  <<  "No image data" << endl;
											} else {				   
												//cout << currentDateTime()  <<  "Image data!" << endl;
												full_video_out.write(a);
											}
										}

										// bg_full rois video
										vec bg_full_roi_path_vector;
										
										path bg_full_video_frames_pathname(profile_dir.string() + "\\frames\\bg\\");

										string bg_full_video_dir_pathname = profile_dir.string() + "\\video_bg.avi";

										//cout << currentDateTime()  << "bg_full_video_dir_pathname: " << bg_full_video_dir_pathname << endl;

										VideoWriter bg_full_video_out(bg_full_video_dir_pathname, CV_FOURCC('M','J','P','G'), 30, Size(512, 384));									

										copy(directory_iterator(bg_full_video_frames_pathname), directory_iterator(), back_inserter(bg_full_roi_path_vector));

										for (vec::const_iterator it_bg_full_frame(bg_full_roi_path_vector.begin()), it_bg_full_frame_end(bg_full_roi_path_vector.end()); it_bg_full_frame != it_bg_full_frame_end; ++it_bg_full_frame)
										{
											//cout << currentDateTime()  << "   " << *it_bg_full_frame << '\n';

											path frame_filename_path (*it_bg_full_frame);										

											Mat a;
											//cout << currentDateTime()  << "Reading " <<  frame_filename_path.string() << endl;
											a = imread(frame_filename_path.string());

											if(!a.data)
											{
												//cout << currentDateTime()  <<  "No image data" << endl;
											} else {				   
												//cout << currentDateTime()  <<  "Image data!" << endl;
												bg_full_video_out.write(a);
											}
										}
										
										cout << currentDateTime()  << "Saved videos for " << person_dir.filename().string() << "\\" << profile_dir.filename().string() << "..." << endl;
									}		
								}								

								if (strcmp(argv[7], "1") == 0) {									
									// TODO Compute STIP descriptors ...
								}
							}
						}
						else 
						{
							cout << currentDateTime()  << roi_dir << " is not a directory\n";
							return 1;
						}
					}
					else {
						cout << currentDateTime()  << roi_dir << " does not exist\n";
					}
				}
				catch (const filesystem_error& ex)
				{
					cout << currentDateTime()  << ex.what() << '\n';
				}
			}	
		}
	}
	catch (const filesystem_error& ex)
	{
		cout << currentDateTime()  << ex.what() << '\n';
	}

	XQillaPlatformUtils::terminate();		
	XMLPlatformUtils::Terminate();	

	return 0;
}