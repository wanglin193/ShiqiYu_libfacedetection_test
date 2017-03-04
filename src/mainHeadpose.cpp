//Created by Wang Lin
//Contact me by wanglin193 at gmail

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "libfaceInterface.h"

#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")
#pragma comment(lib,"opencv_calib3d249.lib")

//#pragma comment(lib,"libfacedetect.lib")
#pragma comment(lib,"libfacedetect-x64.lib")

using namespace cv;
using namespace YuShiqiLibFace;

std::string strName[4] = { "Frontal", "FrontalSurveillance", "Multiview", "MultiviewReinforce" };

int main(int argc, char* argv[])
{ 
	cv::Mat MatK = (Mat_<float>(3,3)<<500,0,320,0,500,240,0,0,1);
	//define
	Mat image,imcanvas;
	Mat gray;
	VideoCapture capture;
	FaceDetectAlignment FDA( true, 68 );

	//init
	FDA.init(true,FrontalSurveillance);
	FDA.setMinSize( 36 );
	capture.open(0);

	if( capture.isOpened() )
	{
		//loop
		std::cout << "Capture is opened." << std::endl;
		for(;;)
		{
			capture >> image;
			if(image.empty())
				break;
			resize(image,image,Size(),0.5,0.5,CV_INTER_LINEAR);
			MatK = (Mat_<float>(3,3)<<250,0,160,0,250,120,0,0,1); 

			cvtColor(image, gray, CV_BGR2GRAY);
		 
			//4 methods
			//for( int method = 0; method<4;method++ )
			int method = 3;
			{
				FDA.setFDMethod( method ); 
				int nFace = FDA.run(gray); 
				
				imcanvas = image.clone();
				if( nFace>0 ) 
				{
					for(int j=0;j<nFace;j++)
					{
						FDA.vFaceInfo[j].drawResult(imcanvas);
						FDA.vFaceInfo[j].solveHeadPose( MatK );
						FDA.vFaceInfo[j].drawModelAxis( imcanvas,MatK );  	

						//FDA.vFaceInfo[j].ProjectHeadModel( MatK );
						//FDA.vFaceInfo[j].drawProjectHeadModel( imcanvas );
					}
				}
				imshow(strName[method], imcanvas);
			}
 
			if(waitKey(10) == 27)
				break;
		}
	}
	else
	{
		image = imread("keliamoniz2.jpg"); 
		if(image.empty())
		{
			std::cout<<"Need Web-cam or Image as input."<<std::endl;
			return 0;
		}
		cvtColor(image, gray, CV_BGR2GRAY);

		//4 methods
		for( int method = 0; method<4;method++ )
		{
			FDA.setFDMethod( method );
			FDA.setMinSize( 36 );
			int nFace = FDA.run(gray); 

			imcanvas = image.clone();
			if( nFace>0 ) 
			{
				printf("%d faces found.\n",nFace);
				for(int j=0;j<nFace;j++)
				{
					FDA.vFaceInfo[j].drawResult(imcanvas);
					FDA.vFaceInfo[j].solveHeadPose( MatK );
					FDA.vFaceInfo[j].drawModelAxis( imcanvas,MatK );  	

					//FDA.vFaceInfo[j].ProjectHeadModel( MatK );
					//FDA.vFaceInfo[j].drawProjectHeadModel( imcanvas );
				}
			}
			imshow(strName[method], imcanvas);
		}
		waitKey(0);
	} 
	//un-init
	FDA.release();

	return 0;
}