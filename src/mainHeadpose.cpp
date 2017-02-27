//Created by Wang Lin
//Contact me by wanglin193 at gmail

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "libfaceInterface.h"

#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")

//#pragma comment(lib,"libfacedetect.lib")
#pragma comment(lib,"libfacedetect-x64.lib")

using namespace cv;
using namespace YuShiqiLibFace;

std::string strName[4] = { "Frontal", "FrontalSurveillance", "Multiview", "MultiviewReinforce" };

int main(int argc, char* argv[])
{ 
	//define
	Mat image,imcanvas;
	Mat gray;
	VideoCapture capture;
	FaceDetectAlignment FDA(true,68);

	//init
	FDA.init(true,FrontalSurveillance);
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
			//resize(image,image,Size(),0.5,0.5,CV_INTER_LINEAR);
			cvtColor(image, gray, CV_BGR2GRAY);

			//4 methods
			for( int i=0; i<4;i++ )
			{
				FDA.setFDMethod(i);
				int nFace = FDA.run(gray);
				//printf("%d  face(s)  detected.\r",nFace);
				
				imcanvas = image.clone();
				if( nFace>0 ) 
					FDA.draw(imcanvas);
				imshow(strName[i], imcanvas);
			}
 
			if(waitKey(10) == 27)
				break;
		}
	}
	else
	{
		image = imread("face.jpg"); 
		if(image.empty())
		{
			std::cout<<"Need Web-cam or Image as input."<<std::endl;
			return 0;
		}
		//resize(image,image,Size(),0.5,0.5,CV_INTER_LINEAR); 

		cvtColor(image, gray, CV_BGR2GRAY);

		//4 methods
		for( int i=0; i<4;i++ )
		{
			FDA.setFDMethod(i);
			int nFace = FDA.run(gray);
			//printf("%d  face(s)  detected.\n",nFace);

			imcanvas = image.clone();
			if( nFace>0 ) 
				FDA.draw(imcanvas);
			imshow(strName[i], imcanvas);
		}
		waitKey(0);
	}

	//un-init
	FDA.release();

	return 0;
}