//Created by Wang Lin
//Contact me by wanglin193 at gmail

#include "HeadPose.h"
#include "facedetect-dll.h" 

using namespace cv;

namespace YuShiqiLibFace
{
enum {	Frontal, FrontalSurveillance, Multiview, MultiviewReinforce };

class FaceDetectAlignment
{
//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000

private:
	unsigned char * pBuffer;
	int * pResults; 
	int mMethod;
	bool bDoLandmark;
	int nLmk;

	float fScale;// = 1.2f;
	int nMinSize;// = 48; 

public:
	std::vector<HeadPose> vFaceInfo;

	FaceDetectAlignment( bool bDoLandmark_,int nLmk_ ):bDoLandmark(true),nLmk(68),fScale(1.2f),nMinSize(48)
	{
		bDoLandmark = bDoLandmark_;
		nLmk = nLmk_;
		vFaceInfo.clear(); 
	}
	
	int init( bool bDoLandmark_, int method )
	{
		pResults = 0; 
		//pBuffer is used in the detection functions.
		//If you call functions in multiple threads, please create one buffer for each thread!
		pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
		if(!pBuffer)
		{
			std::cerr<<"Can not alloc buffer.\n";
			return -1;
		}

		bDoLandmark = bDoLandmark_;
		mMethod = method;

		return 0;
	}
	
	void setFDMethod( int method ) { mMethod = method; }
	void setMinSize( int nSize ) { nMinSize = nSize; };

	int run( cv::Mat gray )
	{
		unsigned char * chImdata = (unsigned char*)(gray.ptr(0)); 
		int nWid = gray.cols;
		int nHei = gray.rows;
		int nStep = (int)gray.step;
 
		int nMinNeighbor = 2;
		switch( mMethod )
		{
		case Frontal:
			pResults = facedetect_frontal(pBuffer, chImdata,nWid,nHei,nStep,fScale, nMinNeighbor, nMinSize, 0, bDoLandmark);
			break;

		case FrontalSurveillance:
			pResults = facedetect_frontal_surveillance(pBuffer, chImdata,nWid,nHei,nStep,fScale, nMinNeighbor, nMinSize, 0, bDoLandmark);
			break;

		case Multiview:
			pResults = facedetect_multiview(pBuffer, chImdata,nWid,nHei,nStep,fScale, nMinNeighbor, nMinSize, 0, bDoLandmark);
			break;

		case MultiviewReinforce:
			pResults = facedetect_multiview_reinforce(pBuffer,chImdata,nWid,nHei,nStep,fScale, nMinNeighbor, nMinSize, 0, bDoLandmark);
			break;

		default:
			break;
		}

		int nNumFaces = *pResults;

		//parsing faces info
		vFaceInfo.clear();
		int offset = nLmk * 2 + 6;
		for( int i=0; i<nNumFaces; i++ )
		{ 
			short * p = ((short*)(pResults+1))+offset*i; 
 
			HeadPose face;
			face.numLmk = nLmk;
			face.rectFace = cvRect(p[0],p[1],p[2],p[3]);
			//face.nNeighbors = p[4];
			//face.nAngle = p[5];

			if (bDoLandmark)
			{
				for (int j = 0; j < nLmk; j++)
					face.ptLmk[j] =  cv::Point2f((float)p[6 + 2 * j], (float)p[6 + 2 * j + 1]);
			}
			vFaceInfo.push_back(face);
		}
		return nNumFaces;
	}
	
	void release()
	{
		 //release the buffer
		free(pBuffer);
		vFaceInfo.clear();
	} 

};

}