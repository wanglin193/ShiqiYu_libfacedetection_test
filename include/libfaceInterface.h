//Created by Wang Lin
//Contact me by wanglin193 at gmail

#include "facedetect-dll.h"

namespace YuShiqiLibFace
{

struct FaceInfo
{
	cv::Rect rectFace;
	int nNeighbors;
	int nAngle;
	cv::Point2i ptLmk[68];
};
	
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

public:
	std::vector<FaceInfo> vFaceInfo;

	FaceDetectAlignment(bool bDoLandmark_,int nLmk_):bDoLandmark(true),nLmk(68)
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
	
	void setFDMethod( int method ) { mMethod = method; 	}

	int run( cv::Mat gray )
	{
		unsigned char * chImdata = (unsigned char*)(gray.ptr(0)); 
		int nWid = gray.cols;
		int nHei = gray.rows;
		int nStep = (int)gray.step;
		float fScale = 1.2;
		int nMinNeighbor = 2;
		int nMinSize = 48;

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
		for( int i=0; i<nNumFaces; i++ )
		{
			short * p = ((short*)(pResults+1))+142*i; 
 
			FaceInfo face;
			face.rectFace = cvRect(p[0],p[1],p[2],p[3]);
			face.nNeighbors = p[4];
			face.nAngle = p[5];

			if (bDoLandmark)
			{
				for (int j = 0; j < nLmk; j++)
					face.ptLmk[j] =  cv::Point2i((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]);
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
	
	//draw results on canvas
	void draw( cv::Mat image  )
	{
		for( int i=0;i<vFaceInfo.size();i++ )
		{
			rectangle(image, vFaceInfo[i].rectFace, cv::Scalar(0, 255, 0), 2);

			for ( int j = 0; j < nLmk; j++ )
			{
				cv::Point2i p = vFaceInfo[i].ptLmk[j];
				cv::circle(image, p, 1, cv::Scalar(0, 255, 0),2);	

			//	cv::putText(image, std::to_string((long long)j), p, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(255,255,255));
			}
		}
	}

};

}