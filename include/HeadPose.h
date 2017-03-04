//Created by Wang Lin
//Contact me by wanglin193 at gmail
#include "headdata3d.h"

using namespace cv;
//SE3 convertor
cv::Mat RT2POSE( cv::Mat& rvec, cv::Mat& tvec )
{ 
	cv::Mat P = Mat::eye( 3,4,CV_32FC1 );  
	cv::Mat rotation;
	cv::Rodrigues(rvec, rotation);  

	for( int i=0; i<3; i++ ) 
	{
		P.at<float>(i,3) =  tvec.at<double>(i);  
		for( int j=0; j<3; j++ ) 
		{
			P.at<float>(i,j) =  rotation.at<double>(i,j); 
		}
	}
	return P.clone();
}
//p3d->p2d
cv::Mat Project3d(cv::Mat& p3d, cv::Mat& MatK,cv::Mat& Pose) //Pose 3*4 or 4*4
{ 
	assert(p3d.rows == 4); 

	//lambda * p2d = K * [R|t] * p3d
	//cv::Mat_<float> Proj = MatK* Pose.rowRange(0,3);
	//cv::Mat_<float> p2d =  Proj*pModel;   

	cv::Mat p2d = MatK * Pose.rowRange(0,3) * p3d; 

	p2d.row(0) = p2d.row(0).mul(1/p2d.row(2));
	p2d.row(1) = p2d.row(1).mul(1/p2d.row(2)); 

	return p2d.clone();
}

#define numMax 100 
class HeadPose
{	
public:
	int numLmk;
	
	//face detect result
	cv::Rect rectFace;
	
	//face aligment result
	cv::Point2f ptLmk[numMax];
	
	//2d projection of 3d face model 
	cv::Point2f ptModelProj[numMax]; 
	
	//6 dof pose
	cv::Mat rvec, tvec;

	//pose from 6dof, 3*4 matrix
	cv::Mat Pose;

	HeadPose()
	{ 
		numLmk=0;
		rvec = (Mat_<double>(3,1)<<0,0,0); //solvePnP need double
		tvec = (Mat_<double>(3,1)<<0,0,500);
		Pose = RT2POSE(rvec,tvec);
	};
	//use solvePnp
	void solveHeadPose( cv::Mat& MatK )
	{  
		if( numLmk < 1 )
			return;

		std::vector<Point3f> p3d;
		std::vector<Point2f> p2d;

		for( int j=0;j<NUMID;j++ )
		{
			int i = idPosePoint[j];
			Point3f phead = Point3f(head3d_68[i][0],head3d_68[i][1],head3d_68[i][2]);
			p3d.push_back( phead );
			p2d.push_back(ptLmk[i]);
		}
		//converge from init pose
		rvec = (Mat_<double>(3,1)<<0,0,0);
		tvec = (Mat_<double>(3,1)<<0,0,500);
		cv::solvePnP( p3d, p2d,	MatK, cv::noArray(),rvec, tvec, true, cv::EPNP );  //ITERATIVE, EPNP, P3P 
	   
		Pose = RT2POSE(rvec,tvec);
 
		if ( Pose.at<float>(0,2) > 0) printf("toward right \n");
		if ( Pose.at<float>(0,2) < 0) printf("toward left  \n");  			
	}
	//calculate ptModelProj
	void ProjectHeadModel( cv::Mat& MatK )
	{
		cv::Mat_<float> pModel(4 ,numLmk ); 

		for(int i=0;i<numLmk;i++)
		{
			pModel(0,i)=head3d_68[i][0]; 
			pModel(1,i)=head3d_68[i][1]; 
			pModel(2,i)=head3d_68[i][2]; 
			pModel(3,i)=1; 
		}

		cv::Mat pModel2d = Project3d(pModel,MatK,Pose);
		
		for( int i=0;i<numLmk;i++ )
		{
			ptModelProj[i] =  Point2f( pModel2d.at<float>(0,i), pModel2d.at<float>(1,i) ); 
		} 
		return; 
	}
	//project 3d mode to image
	void drawProjectHeadModel( cv::Mat image )
	{ 
		for ( int j = 0; j < NUMID; j++ )
		{
			int i = idPosePoint[j];
			cv::Point2i p = ptModelProj[i]; 
			cv::circle( image, p, 1, cv::Scalar(255, 0, 0),2 );	
		}	 
	}
	//draw 3d axis of model 
	void drawModelAxis( cv::Mat image, cv::Mat& MatK )
	{
		const int nPt = 4;
		cv::Mat pModel = (Mat_<float>(nPt,4)<< 0,0,0,1,   100,0,0,1,  0,100,0,1,  0,0,-100,1) ;  
		pModel = pModel.t();

		//std::cout<<Pose<<std::endl; 
		cv::Mat p2d = Project3d( pModel,MatK,Pose );  
		Point2f p0,p1;
		
		p0 = Point2f( p2d.at<float>(0,0),p2d.at<float>(1,0) );
		p1 = Point2f( p2d.at<float>(0,1),p2d.at<float>(1,1) ); 
		cv::line( image, p0,p1, cv::Scalar(0,0,255),2);	

		p0 = Point2f( p2d.at<float>(0,0),p2d.at<float>(1,0) );
		p1 = Point2f( p2d.at<float>(0,2),p2d.at<float>(1,2) ); 
		cv::line( image, p0,p1, cv::Scalar(0,255,0),2);	

		p0 = Point2f( p2d.at<float>(0,0),p2d.at<float>(1,0) );
		p1 = Point2f( p2d.at<float>(0,3),p2d.at<float>(1,3) ); 
		cv::line( image, p0,p1, cv::Scalar(255,0,0),2 );	 
	}
	//draw face alignment results on canvas
	void drawResult( cv::Mat image  )
	{
		rectangle( image, rectFace, cv::Scalar(0, 255, 0), 1 );

		for ( int j = 0; j < numLmk; j++ )
		{
			cv::Point2i p =  ptLmk[j];
			cv::circle( image, p, 1, cv::Scalar(0, 255, 0),2 );	
			//cv::putText(image, std::to_string((long long)j), p, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(255,255,255));
		}
	}
};