/**
 *  * Display video from webcam and detect faces
 *   */
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "cv.h"
//#include "core.h"
//#include "highgui.h"
 
using namespace std;
using namespace cv;
String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String window_name = "webcam_stream";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
static const int font_face = FONT_HERSHEY_DUPLEX;
static const double font_scale = 0.8;
static const int font_thickness = 1;
static double dt = 1.0/15.0;


void print(cv::Mat &x){

  std::cout << x << std::endl;

}
//general kalman equations:
// x_est(k+1) = A*x_est(k) + B*u(k);
// ..
// z(k+1) = new_measurement? z_measured: z_est(k);
// v(k+1) = z(k+1) - H*x_est(k+1);
// kalman Gain: K = ..
// ..
// x_est(k+1) = x_est(k+1) + K*v(k+1);
void kalman(cv::Mat &img, double x, double y, double width, double height){
  using namespace cv;

  static Mat A = Mat::eye(8, 8, CV_32FC1); //state transition matrix
  A.at<float>(0,4) = dt;
  A.at<float>(1,5) = dt;
  //no velocity change in width and height
  //A.at<float>(2,6) = dt;
  //A.at<float>(3,7) = dt;
  static Mat B = Mat::zeros(8, 8, CV_32FC1); //control transition matrix
  static Mat H = Mat::eye(4, 8, CV_32FC1); //measurement matrix (from state to measurement)
  static int states = A.rows;
  static int measurement_size = H.rows;
  static double noise = 1;

  static double p0 = 1;
  static Mat R = noise*Mat::eye(states, states, CV_32FC1);
  static Mat Q = noise*Mat::eye(measurement_size, measurement_size, CV_32FC1);
  static Mat P = p0*Mat::eye(states, states, CV_32FC1); //state covariance matrix

  static Mat x0 = (Mat_<float>(8,1) << 500,500,200,200, 0, 0, 0, 0);
  static Mat x_est = x0;

  
  x_est = A*x_est;
  P = A*P*A.t() + R;

  //The real measurement

  Mat z = (Mat_<float>(4,1) << x,y,width,height);
  static Mat z_est = (Mat_<float>(4,1) << 0,0,0,0);
  if(x==0 && y==0 && width==0 && height ==0){
    z = z_est;
  }

  z_est = H*x_est;
  Mat v = z - z_est;

  Mat K = P*H.t()*(( H*P*H.t() + Q).inv());
  P = (Mat::eye(K.rows, K.rows, CV_32FC1) - K*H)*P;
  x_est = x_est + K*v;

  //print(x_est);
  //show estimated parameters in image frame
  float c_x = x_est.at<float>(0);
  float c_y = x_est.at<float>(1);
  float c_w = x_est.at<float>(2);
  float c_h = x_est.at<float>(3);
  cv::rectangle(  img,
                  cvPoint( c_x, c_y ),
                  cvPoint( c_x+c_w, c_y+c_h ),
                  CV_RGB( 0, 0, 255 ), 2, 8, 0 );
  cv::putText(img,  "Kalman filtered", cvPoint(c_x, c_y-10), font_face,
      font_scale, CV_RGB(0,0,255), font_thickness, 8);
}


void detectFaces( cv::Mat & );
 
int main( int argc, char** argv )
{
  CvCapture *capture;
  Mat frame;
  int key;

  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  capture = cvCaptureFromCAM( 0 );
  //double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
  //std::cout << "webcam running at " << fps << " frames per second" << std::endl;

  while( key != 'q' ) {
    frame = cvQueryFrame( capture );
    if( frame.empty() ) break;
    cv::putText(frame,  "Press q to exit", cvPoint(0, frame.rows-10), font_face,
      font_scale, Scalar::all(255), font_thickness, 8);
    //preprocess frame
    
    //cv::morphologyEx(frame, frame, CV_MOP_OPEN, cv::Mat(), cv::Point(-1,-1),5);
    detectFaces( frame );
    cv::imshow(window_name, frame);
                       
    key = cvWaitKey(1);
  }

  return 0;
}
 
void detectFaces( Mat &frame )
{

   std::vector<Rect> faces;
   Mat frame_gray;

   cvtColor( frame, frame_gray, CV_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );
   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  if(faces.size()!=1){
    kalman(frame, 0,0,0,0); 
  }else{
   for( int i = 0; i < faces.size(); i++ )
    {
      if(i==0) kalman(frame,faces[i].x, 
                            faces[i].y,
                            faces[i].width,
                            faces[i].height);
      Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
      cv::rectangle(  frame,
                  cvPoint( faces[i].x, faces[i].y ),
                  cvPoint( faces[i].x+faces[i].width, faces[i].y+faces[i].height),
                  CV_RGB( 255, 0, 0 ), 2, 8, 0 );
      cv::putText(frame,  "Haar feature output", 
                  cvPoint(faces[i].x, faces[i].y+faces[i].height+20), 
                  font_face, font_scale, CV_RGB(255,0,0), font_thickness, 8);
    } 
  }
}












