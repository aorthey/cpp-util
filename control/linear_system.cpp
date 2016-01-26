#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
using namespace cv;
String window_name = "system";
const double W = 600;
const double H = 400;

//system can be described by rotation angle [0,360[ and mid point x,y
void drawSystem(cv::Mat &img, cv::Mat &state){

  double length = W/8;
  double cx = W/2+state.at<double>(0,0);
  double cy = H/2+state.at<double>(1,0);
  printf("cx: %f, cy: %f\n", state.at<double>(0,0), state.at<double>(1,0));
  double cyl_width = length*6;
  cv::rectangle(img, cvPoint(cx,cy), cvPoint(cx+cyl_width/2,cy+length/4),CV_RGB(255,0,0),-1); 
  cv::rectangle(img, cvPoint(cx,cy), cvPoint(cx+cyl_width/2,cy-length/4),CV_RGB(255,0,0),-1); 
  cv::rectangle(img, cvPoint(cx,cy), cvPoint(cx-cyl_width/2,cy+length/4),CV_RGB(255,0,0),-1); 
  cv::rectangle(img, cvPoint(cx,cy), cvPoint(cx-cyl_width/2,cy-length/4),CV_RGB(255,0,0),-1); 
  cv::circle(img, cvPoint(cx,cy), length, CV_RGB(255,0,0), -1);
}

void clear(cv::Mat &img){
  img.setTo(255.0);
}
void print(cv::Mat &img){
  cv::imshow(window_name, img);
}

int main(void)
{

  Mat img = Mat::ones(H, W, CV_8UC3);
  Mat state = Mat::ones(2, 1, CV_64F);
  int key;

  state.at<double>(0,0)=0.0;
  state.at<double>(1,0)=0.0;
  while( key!= 'q'){
    clear(img);

    drawSystem(img, state);

    print(img);

    key = cvWaitKey(1);
    double offset = 2.0;

    switch(key){
     case 'h': state.at<double>(0,0)-=offset;break;
     case 'l': state.at<double>(0,0)+=offset;break;
     case 'j': state.at<double>(1,0)+=offset;break;
     case 'k': state.at<double>(1,0)-=offset;break;
     case 'q': return 0;break;
     default: break;
    }
    printf("state 0: %f, 1: %f \n", state.at<double>(0,0), state.at<double>(1,0));

  }

  cv::waitKey();
  return 0;

}
