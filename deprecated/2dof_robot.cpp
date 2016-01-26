
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
static const int font_face = FONT_HERSHEY_DUPLEX;
static const double font_scale = 0.5;
static const int font_thickness = 1;
static double dt = 1.0/15.0;

//system can be described by rotation angle [0,360[ and mid point x,y
void drawSystem(cv::Mat &img, cv::Mat &state){

  //get joint angles
  double theta1 =state.at<double>(0,0);
  double theta2 =state.at<double>(1,0);

  double torque1 =state.at<double>(2,0);
  double torque2 =state.at<double>(3,0);

  double theta1_dev = torque1;
  double theta2_dev = torque2;

  //robot specific variables
  const double m1 = 1;
  const double m2 = 1;
  const double l1 = W/20;
  const double l2 = W/20;
  const double g = -9.81;

  cv::Mat M = cv::Mat::eye(2,2,CV_64F);
  cv::Mat B = cv::Mat::eye(2,2,CV_64F);
  cv::Mat G = cv::Mat::eye(2,1,CV_64F);

  M.at<double>(0,0) = l2*l2*m2*m2 + 2*l1*l2*cos(theta2)+l1*l1*(m1+m2);
  M.at<double>(1,0) = l2*l2 + l1*l2*m2*cos(theta2);
  M.at<double>(1,1) = l2*l2*m2;
  M.at<double>(0,1) = l2*l2 + l1*l2*m2*cos(theta2);

  B.at<double>(0,0) = -2*m2*l1*l2*theta2_dev*sin(theta2);
  B.at<double>(0,1) = -m2*l1*l2*theta2_dev*sin(theta2);
  B.at<double>(1,0) = m2*l1*l2*theta1_dev*sin(theta2);
  B.at<double>(1,1) = 0;


  G.at<double>(0,0) = g*(m2*l2*cos(theta1-theta2) + (m1+m2)*l1*cos(theta1));
  G.at<double>(1,0) = g*(m2*l2*cos(theta1-theta2));


  cv::line(img, cvPoint(cx,cy), cvPoint(cx_front, cy_front), CV_RGB(255,0,0), 2);
  cv::circle(img, cvPoint(cx,cy), length_car/4, CV_RGB(255,0,0), -1);

  char pos[50];
  sprintf(pos, "X: %d, Y: %d",(int) cx, (int)cy);
  cv::putText(img, pos, cvPoint(0,20), font_face,
      font_scale, CV_RGB(0,0,255), font_thickness, 8);
  sprintf(pos, "THETA: %.2f, VELX: %.2f", theta*180.0/M_PI, state.at<double>(3,0));
  cv::putText(img, pos, cvPoint(0,40), font_face,
      font_scale, CV_RGB(0,0,255), font_thickness, 8);
}

void manualControl(cv::Mat &control, cv::Mat &state, int key){

  double theta = state.at<double>(0,2);
   control.at<double>(0,0)=0.0;
   control.at<double>(1,0)=0.0;
  double offset = 2.0;
  switch(key){
   case 'w': control.at<double>(0,0)=offset;
             break;
   case 's': control.at<double>(0,0)=-offset;
             break;
   case 'a': control.at<double>(1,0)=M_PI/32;
             break;
   case 'd': control.at<double>(1,0)=-M_PI/32;
             break;
  }

}

void systemCheckConstraints(cv::Mat &state){
  static double max_velocity = 20.0;
  //check for constraints
  if(state.at<double>(0,0)>W) state.at<double>(0,0)-=W;
  if(state.at<double>(0,0)<0) state.at<double>(0,0)+=W;
  if(state.at<double>(1,0)>H) state.at<double>(1,0)-=H;
  if(state.at<double>(1,0)<0) state.at<double>(1,0)+=H;

  if(state.at<double>(3,0)>max_velocity) state.at<double>(3,0)=max_velocity;
  if(state.at<double>(3,0)<0) state.at<double>(3,0)=0;

  if(state.at<double>(2,0)>2*M_PI) state.at<double>(2,0)-=2*M_PI;
  if(state.at<double>(2,0)<0) state.at<double>(2,0)+=2*M_PI;
}

void updateSystem(cv::Mat &state, cv::Mat &control){
  double theta_k = state.at<double>(2,0);

  static cv::Mat A = Mat::eye(4,4,CV_64F);
  A.at<double>(0,3)= cos(theta_k)*dt;
  A.at<double>(1,3)= -sin(theta_k)*dt;
  static cv::Mat B = Mat::zeros(4,2,CV_64F);
  B.at<double>(0,0) = cos(theta_k);
  B.at<double>(1,0) = -sin(theta_k);
  B.at<double>(2,1) = 1;
  B.at<double>(3,0) = 1;


  state = A*state + B*control;

  systemCheckConstraints(state);
}

void updateSystemPID(cv::Mat &state, cv::Mat &desiredState, cv::Mat &control){
  double theta_k = state.at<double>(2,0);

  static cv::Mat A = Mat::eye(4,4,CV_64F);
  A.at<double>(0,3)= cos(theta_k)*dt;
  A.at<double>(1,3)= -sin(theta_k)*dt;
  static cv::Mat B = Mat::zeros(4,2,CV_64F);
  B.at<double>(0,0) = cos(theta_k);
  B.at<double>(1,0) = -sin(theta_k);
  B.at<double>(2,1) = 1;
  B.at<double>(3,0) = 1;


  static double Kp = 0.1;
  
  //calculate non-linear control error
  double x = state.at<double>(0,0);
  double y = state.at<double>(1,0);
  double theta = state.at<double>(2,0);
  double v = state.at<double>(3,0);
  double xd = desiredState.at<double>(0,0);
  double yd = desiredState.at<double>(1,0);
  double thetad = desiredState.at<double>(2,0);
  double vd = desiredState.at<double>(3,0);
  double dist = sqrtf((x-xd)*(x-xd)+(y-yd)*(y-yd));

  double cur_dir = state.at<double>(2,0);

  printf("desired: %.2f %.2f\n", xd, yd);
  double theta_to_target = atan2(yd-y, xd-x);
  if(theta_to_target>2*M_PI) theta_to_target -= 2*M_PI;
  if(theta_to_target<0) theta_to_target += 2*M_PI;
  printf("theta to target is : %.2f\n", theta_to_target*180.0/M_PI);
  printf("theta is : %.2f\n", theta);
  printf("distance to target : %.2f\n", dist);

  if(dist < 50){
  control.at<double>(0,0) = -Kp*(v-vd);
  control.at<double>(1,0) = 0;
  }else{
  control.at<double>(0,0) = Kp*(dist);
  control.at<double>(1,0) = -Kp*(theta-theta_to_target);
  }

  state = A*state + B*control;

  systemCheckConstraints(state);
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
  Mat control = Mat::zeros(2, 1, CV_64F);
  Mat state = Mat::ones(4, 1, CV_64F);
  Mat desiredState = Mat::zeros(4,1,CV_64F);
  int key;

  state.at<double>(0,0)=0/2;
  state.at<double>(1,0)=H;
  state.at<double>(2,0)=M_PI/2;
  state.at<double>(3,0)=0;

    desiredState.at<double>(0,0)=W/2;
    desiredState.at<double>(1,0)=H/2;
    desiredState.at<double>(2,0)=0;
    desiredState.at<double>(3,0)=0;
  while( key!= 'q'){
    clear(img);

    //state x_k -> x_k+1
    key = cvWaitKey(100);

    updateSystemPID(state, desiredState, control);

    manualControl(control, state, key);

    updateSystem(state, control);
    //draw state x_k+1
    drawSystem(img, state);

    print(img);

  }

  cv::waitKey();
  return 0;

}
