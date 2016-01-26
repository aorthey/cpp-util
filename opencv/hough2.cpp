#include <cv.h>
#include <highgui.h>
#include <math.h>

int main(int argc, char** argv)
{
    IplImage* img;

    //CvCapture *cap = cvCaptureFromCAM(2);

    //if(!cvGrabFrame(cap)){
       //printf("Could not grab frame\n");
       //exit(0);
    //}
    //img = cvRetrieveFrame(cap);

    img=cvLoadImage(argv[1], 1);
    if( img != 0 ) 
    {
        IplImage* gray = cvCreateImage( cvGetSize(img), 8, 1 );
        IplImage* dilate = cvCloneImage(img);
        IplImage* median = cvCloneImage(img);
        IplImage* erode = cvCloneImage(img);





        CvMemStorage* storage = cvCreateMemStorage(0);
        cvCvtColor( img, gray, CV_BGR2GRAY );
	cvNamedWindow("gray", 0);
        cvSmooth( gray, gray, CV_GAUSSIAN, 9, 9 ); // smooth it, otherwise a lot of false circles may be detected
	cvShowImage("gray", gray);

        CvSeq* circles = cvHoughCircles( gray, storage, CV_HOUGH_GRADIENT, 2, gray->height/4, 200, 100 );
        int i;
        for( i = 0; i < circles->total; i++ )
        {
             float* p = (float*)cvGetSeqElem( circles, i );
             cvCircle( img, cvPoint(cvRound(p[0]),cvRound(p[1])), 3, CV_RGB(0,255,0), -1, 8, 0 );
             cvCircle( img, cvPoint(cvRound(p[0]),cvRound(p[1])), cvRound(p[2]), CV_RGB(255,0,0), 3, 8, 0 );
        }
        cvNamedWindow( "dilate", 1 );
        cvNamedWindow( "erode", 2);
        cvNamedWindow( "median", 3);

        //3x3 dilate filter (maximum value of rectangle) (8 times iteration) 
        cvDilate(img, dilate, NULL, 8);
	cvMoveWindow("dilate",0, 0);
    	cvShowImage("dilate", dilate);

        //3x3 erode filter (minimum value of rectangle) (8 times iteration)    
        cvErode(img, erode, NULL, 8);
	cvMoveWindow("erode",0 , 0);
    	cvShowImage("erode", erode);

        //3x3 erode filter (minimum value of rectangle) (8 times iteration)    
        cvSmooth(img, median, CV_MEDIAN, 3); ///3x3 median filter
	cvMoveWindow("median", 0, 0);
    	cvShowImage("median", median);


        cvWaitKey();
    }else{
      printf("Please specify image location\n");
    }
    return 0;
}
