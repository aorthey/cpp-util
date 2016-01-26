/*
playetruct _IplImage
{
    int  nSize;
    int  ID;
    int  nChannels;
    int  alphaChannel;
    int  depth;
    char colorModel[4];
    char channelSeq[4];
    int  dataOrder;
    int  origin;
    int  align;
    int  width;
    int  height;
    struct _IplROI *roi;
    struct _IplImage *maskROI;
    void  *imageId;
    struct _IplTileInfo *tileInfo;
    int  imageSize;
    char *imageData;
    int  widthStep;
    int  BorderMode[4];
    int  BorderConst[4];
    char *imageDataOrigin;
}
*/

#include <cv.h>
#include <highgui.h>

int main(int argc, char* argv[]){

	if(argc == 2){
		IplImage *img = cvLoadImage(argv[1]);
		printf("Read image %s: \n",argv[1]);
		printf("-- total size: %d bytes\n",img->nSize);
		printf("-- image size: %d bytes\n",img->imageSize);
		printf("-- width x height: %d x %d \n",img->width,img->height);
		printf("-- color channels: %d \n",img->nChannels);
		printf("-- channel depth: %d \n",img->depth);
		cvNamedWindow("image loaded", CV_WINDOW_AUTOSIZE);
		cvMoveWindow("image loaded", 0, 0);
		cvShowImage("image loaded", img);
		cvWaitKey(0);
		cvReleaseImage(&img);
	}else{
		printf("usage: <imagepath>\n");
	}
	return 0;

}
