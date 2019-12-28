#include <opencv2/opencv.hpp>
#include <iostream>
#include "hog_visualization.h"

using namespace std;
using namespace cv;


/*#include "HOGDescriptor.h"*/

int main(int argc, const char * argv[]){
    /*Mat im = cv::imread("data/task1/obj1000.jpg");*/

	//Fill Code here

    /*
    	* Create instance of HOGDescriptor and initialize
    	* Compute HOG descriptors
    	* visualize
    */

    Mat img = imread("data/task1/obj1000.jpg");
    Mat grayImg;

    resize(img, img, Size(64, 64));
    cvtColor(img, grayImg, COLOR_RGB2GRAY);
    
    HOGDescriptor myHog(Size(64, 64), Size(8, 8), Size(4, 4), Size(8, 8), 9);
    vector< float> feat_values;
    vector< Point> locations;
    myHog.compute(grayImg, feat_values, Size(0, 0), Size(0, 0), locations);
    visualizeHOG(img, feat_values, HOGDescriptor (Size(64, 64), Size(8, 8), Size(4, 4), Size(8, 8), 9), 5);
    /*for ( const auto &row : feat_vect ){
        visualizeHOG(img, row, myHog, 1);
    }*/
    
    return 0;
}
