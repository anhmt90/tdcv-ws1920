//
//  main.cpp
//  tdcv
//
//  Created by Fernando Benito Abad on 22.12.19.
//  Copyright © 2019 Fernando Benito Abad. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

Mat rotate(Mat src, double angle)
{
    Mat dst;
    Point2f pt(src.cols/2., src.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}
int main(int argc, const char * argv[]) {
    Mat img = imread("data/task1/obj1000.jpg");
    img.convertTo(img, CV_32F, 1/255.0);
    Mat gx, gy, mag, angle;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);
    cartToPolar(gx, gy, mag, angle, 1);
    Mat img_temp;
    // create descriptor object and set the size of the hog descriptor to image size
    /*cv::HOGDescriptor hog;
    hog.winSize = grayImg.size();*/
    for (int i=0; i<360/10; i++) {
        img_temp = rotate(mag, i*10);
        namedWindow("image", WINDOW_NORMAL);
        imshow("image", img_temp);
        waitKey(100);
    }
    // set the descriptor position to the middle of the image
    /*std::vector<cv::Point> positions;
    positions.push_back(cv::Point(grayImg.cols / 2, grayImg.rows / 2));
    std::vector<float> descriptor;
    hog.compute(grayImg,descriptor,cv::Size(),cv::Size(),positions);
    namedWindow("image", WINDOW_NORMAL);
    imshow("image", grayImg);
    waitKey(0);*/
    return 0;
}
