//
//  hog_visualization.h
//  tdcv
//
//  Created by Maria on 27/12/2019.
//  Copyright © 2019 Fernando Benito Abad. All rights reserved.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>

void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor);
