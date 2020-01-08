#include <iostream>
#include <fstream>
#include <filesystem>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>


#include "../task2/RandomForest.h"

using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;
namespace fs = std::filesystem;

const fs::path IMG_PATH("data/task3/");
const fs::path FILES("*jpg");

const float LOWER_PERCENT = 0.01;
const float UPPER_PERCENT = 0.25;

const uint NUM_TREES = 128;
const uint BACKGROUND_CLASS = 3;

struct CandidateBox {
	Rect rect;
	int num_vote;
	int class_;
	float confidence;
};


float closest_power_2(int x)
{
	return pow(2, round(log(x) / log(2)));
}


Mat compute_HOG(Mat& img)
{
	// Perform same operations on image as in task 1 & calculate HOG descriptors
	Mat grayImg;

	resize(img, img, Size(64, 64));
	cvtColor(img, grayImg, COLOR_RGB2GRAY);

	cv::HOGDescriptor myHog(Size(48, 48), Size(24, 24), Size(12, 12), Size(12, 12), 9);
	std::vector<float> HOG_values;
	std::vector<Point> locations;
	myHog.compute(grayImg, HOG_values, Size(0, 0), Size(0, 0), locations);

	//cout<< "Vector HOG features size " <<feat_values.size()<<endl;

	// Convert feature vector to a matrix
	Mat HOG_features(1, HOG_values.size(), CV_32FC1);// ml needs float data and flatten

	for (int i = 0; i < HOG_values.size(); i++)
		HOG_features.at<float>(0, i) = HOG_values.at(i);

	//cout<< "Mat HOG Features size" << hog_features.size()<<endl;
	return HOG_features;
}

void prepare_train_features(Mat& hog_features_train, Mat& labels_train)
{
	const std::string dir("TRAIN");
	cout << "Reading and processing of TRAIN data ..." << std::endl;

	std::vector<std::string> train_subdirs = { "00", "01", "02", "03" };
	for (const std::string subdir : train_subdirs)
	{
		cout << "Reading Images on subfolder " << subdir << std::endl;
		fs::path full_path = (IMG_PATH / dir / subdir / FILES);

		std::vector<String> filenames;
		// OpenCV function which gives all files names in that directory
		glob(full_path.string(), filenames);
		for (size_t j = 0; j < filenames.size(); j++)
		{
			// Print each file name read
			//cout << filenames[j] << std::endl;
			Mat img = imread(filenames[j]);

			Mat hog_features = compute_HOG(img);

			// We need labels to be numbers, and we get it from the folder name
			// object from the class stringstream
			int class_label;
			std::stringstream subdir_name(subdir);
			subdir_name >> class_label;

			hog_features_train.push_back(hog_features);
			labels_train.push_back(class_label);
		}
	}
	cout << hog_features_train.rows << " rows of HOG features added to TRAIN data" << std::endl;
}

bool draw_boxes(Mat& img, const vector<CandidateBox>& candidate_boxes = vector<CandidateBox>())
{
	int numShowRects = 1000;
	Mat _img = img.clone();
	for (int i = 0; i < candidate_boxes.size(); i++) {
		CandidateBox cb = candidate_boxes[i];
		if (i < numShowRects) {
			Scalar color;
			switch (cb.class_)
			{
			case 0:
				color = Scalar(0, 255, 255); break;
			case 1:
				color = Scalar(255, 0, 0); break;
			case 2:
				color = Scalar(0, 0, 255); break;
			case 3:
				color = Scalar(255, 255, 255); break;
			default: break;
			}
			rectangle(_img, cb.rect, color);
		}
		else
			break;
	}

	imshow("Image with bounding boxes", _img);

	int k = waitKey();
	if (k == 113 || k == 27) {
		cout << "Skip showing the remaining images" << endl;
		return false;

	}
	return true;
}


Mat get_padded_ROI(const Mat& input, cv::Rect roi) {
	int top_left_x = roi.x;
	int top_left_y = roi.y;
	int width = roi.width;
	int height = roi.height;

	int bottom_right_x = top_left_x + width;
	int bottom_right_y = top_left_y + height;

	Mat output;
	if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
		// border padding will be required
		int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

		if (top_left_x < 0) {
			width = width + top_left_x;
			border_left = -1 * top_left_x;
			top_left_x = 0;
		}
		if (top_left_y < 0) {
			height = height + top_left_y;
			border_top = -1 * top_left_y;
			top_left_y = 0;
		}
		if (bottom_right_x > input.cols) {
			width = width - (bottom_right_x - input.cols);
			border_right = bottom_right_x - input.cols;
		}
		if (bottom_right_y > input.rows) {
			height = height - (bottom_right_y - input.rows);
			border_bottom = bottom_right_y - input.rows;
		}

		Rect R(top_left_x, top_left_y, width, height);
		copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, BORDER_REPLICATE);

		/*imshow("Image with bounding boxes", output);
		int k = waitKey();*/
	}
	else {
		// no border padding required
		Rect R(top_left_x, top_left_y, width, height);
		output = input(R);
	}

	return output;
}


void eval_perf(cv::Ptr<RandomForest> classifier, const cv::Mat data, const cv::Mat labels, bool test = true) {
	float error = classifier->calcError(data, labels);
	if (test) {
		cout << "Prediction error for training set: " << error << "%" << std::endl;
	}
	else {
		cout << "Prediction error for test set: " << error << "%" << std::endl;
	}
};

Ptr<RandomForest> train_random_forest(cv::Mat data, cv::Mat labels) {
	cout << "\nStart training RANDOM FOREST..." << endl;

	// Configuration of runtime parameters
	int treeCount = NUM_TREES;
	int CVFolds = 1; // If (CVFolds > 1) then prune the decision tree using K-fold cross-validation where K is equal to CVFolds
	int maxCategories = 4; // Limits the number of categorical values before which the decision tree will precluster those categories
	int maxDepth = 15; // Tree will not exceed this depth, but may be less deep
	int minSampleCount = 5; // Do not split a node if there are fewer than this number of samples at that node

	// Initializing random forest with runtime parameters
	std::shared_ptr<RandomForest> rf_classifier(new RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories));

	// Train random forest
	rf_classifier->train(data, labels);

	// Calls function to compute the error of the trained random forest
	eval_perf(rf_classifier, data, labels, true);
	return rf_classifier;
}

vector<CandidateBox> get_best_boxes(const vector<Rect>& rects, const std::vector<std::vector<int>>& votes, const vector<int>& best_box_indices)
{
	vector<CandidateBox> candidate_boxes;
	for (const uint& indx : best_box_indices) {
		CandidateBox cb;
		cb.class_ = votes.at(indx).at(0);
		if (cb.class_ == BACKGROUND_CLASS)
			continue;
		cb.num_vote = votes.at(indx).at(1);
		cb.rect = rects.at(indx);
		cb.confidence = cb.num_vote / NUM_TREES;
		candidate_boxes.push_back(cb);
	}
	return candidate_boxes;
}

void perform(Ptr<RandomForest>& rf_classifier, bool show_imgs = false)
{
	setUseOptimized(true);
	setNumThreads(4);

	const std::string dir("TEST");
	fs::path full_path = (IMG_PATH / dir / FILES);

	std::vector<String> filenames;
	glob(full_path.string(), filenames);


	for (const auto& file : filenames) {
		Mat img = imread(file);

		/*SELECTIVE SEARCH*/
		Ptr<SelectiveSearchSegmentation> ss_seg = createSelectiveSearchSegmentation();
		ss_seg->setBaseImage(img);
		ss_seg->switchToSelectiveSearchQuality();

		std::vector<Rect> rects;
		ss_seg->process(rects);
		std::cout << "Total Number of Region Proposals for image " << file << ": " << rects.size() << std::endl;

		Mat _img = img.clone();

		/*PROPOSAL FILTERING*/
		double img_area = static_cast<double>(_img.cols)* _img.rows;
		double lower_bound = LOWER_PERCENT * img_area;
		double upper_bound = UPPER_PERCENT * img_area;

		auto iter = rects.begin();
		while (iter != rects.end()) {
			(*iter).height = (*iter).width = max((*iter).height, (*iter).width); // squaring all proposal windows


			double area = static_cast<double>((*iter).width)* (*iter).height;
			(area <= lower_bound || area >= upper_bound) ?
				iter = rects.erase(iter) : ++iter;
		}

		std::cout << "Total Number of Region Proposals for image AFTER FILTERED " << file << ": " << rects.size() << std::endl;

		/*PROPOSAL PREPROCESSING & RUNNING HOG*/
		Mat HOG_features;
		for (auto& rect : rects) {
			Mat roi = get_padded_ROI(_img, rect);
			Mat HOG_vector = compute_HOG(roi);

			HOG_features.push_back(HOG_vector);
		}
		std::vector<std::vector<int>> votes = rf_classifier->predict(HOG_features);


		/*NON-MAXIMUM SUPRESSION*/
		const float CONFIDENCE_THRESHOLD = 0.6;
		const float SUPPRESS_THRESHOLD = 0.3;


		vector<float> confidence_scores;
		for (uint i = 0; i < votes.size(); i++) {
			auto& vote = votes[i];
			assert(vote.size == 2);
			float confidence = static_cast<float>(vote.at(1)) / NUM_TREES;
			confidence_scores.push_back(confidence);
		}

		vector<int> best_box_indices;

		cv::dnn::NMSBoxes(rects, confidence_scores, CONFIDENCE_THRESHOLD, SUPPRESS_THRESHOLD, best_box_indices);
		
		vector<CandidateBox> best_boxes = get_best_boxes(rects, votes, best_box_indices);
		cout << "Number of best Region Proposals: " << best_boxes.size() << endl;
		

		if(show_imgs)
			show_imgs =	draw_boxes(_img, best_boxes);
	}
}


int main() {
	Mat hog_features_train, labels_train;

	prepare_train_features(hog_features_train, labels_train);
	Ptr<RandomForest> rf_classifier = train_random_forest(hog_features_train, labels_train);

	perform(rf_classifier, true);
	return 0;
}
