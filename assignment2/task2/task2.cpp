#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>




//#include "HOGDescriptor.h"
#include "RandomForest.h"

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

void performanceEval(cv::Ptr<cv::ml::DTrees> classifier, cv::Ptr<cv::ml::TrainData> data, bool test) {
	cv::Mat output;
	auto error = classifier->calcError(data, test, output);
	if (test) {
		cout << "Prediction error for training set: " << error << "%" << std::endl;
	}
	else {
		cout << "Prediction error for test set: " << error << "%" << std::endl;
	}
};

void eval_perf(cv::Ptr<RandomForest> classifier, const cv::Mat data, const cv::Mat labels, bool test) {
	auto error = classifier->calcError(data, labels);
	if (test) {
		cout << "Prediction error for training set: " << error << "%" << std::endl;
	}
	else {
		cout << "Prediction error for test set: " << error << "%" << std::endl;
	}
};

void testDTrees(cv::Mat train_data, cv::Mat labels, cv::Mat test_data, cv::Mat test_labels) {
	cout << "\nStart training DTREE..." << endl;
	int num_classes = 6;

	// Creates empty decision tree
	auto ptr_tree = ml::DTrees::create();

	// Configuration of runtime parameters
	ptr_tree->setCVFolds(1); // If (CVFolds > 1) then prune the decision tree using K-fold cross-validation where K is equal to CVFolds
	ptr_tree->setMaxCategories(num_classes); // Limits the number of categorical values before which the decision tree will precluster those categories
	ptr_tree->setMaxDepth(10); // Tree will not exceed this depth, but may be less deep
	ptr_tree->setMinSampleCount(20); // Do not split a node if there are fewer than this number of samples at that node

	// Creates training data from feature array
	auto ptr_training_data = ml::TrainData::create(train_data, ml::ROW_SAMPLE, labels);
	auto ptr_test_data = ml::TrainData::create(test_data, ml::ROW_SAMPLE, test_labels);

	// Train decision tree
	ptr_tree->train(ptr_training_data);

	// Calls function to compute the error of the trained decision tree
	cout << "Evaluation for Decision Tree" << std::endl;
	performanceEval(ptr_tree, ptr_training_data, true);
	performanceEval(ptr_tree, ptr_test_data, false);
}

void testForest(cv::Mat train_data, cv::Mat labels, cv::Mat test_data, cv::Mat test_labels) {
	cout << "\nStart training RANDOM FOREST..." << endl;

	// Configuration of runtime parameters
	int treeCount = 32; // Number of trees making up the forest
	int CVFolds = 1; // If (CVFolds > 1) then prune the decision tree using K-fold cross-validation where K is equal to CVFolds
	int maxCategories = 6; // Limits the number of categorical values before which the decision tree will precluster those categories
	int maxDepth = 10; // Tree will not exceed this depth, but may be less deep
	int minSampleCount = 15; // Do not split a node if there are fewer than this number of samples at that node

	// Initializing random forest with runtime parameters
	std::shared_ptr<RandomForest> ptr_random_forest(new RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories));

	// Train random forest
	ptr_random_forest->train(train_data, labels);

	// Calls function to compute the error of the trained random forest
	eval_perf(ptr_random_forest, train_data, labels, true);
	eval_perf(ptr_random_forest, test_data, test_labels, false);
}



int main() {
	// CODE TO READ THE IMAGES FROM TRAIN AND TEST, CALCULATE HOG FEATURES, AND SAVE THE LABELS OF EACH IMAGE //


	// TRAINING DATA //
	Mat hog_features_train, labels_train;

	// TEST DATA //
	Mat hog_features_test, labels_test; 

	const std::string str1("TRAIN");
	const std::string str2("TEST");

	std::vector<std::string> data_dirs = { str1, str2 };

	for (const std::string &dir: data_dirs)
	{
		cout << "Reading and processing of " << dir << " data ..." << std::endl;


		// Vector defining the folders inside the training folder. We will iterate through all folders to read the images inside each one of the labels.
		std::vector<std::string> train_subdirs = { "00", "01", "02", "03", "04", "05" };


		for (const std::string subdir : train_subdirs)
		{
			cout << "Reading Images on subfolder " << subdir << std::endl;

			// Define the general path where all training images are
			fs::path img_path("data/task2/");

			// Append each time one of the label folders you want to read
			// Generate the full path by adding the type of image we are searching for in the directory
			fs::path file("*jpg");
			fs::path full_path = (img_path / dir / subdir / file);

			cout << "path: " << full_path << endl;


			String folderpath = full_path.string();
			std::vector<String> filenames;
			// OpenCV function which gives all files names in that directory
			glob(folderpath, filenames);

			for (size_t j = 0; j < filenames.size(); j++)
			{
				// Print each file name read
				cout << filenames[j] << std::endl;
				Mat img = imread(filenames[j]);

				// Perform same operations on image as in task 1 & calculate HOG descriptors
				Mat grayImg;

				resize(img, img, Size(64, 64));
				cvtColor(img, grayImg, COLOR_RGB2GRAY);

				cv::HOGDescriptor myHog(Size(64, 64), Size(8, 8), Size(4, 4), Size(8, 8), 9);
				std::vector<float> feat_values;
				std::vector<Point> locations;
				myHog.compute(grayImg, feat_values, Size(0, 0), Size(0, 0), locations);

				//cout<< "Vector HOG features size " <<feat_values.size()<<endl;

				// Convert feature vector to a matrix
				Mat HOG_vector(1, feat_values.size(), CV_32FC1);// ml needs float data and flatten

				for (int i = 0;i < feat_values.size();i++)
					HOG_vector.at<float>(0, i) = feat_values.at(i);

				//cout<< "Mat HOG Features size" << hog_features.size()<<endl;

				// We need labels to be numbers, and we get it from the folder name
				// object from the class stringstream
				int class_label;
				std::stringstream subdir_name(subdir);
				subdir_name >> class_label;

				// Depending if if belongs to train or test save features & labels in different matrices
				if (str1.compare(dir) == 0) {
					hog_features_train.push_back(HOG_vector);         // append at bottom
					labels_train.push_back(class_label);
					cout << "Saved on TRAIN matrix " << std::endl;
				}
				else if (str2.compare(dir) == 0) {
					hog_features_test.push_back(HOG_vector);         // append at bottom
					labels_test.push_back(class_label);
					cout << "Saved on TEST matrix " << std::endl;
				}
			}
		}
	}

	cout << "Number of instances for training: " << hog_features_train.rows << std::endl;
	cout << "Number of instances for testing: " << hog_features_test.rows << std::endl;
	//cout << labels << endl;
	testDTrees(hog_features_train, labels_train, hog_features_test, labels_test);
	testForest(hog_features_train, labels_train, hog_features_test, labels_test);
	// TEST THE DECISION TREE ARCHITECTURE AND THE RANDOM FOREST //
	return 0;
}
