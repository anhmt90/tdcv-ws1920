#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>
#include <random>



#include "HOGDescriptor.h"
#include "RandomForest.h"

using namespace cv;

namespace fs = std::experimental::filesystem;

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data, bool test) {
	cv::Mat output;
	auto error = classifier->calcError(data, test, output);
	std::cout << "Evaluation for Decision Tree" << std::endl;
	if (test) {
		std::cout << "Prediction error for training set: " << error << "%" << std::endl;
	}
	else {
		std::cout << "Prediction error for test set: " << error << "%" << std::endl;
	}
};

void testDTrees(cv::Mat train_data, cv::Mat labels, cv::Mat test_data, cv::Mat test_labels) {

	int num_classes = 6;

	// Creates empty decision tree
	auto ptr_tree = ml::DTrees::create();

	// Configuration of runtime parameters
	ptr_tree->setCVFolds(1); // If (CVFolds > 1) then prune the decision tree using K-fold cross-validation where K is equal to CVFolds
	ptr_tree->setMaxCategories(num_classes); // Limits the number of categorical values before which the decision tree will precluster those categories
	ptr_tree->setMaxDepth(10); // Tree will not exceed this depth, but may be less deep
	ptr_tree->setMinSampleCount(40); // Do not split a node if there are fewer than this number of samples at that node

	// Creates training data from feature array
	auto ptr_training_data = ml::TrainData::create(train_data, ml::ROW_SAMPLE, labels);
	auto ptr_test_data = ml::TrainData::create(test_data, ml::ROW_SAMPLE, test_labels);

	// Train decision tree
	ptr_tree->train(ptr_training_data);

	// Calls function to compute the error of the trained decision tree
	performanceEval<cv::ml::DTrees>(ptr_tree, ptr_training_data, true);
	performanceEval<cv::ml::DTrees>(ptr_tree, ptr_test_data, false);
}

void testForest(cv::Mat train_data, cv::Mat labels, cv::Mat test_data, cv::Mat test_labels) {

	// Configuration of runtime parameters
	int treeCount = 10;
	int CVFolds = 1;
	int maxCategories = 6;
	int maxDepth = 10;
	int minSampleCount = 40;

	// Initializing random forest with runtime parameters
	std::shared_ptr<RandomForest> ptr_random_forest(new RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories));

	// Train random forest
	ptr_random_forest->train(train_data, labels);

	// Creates training data from feature array
	auto ptr_training_data = ml::TrainData::create(train_data, ml::ROW_SAMPLE, labels);
	auto ptr_test_data = ml::TrainData::create(test_data, ml::ROW_SAMPLE, test_labels);

	// Calls function to compute the error of the trained random forest
	//performanceEval<RandomForest>(ptr_random_forest, ptr_training_data, true);
	//performanceEval<RandomForest>(ptr_random_forest, ptr_test_data, false);
}



int main() {
	// CODE TO READ THE IMAGES FROM TRAIN AND TEST, CALCULATE HOG FEATURES, AND SAVE THE LABELS OF EACH IMAGE //


// TRAINING DATA //
	Mat featsImages, labels; // start empty

	// TEST DATA //
	Mat featsTestImages, labelsTest; // start empty

	std::string str1("train");
	std::string str2("test");

	std::vector<std::string> data = { "train", "test" };

	for (std::vector<std::string>::const_iterator h = data.begin(); h != data.end(); ++h)
	{
		std::cout << "Reading and processing of " << *h << " data ..." << std::endl;


		// Vector defining the folders inside the training folder. We will iterate through all folders to read the images inside each one of the labels.
		std::vector<std::string> train_ImagesFolders = { "00", "01", "02", "03", "04", "05" };


		for (std::vector<std::string>::const_iterator i = train_ImagesFolders.begin(); i != train_ImagesFolders.end(); ++i)
		{
			std::cout << "Reading Images on subfolder " << *i << std::endl;

			// Define the general path where all training images are
			fs::path dir("./data/task2/");

			// Append each time one of the label folders you want to read
			// Generate the full path by adding the type of image we are searching for in the directory
			fs::path file("*jpg");
			fs::path full_path = (dir / *h / *i / file);

			//cout << "path: " << full_path << endl;


			String folderpath = full_path.string();
			std::vector<String> filenames;
			// OpenCV function which gives all files names in that directory
			glob(folderpath, filenames);

			for (size_t j = 0; j < filenames.size(); j++)
			{
				// Print each file name read
				std::cout << filenames[j] << std::endl;
				Mat img = imread(filenames[j]);

				// Perform same operations on image as in task 1 & calculate HOG descriptors
				Mat grayImg;
				Mat matFloat;

				resize(img, img, Size(64, 64));
				cvtColor(img, grayImg, COLOR_RGB2GRAY);

				cv::HOGDescriptor myHog(Size(64, 64), Size(8, 8), Size(4, 4), Size(8, 8), 9);
				std::vector<float> feat_values;
				std::vector<Point> locations;
				myHog.compute(grayImg, feat_values, Size(0, 0), Size(0, 0), locations);

				//cout<< "Vector HOG features size " <<feat_values.size()<<endl;

				// Convert feature vector to a matrix
				Mat Hogfeat(1, feat_values.size(), CV_32FC1);// ml needs float data and flatten

				for (int i = 0;i < feat_values.size();i++)
					Hogfeat.at<float>(0, i) = feat_values.at(i);

				//cout<< "Mat HOG Features size" << Hogfeat.size()<<endl;

				// We need labels to be numbers, and we get it from the folder name
				// object from the class stringstream
				std::stringstream geek(*i);

				// The object has the value corresponding to name of the folder and stream
				// it to the integer x
				int x;
				geek >> x;

				// Depending if if belongs to train or test save features & labels in different matrices
				if (str1.compare(*h) == 0) {
					featsImages.push_back(Hogfeat);         // append at bottom
					labels.push_back(x); // an integer, this is, what you get back in the prediction
					std::cout << "Saved on TRAIN matrix " << std::endl;
				}
				else if (str2.compare(*h) == 0) {
					featsTestImages.push_back(Hogfeat);         // append at bottom
					labelsTest.push_back(x); // an integer, this is, what you get back in the prediction
					std::cout << "Saved on TEST matrix " << std::endl;
				}
			}
		}
	}

	std::cout << featsImages.rows << std::endl;
	//cout << labels << endl;
	//testDTrees(featsImages, labels, featsTestImages, labelsTest);
	testForest(featsImages, labels, featsTestImages, labelsTest);
	// TEST THE DECISION TREE ARCHITECTURE AND THE RANDOM FOREST //
	return 0;
}
