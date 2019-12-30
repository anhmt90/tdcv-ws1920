#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

class RandomForest
{
public:
	RandomForest();

	// You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);

	~RandomForest();

	void setTreeCount(int treeCount);
	void setMaxDepth(int maxDepth);
	void setCVFolds(int cvFols);
	void setMinSampleCount(int minSampleCount);
	void setMaxCategories(int maxCategories);

	std::vector<int> GenerateRandomVector(int NumberCount);

	void train(const cv::Mat train_data, const cv::Mat train_labels);

	float calcError(cv::Ptr<cv::ml::TrainData> data);


private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;
	// M-Trees for constructing thr forest
	std::vector<cv::Ptr<cv::ml::DTrees>> mTrees;
};

#endif //RF_RANDOMFOREST_H
