#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

using namespace std;

class RandomForest
{
public:
	RandomForest();

	// You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories, string);

	~RandomForest();

	void setTreeCount(int treeCount);
	void setMaxDepth(int maxDepth);
	void setCVFolds(int cvFols);
	void setMinSampleCount(int minSampleCount);
	void setMaxCategories(int maxCategories);

	int getNumTree() const;
	int getMaxDepth() const;
	int getCVFolds() const;
	string getName() const;

	std::vector<int> GenerateRandomVector(int NumberCount);
	std::vector<int> compute_majority_votes(std::vector<int> unsortedPredictions);

	void train(const cv::Mat train_data, const cv::Mat train_labels);
	std::vector<std::vector<int>> predict(const cv::Mat data);
	float calcError(const cv::Mat data, const cv::Mat labels);


private:
	string name;
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;
	// M-Trees for constructing the forest
	std::vector<cv::Ptr<cv::ml::DTrees>> mTrees;
};

#endif //RF_RANDOMFOREST_H
