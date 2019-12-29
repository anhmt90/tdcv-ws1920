#include "RandomForest.h"

RandomForest::RandomForest()
{
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
	:mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
	// Creates vector of decision trees = random forest
	for (uint treeIdx = 0; treeIdx < treeCount; treeIdx++) {
		// Creates empty decision tree
		mTrees.push_back(cv::ml::DTrees::create());

		// Configuration of runtime parameters
		mTrees[treeIdx]->setCVFolds(CVFolds); // If (CVFolds > 1) then prune the decision tree using K-fold cross-validation where K is equal to CVFolds
		mTrees[treeIdx]->setMaxCategories(maxCategories); // Limits the number of categorical values before which the decision tree will precluster those categories
		mTrees[treeIdx]->setMaxDepth(maxDepth); // Tree will not exceed this depth, but may be less deep
		mTrees[treeIdx]->setMinSampleCount(minSampleCount);  // Do not split a node if there are fewer than this number of samples at that node
	}
}

RandomForest::~RandomForest()
{
}

void RandomForest::setTreeCount(int treeCount)
{
	mTreeCount = treeCount;
}

void RandomForest::setMaxDepth(int maxDepth)
{
	mMaxDepth = maxDepth;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFolds)
{
	mCVFolds = cvFolds;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setCVFolds(mCVFolds);
}

void RandomForest::setMinSampleCount(int minSampleCount)
{
	mMinSampleCount = minSampleCount;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setMinSampleCount(mMinSampleCount);
}

void RandomForest::setMaxCategories(int maxCategories)
{
	mMaxCategories = maxCategories;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setMaxCategories(mMaxCategories);
}

std::vector<int> RandomForest::GenerateRandomVector(int NumberCount) {
	std::random_device rd;
	std::mt19937 gen(rd());

	std::vector<int> values(NumberCount);
	std::uniform_int_distribution<> dis(0, NumberCount - 1);
	std::generate(values.begin(), values.end(), [&]() { return dis(gen); });
	std::vector<int> v1(values.begin() + (rand() % (NumberCount / 2)), values.end() - (rand() % (NumberCount / 2)));
	std::fill(values.begin(), values.end(), 0);

	for (int i = 0; i < NumberCount; i++) {
		values[i] = v1[rand() % (v1.size() - 1)];
	}

	return values;
}

void RandomForest::train(const cv::Mat train_data, const cv::Mat train_labels)
{
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++) {

		auto rd = GenerateRandomVector(train_data.rows);
		cv::Mat rd_train_data, rd_train_labels;
		for (int i = 0; i < rd.size(); i++) {
			rd_train_data.push_back(train_data.row(rd[i]));
			rd_train_labels.push_back(train_labels.row(rd[i]));
		}
		auto ptr_training_data = cv::ml::TrainData::create(rd_train_data, cv::ml::ROW_SAMPLE, rd_train_labels);
		mTrees[treeIdx]->train(ptr_training_data);
	}
}

void RandomForest::calcError(cv::Ptr<cv::ml::TrainData> data, bool test)
{
	cv::Mat output, output_mat;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++) {
		mTrees[treeIdx]->calcError(data, false, output);
		output_mat.push_back(output.t());
	}

	// To do: Find majority vote over all predictions in output_mat and compute error

}

