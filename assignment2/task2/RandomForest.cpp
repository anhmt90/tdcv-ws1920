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

float RandomForest::calcError(cv::Ptr<cv::ml::TrainData> data)
{
	cv::Mat output, output_mat;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++) {
		mTrees[treeIdx]->calcError(data, false, output);
		output_mat.push_back(output.t());
	}

	std::vector<int> majorityVotes;
	for (uint i = 0; i < output_mat.cols; i++) {
		std::vector<int> col_vector;
		for (uint j = 0; j < output_mat.rows; j++) {
			col_vector.push_back(output_mat.at<float>(j, i));
		}
		std::sort(col_vector.begin(), col_vector.end());

		int max_count = 1, mode = col_vector[0], curr_count = 1;
		for (int j = 1; j < col_vector.size(); j++) {
			if (col_vector[j] == col_vector[j - 1]) {
				curr_count++;
				if (curr_count > max_count) {
					max_count = curr_count;
					mode = col_vector[j - 1];
				}
			}
			else {
				if (curr_count > max_count) {
					max_count = curr_count;
					mode = col_vector[j - 1];
				}
				curr_count = 1;
			}

		}
		majorityVotes.push_back(mode);
	}


	int count_error = 0;
	cv::Mat responses = data->getResponses();
	for (int i = 0; i < majorityVotes.size(); i++) {
		if (majorityVotes[i] != responses.at<signed int>(i)) {
			count_error++;
		}
	}
	float error = (float)count_error / (float)majorityVotes.size() * 100.0;

	return error;
}

