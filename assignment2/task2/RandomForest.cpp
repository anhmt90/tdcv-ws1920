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

int RandomForest::getNumTree() const 
{
	return mTreeCount;
}

std::vector<int> RandomForest::GenerateRandomVector(int NumberCount) {
	// Creates a random and uniform distributed subset  
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

std::vector<int> RandomForest::compute_majority_votes(std::vector<int> predictions) {
	// Sorts vector of predictions on one data sample
	std::sort(predictions.begin(), predictions.end());

	// Computes mode within sorted vector = most frequent vote
	int max_count = 1, mode = predictions[0], curr_count = 1;
	for (int j = 1; j < predictions.size(); j++) {
		if (predictions[j] == predictions[j - 1]) {
			curr_count++;
			if (curr_count > max_count) {
				max_count = curr_count;
				mode = predictions[j - 1];
			}
		}
		else {
			if (curr_count > max_count) {
				max_count = curr_count;
				mode = predictions[j - 1];
			}
			curr_count = 1;
		}
	}
	std::vector<int> majorityVote = { mode, max_count };
	return majorityVote;
}

void RandomForest::train(const cv::Mat train_data, const cv::Mat train_labels)
{
	// Trains each individual decision tree with a random subset of the input training data
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++) {
		// Gets index vector that corresponds to a random subset
		auto rd = GenerateRandomVector(train_data.rows);
		cv::Mat rd_train_data, rd_train_labels;
		// Creates random subset of training data
		for (int i = 0; i < rd.size(); i++) {
			rd_train_data.push_back(train_data.row(rd[i]));
			rd_train_labels.push_back(train_labels.row(rd[i]));
		}
		// Packages up training data along with training labels 
		auto ptr_training_data = cv::ml::TrainData::create(rd_train_data, cv::ml::ROW_SAMPLE, rd_train_labels);
		// Uses packaged training data to train each individual decision tree
		mTrees[treeIdx]->train(ptr_training_data);
	}
}

std::vector<std::vector<int>> RandomForest::predict(const cv::Mat data) {
	cv::Mat output, output_mat;
	// Creates a prediction matrix for the random forest, where each row corresponds to a single tree
	// and each column to a data sample
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++) {
		// Returns the leaf node of a decision tree corresponding to the input vector
		mTrees[treeIdx]->predict(data, output);
		// Adds row vector (output.t()) of predictions to matrix (output_mat)
		output_mat.push_back(output.t());
	}

	// Creates a vector containing the predicted class for each data sample and the number of trees
	// that voted for that particular sample: { class, #votes }
	// majorityVotes corresponds to an nx2 array
	std::vector<std::vector<int>> majorityVotes;
	for (uint i = 0; i < output_mat.cols; i++) {
		std::vector<int> col_vector;
		for (uint j = 0; j < output_mat.rows; j++) {
			col_vector.push_back(output_mat.at<float>(j, i));
		}
		majorityVotes.push_back(compute_majority_votes(col_vector));
	}
	return majorityVotes;
}

float RandomForest::calcError(const cv::Mat data, const cv::Mat labels)
{
	// Gets majority votes for input data
	std::vector<std::vector<int>> majorityVotes = predict(data);

	// Compares majority votes with labels and calculates error based on wrong predictions
	int count_error = 0;
	for (int _class = 0; _class < majorityVotes.size(); _class++) {
		if (majorityVotes[_class][0] != labels.at<signed int>(_class)) {
			count_error++;
		}
	}
	return (static_cast<float>(count_error) / majorityVotes.size()) * 100.0;
}

