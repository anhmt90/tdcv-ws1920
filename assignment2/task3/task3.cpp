#include <iostream>
#include <fstream>
#include <filesystem>
#include <math.h>
#include <numeric>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "../task2/RandomForest.h"
#include "DataAugmentation.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;
namespace fs = std::filesystem;

const fs::path IMG_PATH("data/task3/");
const vector<string> TRAIN_SUBDIRS = { "00", "01", "02", "03" };
const fs::path FILES("*jpg");
const string AUGMENT_DIR = "augmented";

const float LOWER_PERCENT = 0.01F;
const float UPPER_PERCENT = 0.25F;

const uint NUM_TREES = 128;
const uint BACKGROUND_CLASS = 3;

struct BBox {
    Rect rect;
    int num_vote;
    int class_;
    float confidence;
};

struct Preds {
    Rect rect;
    int class_;
    float IoU_score;
};


void PR_curve(vector<Preds> predictions)
{
    cout << "Calculating PRECISSION and RECALL ..." << std::endl;
    
    vector<float> precVect;
    vector<float> recallVect;
    int correct;
    int incorrect;
    
    for (float thres=0; thres < 1; thres += 0.1) {
        correct = 0;
        incorrect = 0;
        
        for (int i = 0; i<predictions.size(); i++) {
            if (predictions[i].IoU_score >= thres) {
                correct++;
            }
            else{
                incorrect++;
            }
        }
        
        precVect.push_back(correct / predictions.size());
        cout << "Precision Value ..." << precVect.back() << std::endl;
        
        recallVect.push_back(correct / (3 * 43)); // Each test image has 3 objects;
        cout << "Recall Value ..." << recallVect.back() << std::endl;
    }
    
    // Export to csv precision & recall values
    cout << "Exporting to csv ..." << std::endl;
    
    ofstream precfile;
    precfile.open ("precision_example.csv");
    for (int n = 0; n <= precVect.size(); n++) {
        precfile << precVect[n] << endl;
        
    }
    precfile.close();
    
    ofstream recallfile;
    recallfile.open ("recall_example.csv");
    for (int n = 0; n <= precVect.size(); n++) {
        recallfile << recallVect[n] << endl;
    }
    recallfile.close();
}


float closest_power_2(int x)
{
	return pow(2, round(log(x) / log(2)));
}

float compute_IoU(BBox best_box, vector<BBox> gt_box)
{
    // Determine the (x,y)-coordinates of the intersection rectangle
    int idx = best_box.class_;
    
    int w_intersection = min(best_box.rect.x + best_box.rect.width, gt_box[idx].rect.x + gt_box[idx].rect.width) - max(best_box.rect.x,gt_box[idx].rect.x);
    
    int h_intersection = min(best_box.rect.y + best_box.rect.height, gt_box[idx].rect.y + gt_box[idx].rect.height) - max(best_box.rect.y,gt_box[idx].rect.y);
    
    if (w_intersection <= 0 || h_intersection <= 0){
        return 0;
    }
        
    // Compute the area of the intersection rectangle
    float interArea = w_intersection * h_intersection;
    
    // Compute the area of both the prediction and ground-truth rectangles
    float predArea = best_box.rect.width * best_box.rect.height;
    float gtArea = gt_box[best_box.class_].rect.width * gt_box[best_box.class_].rect.height;
    
    float unionArea = predArea + gtArea - interArea;
    
    // Compute interction over union
    float iou = interArea / unionArea;
    
    return iou;
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

	for (const std::string& subdir : TRAIN_SUBDIRS)
	{
		cout << "Reading Images in subfolder " << subdir << std::endl;
		fs::path full_path = (IMG_PATH / dir / subdir / AUGMENT_DIR / FILES);

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

bool draw_boxes(Mat& img, const vector<BBox>& candidate_boxes = vector<BBox>())
{
	const uint numShowRects = 1000;
	Mat _img = img.clone();
	for (int i = 0; i < candidate_boxes.size(); i++) {
		BBox cb = candidate_boxes[i];
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

vector<int> get_seq(int start, int end) {
	std::vector<int> vec(end);
	std::iota(std::begin(vec), std::end(vec), start);
	return vec;
}

vector<BBox> get_bboxes(const vector<Rect>& rects, const std::vector<std::vector<int>>& votes, const vector<int>& selected_indices = {})
{
	assert(rects.size() == votes.size());
	vector<int> indices = (selected_indices.size() == 0) ?
		get_seq(0, rects.size()) : selected_indices;

	vector<BBox> bboxes;
	for (const uint& indx : indices) {
		BBox cb;
		cb.class_ = votes.at(indx).at(0);
		if (cb.class_ == BACKGROUND_CLASS)
			continue;
		cb.num_vote = votes.at(indx).at(1);
		cb.rect = rects.at(indx);
		cb.confidence = static_cast<float>(cb.num_vote) / NUM_TREES;
		bboxes.push_back(cb);
	}
	return bboxes;
}

vector<Preds> perform(Ptr<RandomForest>& rf_classifier, bool show_imgs = false)
{
    setUseOptimized(true);
    setNumThreads(4);

    const std::string dir("TEST");
    fs::path full_path = (IMG_PATH / dir / FILES);

    std::vector<String> filenames;
    glob(full_path.string(), filenames);
    
    const std::string dir_gt("GT");
    fs::path gt_path = (IMG_PATH / dir_gt / "*txt");

    std::vector<String> gt_files;
    glob(gt_path.string(), gt_files);

    int fid = 0;
    vector<Preds> predictions;
    
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

        assert(votes.size() == rects.size());

        vector<BBox> bboxes = get_bboxes(rects, votes);
        set<uint> classes;
        for (const auto& cb : bboxes)
            classes.insert(cb.class_);

        

        /*NON-MAXIMUM SUPRESSION*/
        const int &NUM_CLASSES = classes.size();
        const float CONFIDENCE_THRESHOLD = 0.6;
        const float SUPPRESS_THRESHOLD = 0.2;


        vector<vector<Rect>> rects_by_class(NUM_CLASSES);
        vector<vector<float>> scores_by_class(NUM_CLASSES);
        vector<vector<BBox>> bbox_by_class(NUM_CLASSES);

        for (const auto& bbox : bboxes) {
            rects_by_class[bbox.class_].push_back(bbox.rect);
            scores_by_class[bbox.class_].push_back(bbox.confidence);
            bbox_by_class[bbox.class_].push_back(bbox);
        }

        assert(rects_by_class.size() == scores_by_class.size() && bbox_by_class.size() == rects_by_class.size());

        vector<BBox> best_bboxes;
        for (uint i = 0; i < rects_by_class.size(); i++) {
            assert(rects_by_class[i].size() == scores_by_class[i].size() && bbox_by_class[i].size() == rects_by_class[i].size());

            vector<int> nms_output_indices;
            cv::dnn::NMSBoxes(rects_by_class[i], scores_by_class[i], CONFIDENCE_THRESHOLD, SUPPRESS_THRESHOLD, nms_output_indices);

            cout << "Number of best Region Proposals for class " << i << ": " << nms_output_indices.size() << endl;
            for (const int& nms_indx : nms_output_indices) {
                best_bboxes.push_back(bbox_by_class[i].at(nms_indx));
            }


        }
        cout << "Total number of best Region Proposals: "  << best_bboxes.size() << endl;
        
        //Read txt files to get bounding boxes ground truths for each image
        vector<BBox> gt_boxes;
        ifstream gt_file(gt_files[fid]);
        string str;
        while (getline(gt_file, str)){
            istringstream iss(str);
            vector<string> splitted((istream_iterator<string>(iss)),
            istream_iterator<string>());
            BBox cb;
            cb.class_ = stoi(splitted[0]);
            cb.rect.x = round((stoi(splitted[1]) + stoi(splitted[3]))/2.0);
            cb.rect.y = round((stoi(splitted[2]) + stoi(splitted[4]))/2.0);
            cb.rect.width = abs(stoi(splitted[1]) - stoi(splitted[3]));
            cb.rect.height = abs(stoi(splitted[2]) - stoi(splitted[4]));
            gt_boxes.push_back(cb);
        }
        for (int i=0; i<best_bboxes.size(); i++) {
            Preds myPred;
            myPred.rect = best_bboxes[i].rect;
            myPred.class_ = best_bboxes[i].class_;
            myPred.IoU_score = compute_IoU(best_bboxes[i], gt_boxes);
            predictions.push_back(myPred);
        }
        fid++;
        
        if (show_imgs)
            show_imgs = draw_boxes(_img, best_bboxes);
    }
    
    return predictions;
}


int main() {
	bool runAugmentation = 1;
	if (runAugmentation) {
		for (const std::string subdir : TRAIN_SUBDIRS) {
			fs::path augmentInputPath = (IMG_PATH / "train" / subdir);
			fs::path augmentOutputPath(augmentInputPath / AUGMENT_DIR);

			if (fs::is_directory(augmentOutputPath) && fs::exists(augmentOutputPath))
				fs::remove_all(augmentOutputPath);
			
			fs::create_directory(augmentOutputPath);
			DataAugmentation augmentation(DataAugmentation(augmentInputPath, augmentOutputPath));
			augmentation.augment();
		}
	}

	Mat hog_features_train, labels_train;
	prepare_train_features(hog_features_train, labels_train);
	Ptr<RandomForest> rf_classifier = train_random_forest(hog_features_train, labels_train);

    vector<Preds> predictions;
    predictions = perform(rf_classifier, false);
    PR_curve(predictions);
	return 0;
}
