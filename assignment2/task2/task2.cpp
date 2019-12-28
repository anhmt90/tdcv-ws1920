
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>


/*#include "HOGDescriptor.h"*/
/*#include "RandomForest.h"*/

using namespace std;
using namespace cv;


namespace fs = std::experimental::filesystem;


template <class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data) {
    
    /*
     
     Fill Code
     
     */
    
    
};





void testDTrees(cv::Mat train_data, cv::Mat labels, cv::Mat test_data) {
    
    int num_classes = 6;
    
    /*
     * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
     * Train a single Decision Tree and evaluate the performance
     * Experiment with the MaxDepth parameter, to see how it affects the performance
     
     */
    
    Ptr<ml::DTrees> ptr_tree = ml::DTrees::create();
    ptr_tree->train(ml::TrainData::create(train_data, ml::ROW_SAMPLE, labels));
    
    //performanceEval<cv::ml::DTrees>(ptr_tree, train_data);
    //performanceEval<cv::ml::DTrees>(ptr_tree, test_data);
}


void testForest(){
    
    //int num_classes = 6;
    
    /*
     *
     * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
     * Train a Forest and evaluate the performance
     * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance
     
     */
    
    //performanceEval<RandomForest>(forest, train_data);
    //performanceEval<RandomForest>(forest, test_data);
}


int main(){

    // CODE TO READ THE IMAGES FROM TRAIN AND TEST, CALCULATE HOG FEATURES, AND SAVE THE LABELS OF EACH IMAGE //
    
    
    // TRAINING DATA //
    Mat featsImages, labels; // start empty
    
    // TEST DATA //
    Mat featsTestImages, labelsTest; // start empty
    
    string str1 ("train");
    string str2 ("test");

    vector<string> data = {"train", "test"};
    
    for (vector<string>::const_iterator h = data.begin(); h != data.end(); ++h)
    {
         cout << "Reading and processing of " << *h << " data ..."<< endl;
        
        
        // Vector defining the folders inside the training folder. We will iterate through all folders to read the images inside each one of the labels.
        vector<string> train_ImagesFolders = {"00", "01", "02", "03", "04", "05"};
        
        
        for (vector<string>::const_iterator i = train_ImagesFolders.begin(); i != train_ImagesFolders.end(); ++i)
        {
            cout << "Reading Images on subfolder " << *i << endl;
            
            // Define the general path where all training images are
            fs::path dir ("./data/task2/");
            
            // Append each time one of the label folders you want to read
            // Generate the full path by adding the type of image we are searching for in the directory
            fs::path file ("*jpg");
            fs::path full_path = dir / *h / *i / file;
        
            //cout << "path: " << full_path << endl;
            
            
            String folderpath = full_path;
            vector<String> filenames;
            // OpenCV function which gives all files names in that directory
            glob(folderpath, filenames);
            
            for (size_t j=0; j<filenames.size(); j++)
            {
                // Print each file name read
                cout<<filenames[j]<<endl;
                Mat img = imread(filenames[j]);
                
                // Perform same operations on image as in task 1 & calculate HOG descriptors
                Mat grayImg;
                Mat matFloat;
                
                resize(img, img, Size(64, 64));
                cvtColor(img, grayImg, COLOR_RGB2GRAY);
                
                HOGDescriptor myHog(Size(64, 64), Size(8, 8), Size(4, 4), Size(8, 8), 9);
                vector< float> feat_values;
                vector< Point> locations;
                myHog.compute(grayImg, feat_values, Size(0, 0), Size(0, 0), locations);
                
                //cout<< "Vector HOG features size " <<feat_values.size()<<endl;
                
                // Convert feature vector to a matrix
                Mat Hogfeat( 1, feat_values.size(), CV_32FC1);// ml needs float data and flatten
                
                for(int i=0;i<feat_values.size();i++)
                    Hogfeat.at<float>(0,i)=feat_values.at(i);
                
                //cout<< "Mat HOG Features size" << Hogfeat.size()<<endl;
                
                // We need labels to be numbers, and we get it from the folder name
                // object from the class stringstream
                stringstream geek(*i);
                
                // The object has the value corresponding to name of the folder and stream
                // it to the integer x
                int x;
                geek >> x;
                
                // Depending if if belongs to train or test save features & labels in different matrices
                if (str1.compare(*h) == 0 ) {
                    featsImages.push_back(Hogfeat);         // append at bottom
                    labels.push_back(x); // an integer, this is, what you get back in the prediction
                    cout<< "Saved on TRAIN matrix " <<endl;
                } else if (str2.compare(*h) == 0) {
                    featsTestImages.push_back(Hogfeat);         // append at bottom
                    labelsTest.push_back(x); // an integer, this is, what you get back in the prediction
                    cout<< "Saved on TEST matrix " <<endl;
                }
            }
        }
    }
    
    //cout << featsImages << endl;
    //cout << labels << endl;
    
    
    // TEST THE DECISION TREE ARCHITECTURE AND THE RANDOM FOREST //
    
    cout<<featsImages.size()<<endl;
    cout<<featsTestImages.size()<<endl;
    testDTrees(featsImages,labels,featsTestImages);
    testForest();
    
    return 0;
}
