#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "DataAugmentation.hpp"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const string EXT = ".jpg";

const string VFLIP = "_vflip";
const string HFLIP = "_hflip";
const string ROTATE = "_rotate";
const string SCALE = "_scale";
const string NOISE = "_noisy";
const string SHEAR = "_shear";


DataAugmentation::DataAugmentation(fs::path ipath, fs::path opath)
	: inputPath(ipath), outputPath(opath) {}


vector<string> DataAugmentation::fetch_input_image_paths() {
	std::vector<string> filenames;
	// OpenCV function which gives all files names in that directory
	glob(inputPath.string(), filenames);
	return filenames;
}




void DataAugmentation::flip(int flipCode) {
	cv::flip(imgOut, imgOut, flipCode);
}


void DataAugmentation::rotate(double angle = 0, double scale = 1.0) {
	Point2f img_center(imgOut.cols / 2.0F, imgOut.rows / 2.0F);
	Mat M = cv::getRotationMatrix2D(img_center, angle, scale);
	cv::warpAffine(this->imgOut, this->imgOut, M, Size(imgOut.cols, imgOut.rows));
}


void DataAugmentation::add_gaussian_noise() {
	Mat noise(Size(imgOut.cols, imgOut.rows), imgOut.type());
	float mean = (10.0F, 12.0F, 34.0F);
	float sigma = (1.0F, 5.0F, 50.0F);
	cv::randn(noise, mean, sigma);
	imgOut += noise;
}

void DataAugmentation::shear(int shearCode)
{
	int r1, c1; // tranformed point
	int rows, cols; // original image rows and columns
	rows = imgOut.rows;
	cols = imgOut.cols;

	float Bx = 0.7; // amount of shearing in x-axis
	float By = 0.7; // amount of shearing in y-axis
	switch (shearCode)
	{
	case 0:
		Bx = 0.7F; By = 0.0F; break;
	case 1:
		Bx = 0.0F; By = 0.7F; break;
	case 2:
		Bx = -0.7F; By = 0.0F; break;
	case 3:
		Bx = 0.0F; By = -0.7F; break;
	default:
		break;
	}

	int maxXOffset = abs(cols * Bx);
	int maxYOffset = abs(rows * By);

	Mat out = Mat::ones(imgOut.rows, imgOut.cols, imgOut.type()); // create output image to be the same as the source

	for (int r = 0; r < out.rows; r++) // loop through the image
	{
		for (int c = 0; c < out.cols; c++)
		{
			r1 = r + c * By - maxYOffset; // map old point to new
			c1 = r * Bx + c - maxXOffset;

			if (r1 >= 0 && r1 <= rows && c1 >= 0 && c1 <= cols) // check if the point is within the boundaries
			{
				out.at<uchar>(r, c) = imgOut.at<uchar>(r1, c1); // set value
			}

		}
	}
	imgOut = out;
}


void DataAugmentation::refresh_img() {
	imgOut.release();
	imgOut = img.clone();
}

void DataAugmentation::save_img(string name) {
	fs::path fullPath = (outputPath / name);
	imwrite(fullPath.string(), imgOut);
	this_thread::sleep_for(chrono::nanoseconds(100));
	cout << fullPath.string() << " saved successfully" << endl;
	refresh_img();
}




string get_flip_sign(int flipCode) {
	switch (flipCode)
	{
	case -1:
		return VFLIP + HFLIP;
	case 0:
		return VFLIP;
	case 1:
		return HFLIP;
	default:
		return "Error in get_flip_sign(int)";
	}
}

string get_shear_sign(int shearCode) {
	switch (shearCode)
	{
	case 0:
		return "shearX";
	case 1:
		return "shearY";
	case 2:
		return "shearXneg";
	case 3:
		return "shearYneg";
	default:
		return "Error in get_flip_sign(int)";
	}
}

string get_rotate_sign(int angle) {
	return ROTATE + to_string(angle);
}

string get_scale_sign(float scale) {
	stringstream stream;
	stream << std::fixed << std::setprecision(0) << (scale * 100);
	return SCALE + stream.str();
}

void DataAugmentation::augment() {
	cout << "Start augmenting data ...";
	vector<string> imagePaths = fetch_input_image_paths();
	for (const string& pathString : imagePaths) {
		// Print each file name read
		//cout << input_names[j] << std::endl;

		fs::path path(pathString);
		string fileName = path.stem().string();
		img = imread(pathString);
		imgOut = img.clone();

		/*====================================================================================*/

		/*FLIP ONLY*/
		refresh_img();
		vector<int> flipCodes = { -1, 0, 1 };
		for (const int& code : flipCodes) {
			flip(code);
			save_img(fileName + get_flip_sign(code) + EXT);
		}

		/*ROTATE ONLY*/
		refresh_img();
		//vector<int> angles = {0, 45, 90, 135, 180, 225, 270, 315 };
		vector<int> angles = {0, 45, 90, 180, 270, };

		for (const auto& angle : angles) {
			rotate(angle);
			save_img(fileName + get_rotate_sign(angle) + EXT);
		}

		/*SCALE ONLY*/
		refresh_img();
		vector<float> scales = {1.0, 2.0 };

		//for (const float& scale : scales) {
		//	rotate(0, scale);
		//	save_img(fileName + get_scale_sign(scale) + EXT);
		//}

		///*SHEAR ONLY*/
		//vector<float> shears = { 0, 1, 2, 3 };
		//for (const float& s : shears) {
		//	shear(s);
		//}

		/*GAUSSIAN NOISE ONLY*/
		refresh_img();
		add_gaussian_noise();
		save_img(fileName + NOISE + EXT);
		/*====================================================================================*/

		/*COMBINE FLIP & ROTATE*/
		for (const int& flipCode : flipCodes) {
			for (const auto& angle : angles) {
				flip(flipCode);
				rotate(angle);
				save_img(fileName + get_flip_sign(flipCode) + get_rotate_sign(angle) + EXT);
			}
		}

		///*COMBINE FLIP & SCALE*/
		//for (const int& flipCode : flipCodes) {
		//	for (const float& scale : scales) {
		//		flip(flipCode);
		//		rotate(0, scale);
		//		save_img(fileName + get_flip_sign(flipCode) + get_scale_sign(scale) + EXT);
		//	}
		//}

		///*COMBINE FLIP & SHEAR*/
		//for (const int& flipCode : flipCodes) {
		//	for (const float& s : shears) {
		//		flip(flipCode);
		//		shear(s);
		//		save_img(fileName + get_flip_sign(flipCode) + get_shear_sign(s) + EXT);
		//	}
		//}

		///*COMBINE FLIP & NOISE*/
		//for (const int& flipCode : flipCodes) {
		//	flip(flipCode);
		//	add_gaussian_noise();
		//	save_img(fileName + get_flip_sign(flipCode) + NOISE + EXT);
		//}

		/*COMBINE ROTATE & SCALE*/
		for (const auto& angle : angles) {
			for (const float& scale : scales) {
				rotate(angle, scale);
				save_img(fileName + get_rotate_sign(angle) + get_scale_sign(scale) + EXT);
			}
		}

		///*COMBINE ROTATE & SHEAR*/
		//for (const auto& angle : angles) {
		//	for (const float& s : shears) {
		//		rotate(angle);
		//		shear(s);
		//		save_img(fileName + get_rotate_sign(angle) + get_shear_sign(s) + EXT);
		//	}
		//}

		///*COMBINE SCALE & SHEAR*/
		//for (const float& scale : scales) {
		//	for (const float& s : shears) {
		//		rotate(0, scale);
		//		shear(s);
		//		save_img(fileName + get_scale_sign(scale) + get_shear_sign(s) + EXT);
		//	}
		//}

		///*COMBINE ROTATE & NOISE*/
		//for (const auto& angle : angles) {
		//	rotate(angle);
		//	add_gaussian_noise();
		//	save_img(fileName + get_rotate_sign(angle) + NOISE + EXT);
		//}

		///*COMBINE SCALE & NOISE*/
		//for (const float& scale : scales) {
		//	rotate(0, scale);
		//	add_gaussian_noise();
		//	save_img(fileName + get_scale_sign(scale) + NOISE + EXT);
		//}

		/*====================================================================================*/

		/*COMBINE FLIP & ROTATE & SCALE & NOISE*/
		for (const int& flipCode : flipCodes) {
			for (const auto& angle : angles) {
				for (const float& scale : scales) {

					flip(flipCode);
					rotate(angle, scale);
					save_img(fileName + get_flip_sign(flipCode) + get_rotate_sign(angle) + get_scale_sign(scale) + EXT);

					//Mat backup = imgOut.clone();
					/*imgOut = backup;
					save_img(fileName + get_flip_sign(flipCode) + get_rotate_sign(angle) + get_scale_sign(scale) + NOISE + EXT);*/

				}
			}
		}


	}

}
