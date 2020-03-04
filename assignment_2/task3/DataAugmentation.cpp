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

void DataAugmentation::shear()
{

	Mat M(2, 3, CV_32F);


	M.at<float>(0, 0) = 1.0F;
	M.at<float>(0, 1) = 0.0F;
	M.at<float>(0, 2) = 0.0F;

	M.at<float>(1, 0) = 0.6F;
	M.at<float>(1, 1) = 1.0F;
	M.at<float>(1, 2) = 0.0F;

	warpAffine(imgOut, imgOut, M, Size(imgOut.cols, imgOut.rows));

	///////////////////////////////////

	/*float Bx;
	float By;
	if (shearCode == 0) {
		Bx = 0.7;
		By = 0;
	}
	else if (shearCode == 2) {
		Bx = -0.7;
		By = 0;
	}


	//if (input.type() != CV_8UC3) return cv::Mat();


	//// shear the extreme positions to find out new image size:
	std::vector<cv::Point2f> extremePoints;
	extremePoints.push_back(cv::Point2f(0, 0));
	extremePoints.push_back(cv::Point2f(imgOut.cols, 0));
	extremePoints.push_back(cv::Point2f(imgOut.cols, imgOut.rows));
	extremePoints.push_back(cv::Point2f(0, imgOut.rows));

	for (unsigned int i = 0; i < extremePoints.size(); ++i)
	{
		cv::Point2f& pt = extremePoints[i];
		pt = cv::Point2f(pt.x + pt.y * Bx, pt.y + pt.x * By);
	}
	cv::Rect offsets = cv::boundingRect(extremePoints);
	cv::Point2f offset = -offsets.tl();
	cv::Size resultSize = offsets.size();

	cv::Mat shearedImage = cv::Mat::zeros(resultSize, imgOut.type()); // every pixel here is implicitely shifted by "offset"
	// perform the shearing by back-transformation
	for (int j = 0; j < shearedImage.rows; ++j)
	{

		for (int i = 0; i < shearedImage.cols; ++i)
		{
			cv::Point2f pp(i, j);

			pp = pp - offset; // go back to original coordinate system

			// go back to original pixel:
			// x'=x+y·Bx
			// y'=y+x*By
			//   y = y'-x*By
			//     x = x' -(y'-x*By)*Bx
			//     x = +x*By*Bx - y'*Bx +x'
			//     x*(1-By*Bx) = -y'*Bx +x'
			//     x = (-y'*Bx +x')/(1-By*Bx)

			cv::Point2f p;
			p.x = (-pp.y * Bx + pp.x) / (1 - By * Bx);
			p.y = pp.y - p.x * By;

			if ((p.x >= 0 && p.x < imgOut.cols) && (p.y >= 0 && p.y < imgOut.rows))
			{
				// TODO: interpolate, if wanted (p is floating point precision and can be placed between two pixels)!
				shearedImage.at<cv::Vec3b>(j, i) = imgOut.at<cv::Vec3b>(p);
			}
		}
	}
	imgOut.release();
	imgOut = shearedImage;*/

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

string get_shear_sign(int shearCode = 0) {
	return "_shear";
	switch (shearCode)
	{
	case 0:
		return "_shearX";
	case 1:
		return "_shearY";
	case 2:
		return "_shearXneg";
	case 3:
		return "_shearYneg";
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
		vector<int> angles = { 0, 45, 90, 135, 180, 225, 270, 315 };
		//vector<int> angles = {0, 45, 90, 180, 270, };

		for (const auto& angle : angles) {
			rotate(angle);
			save_img(fileName + get_rotate_sign(angle) + EXT);
		}

		/*SCALE ONLY*/
		refresh_img();
		vector<float> scales = { 0.63, 1.0, 1.75 };

		//for (const float& scale : scales) {
		//	rotate(0, scale);
		//	save_img(fileName + get_scale_sign(scale) + EXT);
		//}

		/*SHEAR ONLY*/
		shear();
		save_img(fileName + get_shear_sign() + EXT);

		/*GAUSSIAN NOISE ONLY*/
		/*refresh_img();
		add_gaussian_noise();
		save_img(fileName + NOISE + EXT);*/
		/*====================================================================================*/

		/*COMBINE FLIP & ROTATE*/
		for (const int& flipCode : flipCodes) {
			for (const auto& angle : angles) {
				flip(flipCode);
				rotate(angle);
				save_img(fileName + get_flip_sign(flipCode) + get_rotate_sign(angle) + EXT);
			}
		}

		/*COMBINE FLIP & SCALE*/
		for (const int& flipCode : flipCodes) {
			for (const float& scale : scales) {
				flip(flipCode);
				rotate(0, scale);
				save_img(fileName + get_flip_sign(flipCode) + get_scale_sign(scale) + EXT);
			}
		}

		/*COMBINE FLIP & SHEAR*/
		for (const int& flipCode : flipCodes) {
			flip(flipCode);
			shear();
			save_img(fileName + get_flip_sign(flipCode) + get_shear_sign() + EXT);

		}

		/*COMBINE FLIP & NOISE*/
		for (const int& flipCode : flipCodes) {
			flip(flipCode);
			add_gaussian_noise();
			save_img(fileName + get_flip_sign(flipCode) + NOISE + EXT);
		}

		/*COMBINE ROTATE & SCALE*/
		for (const auto& angle : angles) {
			for (const float& scale : scales) {
				rotate(angle, scale);
				save_img(fileName + get_rotate_sign(angle) + get_scale_sign(scale) + EXT);
			}
		}

		/*COMBINE ROTATE & SHEAR*/
		for (const auto& angle : angles) {
			rotate(angle);
			shear();
			save_img(fileName + get_rotate_sign(angle) + get_shear_sign() + EXT);
		}

		/*COMBINE SCALE & SHEAR*/
		for (const float& scale : scales) {
			rotate(0, scale);
			shear();
			save_img(fileName + get_scale_sign(scale) + get_shear_sign() + EXT);
		}

		/*COMBINE ROTATE & NOISE*/
		for (const auto& angle : angles) {
			rotate(angle);
			add_gaussian_noise();
			save_img(fileName + get_rotate_sign(angle) + NOISE + EXT);
		}

		///*COMBINE SCALE & NOISE*/
		//for (const float& scale : scales) {
		//	rotate(0, scale);
		//	add_gaussian_noise();
		//	save_img(fileName + get_scale_sign(scale) + NOISE + EXT);
		//}

		/*====================================================================================*/

		/*COMBINE FLIP & ROTATE & SCALE*/
		for (const int& flipCode : flipCodes) {
			for (const auto& angle : angles) {
				for (const float& scale : scales) {
						flip(flipCode);
						rotate(angle, scale);
						Mat backup = imgOut.clone();
						save_img(fileName + get_flip_sign(flipCode) + get_rotate_sign(angle) + get_scale_sign(scale) + EXT);

						imgOut = backup;
						shear();
						save_img(fileName + get_flip_sign(flipCode) + get_rotate_sign(angle) + get_shear_sign() + get_scale_sign(scale)  + EXT);
				}
			}
		}


	}

}
