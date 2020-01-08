#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

class DataAugmentation
{
private:
	fs::path inputPath;
	fs::path outputPath;

	Mat img;
	Mat imgOut;


	vector<string> fetch_input_image_paths();
	void flip(int);
	void rotate(double, double);
	void add_gaussian_noise();
	void refresh_img();
	void save_img(string name);
	void shear(int);

public:
	DataAugmentation(fs::path, fs::path);
	void augment();

};