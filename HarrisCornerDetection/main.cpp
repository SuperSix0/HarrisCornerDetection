#include <io.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

double* Gaussian;
double k = 0.04;
int k100;
int apertureSize = 3;
Mat picture, GrayPicture;
string path, pic;
int *Ix, *Iy;
int *Ix2, *Iy2, *Ixy;
double *A, *B, *C;
int height, width;
Mat maxPicture, minPicture, RPicture, NMSPicture, finalPicture, window;

double* getGaussian(int n)
{
	double* dst;
	double sum = 0, sigma = (n / 2) * 0.3 + 0.8;
	dst = (double*)malloc(sizeof(double) * n * n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*(dst + i * n + j) = exp(-0.5 * (pow(i - n / 2, 2) + pow(j - n / 2, 2)) / pow(sigma, 2));
			sum += *(dst + i * n + j);
		}
	}
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			*(dst + i * n + j) /= sum;
	return dst;
}

void R()
{
	double a, b, c, max = 0;
	double *tmp, temp;
	tmp = (double*)malloc(sizeof(double) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a = *(A + i * width + j);
			b = *(B + i * width + j);
			c = *(C + i * width + j);
			temp = a * b - c * c - 0.01 * k100 * pow(a + b, 2);
			*(tmp + i * width + j) = temp;
			if (temp > max)
				max = temp;
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			temp = *(tmp + i * width + j);
			if(temp > max * 0.01)
				RPicture.at<double>(i,j) = temp;
			else
				RPicture.at<double>(i, j) = 0;
		}
	}
	imwrite(path + "\\R图_" + pic, RPicture);
}

void NMS()
{
	bool flag;
	int row, col;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			flag = true;
			for (int subrow = -1; subrow <= 1; subrow++)
			{
				for (int subcol = -1; subcol <= 1; subcol++)
				{
					row = i + subrow;
					col = j + subcol;
					if (row < 0 || row >= height)
						row = i;
					if (col < 0 || col >= width)
						col = j;
					if (row == i && col == j)
						continue;
					if (RPicture.at<double>(i,j) < RPicture.at<double>(row, col))
						flag = false;
				}
			}
			if (flag)
				NMSPicture.at<double>(i, j) = RPicture.at<double>(i, j);
			else
				NMSPicture.at<double>(i, j) = 0;
		}
	}
	finalPicture = picture.clone();
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (NMSPicture.at<double>(i, j) > 0)
				circle(finalPicture, Point(j, i), 3, Scalar(0, 0, 255));
		}
	}
	imwrite(path + "\\原图叠加检测效果_" + pic, finalPicture);
}

Mat Merge(Mat mat1, Mat mat2, Mat mat3, Mat mat4)
{
	Mat dst(height * 2, width * 2, picture.type());
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<unsigned char>(i, j * 3) = (unsigned char)mat1.at<double>(i, j);
			dst.at<unsigned char>(i, j * 3 + 1) = (unsigned char)mat1.at<double>(i, j);
			dst.at<unsigned char>(i, j * 3 + 2) = (unsigned char)mat1.at<double>(i, j);
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<unsigned char>(i, (j + width) * 3) = (unsigned char)mat2.at<double>(i, j);
			dst.at<unsigned char>(i, (j + width) * 3 + 1) = (unsigned char)mat2.at<double>(i, j);
			dst.at<unsigned char>(i, (j + width) * 3 + 2) = (unsigned char)mat2.at<double>(i, j);
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<unsigned char>(i + height, j * 3) = (unsigned char)mat3.at<double>(i, j);
			dst.at<unsigned char>(i + height, j * 3 + 1) = (unsigned char)mat3.at<double>(i, j);
			dst.at<unsigned char>(i + height, j * 3 + 2) = (unsigned char)mat3.at<double>(i, j);
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<unsigned char>(i + height, (j + width) * 3) = (unsigned char)mat4.at<unsigned char>(i, j * 3);
			dst.at<unsigned char>(i + height, (j + width) * 3 + 1) = (unsigned char)mat4.at<unsigned char>(i, j * 3 + 1);
			dst.at<unsigned char>(i + height, (j + width) * 3 + 2) = (unsigned char)mat4.at<unsigned char>(i, j * 3 + 2);
		}
	}
	return dst;
}

void on_Trackbar(int, void*)
{
	R();
	NMS();
	
	double font_scale = 1;
	int thickness = 2, baseline;
	Size text_size = getTextSize("一", CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, thickness, &baseline);
	putText(maxPicture, "Max", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(minPicture, "Min", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(RPicture, "R", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(finalPicture, "Result", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	
	window = Merge(maxPicture, minPicture, RPicture, finalPicture);
	imshow("Harris Corner Detection", window);
}


int main(int argc, char** argv)
{
	path = string(*(argv + 1));
	picture = imread(path);
	while (picture.empty())
	{
		cout << "无法打开图片，请输入正确路径：\n";
		cin >> path;
		picture = imread(path);
	}

	pic = path.substr(path.find_last_of("\\") + 1);
	path = path.substr(0, path.find_last_of("\\"));

	if (argc > 2)
		k = atof(*(argv + 2));
	k100 = k * 100;

	if (argc > 3)
		apertureSize = atof(*(argv + 3));
	while (apertureSize % 2 == 0 || apertureSize <= 1)
	{
		cout << "apertureSize不符，请重新输入：\n";
		cin >> apertureSize;
	}

	Gaussian = getGaussian(apertureSize);

	cvtColor(picture, GrayPicture, CV_RGB2GRAY);

	height = picture.rows;
	width = picture.cols;
	Ix = (int *)malloc(sizeof(int) * height * width);
	Iy = (int *)malloc(sizeof(int) * height * width);

	int row, col;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			row = i;
			col = j;
			int row1 = i - 1, row2 = i + 1;
			int col1 = j - 1, col2 = j + 1;
			if (row1 < 0)
				row1 = 0;
			if (row2 >= height)
				row2 = height - 1;
			if (col1 < 0)
				col1 = 0;
			if (col2 >= width)
				col2 = width - 1;

			*(Ix + i * width + j) = GrayPicture.data[row * width + col2] - GrayPicture.data[row * width + col1];
			*(Iy + i * width + j) = GrayPicture.data[row2 * width + col] - GrayPicture.data[row1 * width + col];
		}
	}

	Ix2 = (int *)malloc(sizeof(int) * height * width);
	Iy2 = (int *)malloc(sizeof(int) * height * width);
	Ixy = (int *)malloc(sizeof(int) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			*(Ix2 + i * width + j) = pow(*(Ix + i * width + j), 2);
			*(Iy2 + i * width + j) = pow(*(Iy + i * width + j), 2);
			*(Ixy + i * width + j) = *(Ix + i * width + j) * *(Iy + i * width + j);
		}
	}

	double a, b, c;
	A = (double *)malloc(sizeof(double) * height * width);
	B = (double *)malloc(sizeof(double) * height * width);
	C = (double *)malloc(sizeof(double) * height * width);
	apertureSize /= 2;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a = 0;
			b = 0;
			c = 0;
			for (int subrow = -apertureSize; subrow < apertureSize; subrow++)
			{
				for (int subcol = -apertureSize; subcol < apertureSize; subcol++)
				{
					row = i + subrow;
					col = j + subcol;
					if (row < 0 || row >= height)					//防止图像边界越界
						row = i;
					if (col < 0 || col >= width)
						col = j;
					a += *(Gaussian + (subrow + apertureSize) * width + subcol + apertureSize) * *(Ix2 + row * width + col);
					b += *(Gaussian + (subrow + apertureSize) * width + subcol + apertureSize) * *(Iy2 + row * width + col);
					c += *(Gaussian + (subrow + apertureSize) * width + subcol + apertureSize) * *(Ixy + row * width + col);
				}
			}
			*(A + i * width + j) = a;
			*(B + i * width + j) = b;
			*(C + i * width + j) = c;
		}
	}

	double max, min;
	maxPicture.create(height, width, GrayPicture.type());
	maxPicture.convertTo(maxPicture, CV_64F);
	minPicture.create(height, width, GrayPicture.type());
	minPicture.convertTo(minPicture, CV_64F);
	RPicture.create(height, width, GrayPicture.type());
	RPicture.convertTo(RPicture, CV_64F);
	NMSPicture.create(height, width, GrayPicture.type());
	NMSPicture.convertTo(NMSPicture, CV_64F);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a = *(A + i * width + j);
			b = *(B + i * width + j);
			c = *(C + i * width + j);
			if (a > b)
				max = sqrt(a), min = sqrt(b);
			else
				min = sqrt(a), max = sqrt(b);
			maxPicture.at<double>(i, j) = max;
			minPicture.at<double>(i, j) = min;
		}
	}
	imwrite(path + "\\最大特征值图_" + pic, maxPicture);
	imwrite(path + "\\最小特征值图_" + pic, minPicture);

	R();
	NMS();
	namedWindow("Harris Corner Detection");

	double font_scale = 1;
	int thickness = 2, baseline;
	Size text_size = getTextSize("一", CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, thickness, &baseline);
	putText(maxPicture, "Max", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(minPicture, "Min", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(RPicture, "R", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(finalPicture, "Result", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	
	window = Merge(maxPicture,minPicture,RPicture,finalPicture);
	imshow("Harris Corner Detection", window);

	createTrackbar("k * 100", "Harris Corner Detection", &k100, 25, on_Trackbar);
	waitKey(0);
	return 0;
}