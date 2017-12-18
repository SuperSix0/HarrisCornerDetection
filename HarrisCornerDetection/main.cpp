#include <io.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	string path, pic;
	Mat picture;
	
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
	
	double k = 0.04;
	if (argc > 2)
		k = atof(*(argv + 2));

	int apertureSize = 3;
	if (argc > 3)
		apertureSize = atof(*(argv + 3));
	while (apertureSize % 2 == 0 || apertureSize <= 1)
	{
		cout << "apertureSize不符，请重新输入：\n";
		cin >> apertureSize;
	}
	
	double *Gaussian;										//生成高斯二维分布函数
	double sum = 0, sigma = (apertureSize / 2) * 0.3 + 0.8;
	Gaussian = (double*)malloc(sizeof(double) * apertureSize * apertureSize);
	for (int i = 0; i < apertureSize; i++)
	{
		for (int j = 0; j < apertureSize; j++)
		{
			*(Gaussian + i * apertureSize + j) = exp(-0.5 * (pow(i - apertureSize / 2, 2) + pow(j - apertureSize / 2, 2)) / pow(sigma, 2));
			sum += *(Gaussian + i * apertureSize + j);
		}
	}
	for (int i = 0; i < apertureSize; i++)
		for (int j = 0; j < apertureSize; j++)
			*(Gaussian + i * apertureSize + j) /= sum;
	
	Mat GrayPicture;
	cvtColor(picture, GrayPicture, CV_RGB2GRAY);

	int height = picture.rows;								//计算x、y方向偏导数
	int width = picture.cols;
	double *Ix, *Iy;
	Ix = (double *)malloc(sizeof(double) * height * width);
	Iy = (double *)malloc(sizeof(double) * height * width);
	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int row1 = i > 0 ? (i - 1) : i;
			int row2 = (i + 1) < height ? (i + 1) : i;
			int col1 = j > 0 ? (j - 1) : j;
			int col2 = (j + 1) < width ? (j + 1) : j;

			*(Ix + i * width + j) = GrayPicture.at<unsigned char>(i, col2) - GrayPicture.at<unsigned char>(i, col1);
			*(Iy + i * width + j) = GrayPicture.at<unsigned char>(row2, j) - GrayPicture.at<unsigned char>(row1, j);
		}
	}
	
	double *Ix2, *Iy2, *Ixy;								//计算Ix2、Iy2、Ixy
	Ix2 = (double *)malloc(sizeof(double) * height * width);
	Iy2 = (double *)malloc(sizeof(double) * height * width);
	Ixy = (double *)malloc(sizeof(double) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			*(Ix2 + i * width + j) = pow(*(Ix + i * width + j), 2);
			*(Iy2 + i * width + j) = pow(*(Iy + i * width + j), 2);
			*(Ixy + i * width + j) = *(Ix + i * width + j) * *(Iy + i * width + j);
		}
	}
	
	double a, b, c;											//将二维高斯分布函数作为加权函数计算A、B、C
	int row, col;
	double *A, *B, *C;
	A = (double *)malloc(sizeof(double) * height * width);
	B = (double *)malloc(sizeof(double) * height * width);
	C = (double *)malloc(sizeof(double) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a = 0;
			b = 0;
			c = 0;
			for (int subrow = 0; subrow < apertureSize; subrow++)
			{
				for (int subcol = 0; subcol < apertureSize; subcol++)
				{
					row = i + subrow - apertureSize / 2;
					col = j + subcol - apertureSize / 2;
					if (row < 0 || row >= height)					//防止图像边界越界
						row = i;
					if (col < 0 || col >= width)
						col = j;
					a += *(Gaussian + subrow * width + subcol) * *(Ix2 + row * width + col);
					b += *(Gaussian + subrow * width + subcol) * *(Iy2 + row * width + col);
					c += *(Gaussian + subrow * width + subcol) * *(Ixy + row * width + col);
				}
			}
			*(A + i * width + j) = a;
			*(B + i * width + j) = b;
			*(C + i * width + j) = c;
		}
	}

	double max, min;											//计算最大特征值和最小特征值
	Mat maxPicture(height, width, GrayPicture.type());
	maxPicture.convertTo(maxPicture, CV_64F);
	Mat minPicture(height, width, GrayPicture.type());
	minPicture.convertTo(minPicture, CV_64F);
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
	
	Mat RPicture(height, width, GrayPicture.type());
	RPicture.convertTo(RPicture, CV_64F);
	
	max = 0;														//获得角点响应值R的最大值
	double *tmp, temp;
	tmp = (double*)malloc(sizeof(double) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a = *(A + i * width + j);
			b = *(B + i * width + j);
			c = *(C + i * width + j);
			temp = a * b - k * pow(a + b, 2);
			*(tmp + i * width + j) = temp;
			if (temp > max)
				max = temp;
		}
	}
	for (int i = 0; i < height; i++)								//设置阀值为最大值*0.01
	{
		for (int j = 0; j < width; j++)
		{
			temp = *(tmp + i * width + j);
			if (temp > max * 0.01)
				RPicture.at<double>(i, j) = temp;
			else
				RPicture.at<double>(i, j) = 0;
		}
	}
	imwrite(path + "\\R图_" + pic, RPicture);
	
	Mat NMSPicture(height, width, GrayPicture.type());				//非极大值抑制
	NMSPicture.convertTo(NMSPicture, CV_64F);
	bool flag;
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
					if (RPicture.at<double>(i, j) < RPicture.at<double>(row, col))
						flag = false;
				}
			}
			if (flag)
				NMSPicture.at<double>(i, j) = RPicture.at<double>(i, j);
			else
				NMSPicture.at<double>(i, j) = 0;
		}
	}
	
	Mat finalPicture = picture.clone();								//将角点在原图中用圆标注出来
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (NMSPicture.at<double>(i, j) > 0)
				circle(finalPicture, Point(j, i), 3, Scalar(0, 0, 255));
		}
	}
	imwrite(path + "\\原图叠加检测效果_" + pic, finalPicture);

	namedWindow("Harris Corner Detection");							//合并为一个图像并在窗口显示

	double font_scale = 1;
	int thickness = 2, baseline;
	Size text_size = getTextSize("一", CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, thickness, &baseline);
	putText(maxPicture, "Max", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(minPicture, "Min", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(RPicture, "R", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	putText(finalPicture, "Result", Point(0, text_size.height), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness, 8, false);
	
	Mat window(height * 2, width * 2, picture.type());
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			window.at<Vec3b>(i, j)[0] = (unsigned char)maxPicture.at<double>(i, j);
			window.at<Vec3b>(i, j)[1] = (unsigned char)maxPicture.at<double>(i, j);
			window.at<Vec3b>(i, j)[2] = (unsigned char)maxPicture.at<double>(i, j);
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			window.at<Vec3b>(i, j + width)[0] = (unsigned char)minPicture.at<double>(i, j);
			window.at<Vec3b>(i, j + width)[1] = (unsigned char)minPicture.at<double>(i, j);
			window.at<Vec3b>(i, j + width)[2] = (unsigned char)minPicture.at<double>(i, j);
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			window.at<Vec3b>(i + height, j)[0] = (unsigned char)RPicture.at<double>(i, j);
			window.at<Vec3b>(i + height, j)[1] = (unsigned char)RPicture.at<double>(i, j);
			window.at<Vec3b>(i + height, j)[2] = (unsigned char)RPicture.at<double>(i, j);
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			window.at<Vec3b>(i + height, j + width)[0] = finalPicture.at<Vec3b>(i, j)[0];
			window.at<Vec3b>(i + height, j + width)[1] = finalPicture.at<Vec3b>(i, j)[1];
			window.at<Vec3b>(i + height, j + width)[2] = finalPicture.at<Vec3b>(i, j)[2];
		}
	}
	
	imshow("Harris Corner Detection", window);

	waitKey(0);
	return 0;
}