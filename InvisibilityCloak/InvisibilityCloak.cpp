#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	//打开视频
	VideoCapture cap("video/input.mp4");

	// 检查视频是否打开
	if (!cap.isOpened())
	{
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	Mat background;
	//跳过前30帧
	for (int i = 0; i < 30; i++)
	{
		cap >> background;
	}
	//沿Y轴翻转图像
	flip(background, background, 1);
	//红布第251帧才出现，跳过前250帧
	for (int i = 0; i < 220; i++)
	{
		Mat frame_slip;
		cap >> frame_slip;
		continue;
	}


	//图像读取
	while (1)
	{
		//检测帧
		Mat frame;

		// Capture frame-by-frame
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
		{
			break;
		}
		//hsv图像
		Mat hsv;
		flip(frame, frame, 1);
		cvtColor(frame, hsv, COLOR_BGR2HSV);

		//红色区域1，红色区域2
		Mat mask1, mask2;
		//红色区域
		Mat mask_red;
		//背景区域
		Mat mask_background;
		//过滤颜色
		//二值图，其中黑色0表示无红色，白色1表示红色区域。
		inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
		inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
		mask_red = mask1 + mask2;


		//去除噪声
		Mat kernel = Mat::ones(3, 3, CV_32F);
		morphologyEx(mask_red, mask_red, cv::MORPH_OPEN, kernel);
		morphologyEx(mask_red, mask_red, cv::MORPH_DILATE, kernel);


		//将mask_red中0，1互换，得到背景区域范围。
		bitwise_not(mask_red, mask_background);
		Mat res1, res2, final_output;
		//从当前帧抠出背景区域res1,红布区域被涂成黑色。
		bitwise_and(frame, frame, res1, mask_background);
		//从背景帧提取红布区域覆盖的背景res2
		bitwise_and(background, background, res2, mask_red);

		addWeighted(res1, 1, res2, 1, 0, final_output);
		//展示图像
		imshow("Magic !!!", res1);
		// Press  ESC on keyboard to exit
		char c = (char)waitKey(1);
		if (c == 27)
		{
			break;
		}
	}

	return 0;
}