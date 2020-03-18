#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// In frames. The larger the more stable the video, but less reactive to sudden panning 移动平均滑动窗口大小
const int SMOOTHING_RADIUS = 50;

/**
 * @brief 运动信息结构体
 *
 */
struct TransformParam
{
	TransformParam() {}
	//x轴信息，y轴信息，角度信息
	TransformParam(double _dx, double _dy, double _da)
	{
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	double dx;
	double dy;
	// angle
	double da;

	void getTransform(Mat &T)
	{
		// Reconstruct transformation matrix accordingly to new values 重建变换矩阵
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);

		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;
	}
};

/**
 * @brief 轨迹结构体
 *
 */
struct Trajectory
{
	Trajectory() {}
	Trajectory(double _x, double _y, double _a)
	{
		x = _x;
		y = _y;
		a = _a;
	}

	double x;
	double y;
	// angle
	double a;
};

/**
 * @brief 轨迹累积
 *
 * @param transforms 运动信息结构体
 * @return vector<Trajectory> 轨迹结构体
 */
vector<Trajectory> cumsum(vector<TransformParam> &transforms)
{
	// trajectory at all frames 所有帧的运动轨迹
	vector<Trajectory> trajectory;
	// Accumulated frame to frame transform 累加计算x,y以及a（角度）
	double a = 0;
	double x = 0;
	double y = 0;

	//累加
	for (size_t i = 0; i < transforms.size(); i++)
	{
		x += transforms[i].dx;
		y += transforms[i].dy;
		a += transforms[i].da;

		trajectory.push_back(Trajectory(x, y, a));
	}

	return trajectory;
}

/**
 * @brief 平滑运动轨迹
 *
 * @param trajectory 运动轨迹
 * @param radius 窗格大小
 * @return vector<Trajectory>
 */
vector<Trajectory> smooth(vector<Trajectory> &trajectory, int radius)
{
	//平滑后的运动轨迹
	vector<Trajectory> smoothed_trajectory;
	//移动滑动窗格
	for (size_t i = 0; i < trajectory.size(); i++)
	{
		double sum_x = 0;
		double sum_y = 0;
		double sum_a = 0;
		int count = 0;

		for (int j = -radius; j <= radius; j++)
		{
			if (i + j >= 0 && i + j < trajectory.size())
			{
				sum_x += trajectory[i + j].x;
				sum_y += trajectory[i + j].y;
				sum_a += trajectory[i + j].a;

				count++;
			}
		}

		double avg_a = sum_a / count;
		double avg_x = sum_x / count;
		double avg_y = sum_y / count;

		smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));
	}

	return smoothed_trajectory;
}

/**
 * @brief
 *
 * @param frame_stabilized
 */
void fixBorder(Mat &frame_stabilized)
{
	//将原图扩大为1.04倍，然后截取原图尺寸相等大小区域
	Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows / 2), 0, 1.04);
	//仿射变换
	warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
}

int main(int argc, char **argv)
{
	// Read input video 读取视频
	VideoCapture cap("detect.mp4"); 
	//VideoCapture cap("C:/Users/Smartuil/Desktop/1.mp4");
	
	// Get frame count 读取视频总帧数
	int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));
	// Our test video may be wrong to read the frame after frame 1300
	n_frames = 50;

	// Get width and height of video stream 获取视频图像宽高
	int w = int(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));

	// Get frames per second (fps) 获取视频每秒帧数
	double fps = cap.get(CAP_PROP_FPS);

	// Set up output video 设置输出视频
	VideoWriter out("video_out.mp4", CAP_ANY, fps, Size(2 * w, h));

	// Define variable for storing frames 定义存储帧的相关变量
	//当前帧RGB图像和灰度图
	Mat curr, curr_gray;
	//前一帧RGB图像和灰度图
	Mat prev, prev_gray;

	// Read first frame 获得视频一张图象
	cap >> prev;

	// Convert frame to grayscale 转换为灰度图
	cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

	// Pre-define transformation-store array 仿射变化参数结构体
	vector<TransformParam> transforms;

	//previous transformation matrix 上一张图像的仿射矩阵
	Mat last_T;
	//从第二帧开始循环遍历视频所有帧
	for (int i = 1; i < n_frames; i++)
	{
		// Vector from previous and current feature points 前一帧角点vector，当前帧角点vector
		vector<Point2f> prev_pts, curr_pts;

		// Detect features in previous frame 获取前一帧的角点
		//前一帧灰度图，前一帧角点vector, 最大角点数，检测到的角点的质量等级，两个角点之间的最小距离
		goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);

		// Read next frame 读取当前帧图像
		bool success = cap.read(curr);
		if (!success)
		{
			break;
		}

		// Convert to grayscale 将当前帧图像转换为灰度图
		cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

		// Calculate optical flow (i.e. track feature points) 光流法追寻特征点
		//输出状态矢量(元素是无符号char类型，uchar)，如果在当前帧发现前一帧角点特征则置为1，否则，为0
		vector<uchar> status;
		//输出误差矢量
		vector<float> err;
		//光流跟踪
		//前一帧灰度图像，当前帧灰度图像，前一帧角点，当前帧角点，状态量，误差量
		calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

		// Filter only valid points 获取光流跟踪下有效的角点
		//遍历角点
		auto prev_it = prev_pts.begin();
		auto curr_it = curr_pts.begin();
		for (size_t k = 0; k < status.size(); k++)
		{
			if (status[k])
			{
				prev_it++;
				curr_it++;
			}
			//删除无效角点
			else
			{
				prev_it = prev_pts.erase(prev_it);
				curr_it = curr_pts.erase(curr_it);
			}
		}

		// Find transformation matrix 获得变换矩阵
		//false表示带几何约束的仿射变换，true则是全仿射变化，T为变换矩阵
		//Mat T = estimateRigidTransform(prev_pts, curr_pts, false);
		Mat T = estimateAffinePartial2D(prev_pts, curr_pts);

		// In rare cases no transform is found.
		// We'll just use the last known good transform.
		//极少数情况会找不到变换矩阵，取上一个变换为当前变化矩阵
		//当然第一次检测就没找到仿射矩阵，算法会出问题，不过概率很低
		if (T.data == NULL)
		{
			last_T.copyTo(T);
		}
		T.copyTo(last_T);

		// Extract traslation 提取仿射变化结果
		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);

		// Extract rotation angle 提取角度
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

		// Store transformation 存储仿射变化矩阵
		transforms.push_back(TransformParam(dx, dy, da));

		// Move to next frame 进行下一次检测准测
		curr_gray.copyTo(prev_gray);

		cout << "Frame: " << i << "/" << n_frames << " -  Tracked points : " << prev_pts.size() << endl;
	}

	// Compute trajectory using cumulative sum of transformations 获取累加轨迹
	vector<Trajectory> trajectory = cumsum(transforms);

	// Smooth trajectory using moving average filter 获取平滑后的轨迹
	vector<Trajectory> smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS);

	//平滑后的运动信息结构体
	vector<TransformParam> transforms_smooth;

	//原始运动信息结构体
	for (size_t i = 0; i < transforms.size(); i++)
	{
		// Calculate difference in smoothed_trajectory and trajectory 计算平滑后的轨迹和原始轨迹差异
		double diff_x = smoothed_trajectory[i].x - trajectory[i].x;
		double diff_y = smoothed_trajectory[i].y - trajectory[i].y;
		double diff_a = smoothed_trajectory[i].a - trajectory[i].a;

		// Calculate newer transformation array 计算平滑后的运动信息结构体数据
		double dx = transforms[i].dx + diff_x;
		double dy = transforms[i].dy + diff_y;
		double da = transforms[i].da + diff_a;

		transforms_smooth.push_back(TransformParam(dx, dy, da));
	}

	//定位当前帧为第1帧
	cap.set(CAP_PROP_POS_FRAMES, 0);
	//平滑后的变化矩阵
	Mat T(2, 3, CV_64F);
	Mat frame, frame_stabilized, frame_out;

	//对所有帧进行变化得到稳像结果
	//跳过第一帧
	cap.read(frame);
	for (int i = 0; i < n_frames - 1; i++)
	{
		bool success = cap.read(frame);
		if (!success)
		{
			break;
		}
		// Extract transform from translation and rotation angle. 提取平滑后的仿射变化矩阵
		transforms_smooth[i].getTransform(T);

		// Apply affine wrapping to the given frame 应用仿射变化
		warpAffine(frame, frame_stabilized, T, frame.size());

		// Scale image to remove black border artifact 去除黑边
		fixBorder(frame_stabilized);

		// Now draw the original and stablised side by side for coolness 将原图和变化后的图横向排列输出到视频
		hconcat(frame, frame_stabilized, frame_out);

		// If the image is too big, resize it.
		if (frame_out.cols > 1920)
		{
			resize(frame_out, frame_out, Size(frame_out.cols / 2, frame_out.rows / 2));
		}

		//imshow("Before and After", frame_out);
		out.write(frame_out);
		char ret[8];
		sprintf_s(ret,"%d.jpg",i);
		imwrite(ret, frame_out);
		cout << "out frame：" << i << endl;
		//waitKey(10);
	}

	// Release video
	cap.release();
	out.release();
	// Close windows
	destroyAllWindows();

	return 0;
}