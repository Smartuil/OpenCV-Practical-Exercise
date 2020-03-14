#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

//解码
void decode(const Mat &scores, const Mat &geometry, float scoreThresh,
	std::vector<RotatedRect> &detections, std::vector<float> &confidences);

/**
 * @brief
 *
 * @param srcImg 检测图像
 * @param inpWidth 深度学习图像输入宽
 * @param inpHeight 深度学习图像输入高
 * @param confThreshold 置信度
 * @param nmsThreshold 非极大值抑制算法阈值
 * @param net
 * @return Mat
 */
Mat text_detect(Mat srcImg, int inpWidth, int inpHeight, float confThreshold, float nmsThreshold, Net net)
{
	//输出
	std::vector<Mat> output;
	std::vector<String> outputLayers(2);
	outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
	outputLayers[1] = "feature_fusion/concat_3";

	//检测图像
	Mat frame, blob;
	frame = srcImg.clone();
	//获取深度学习模型的输入
	blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
	net.setInput(blob);
	//输出结果
	net.forward(output, outputLayers);

	//置信度
	Mat scores = output[0];
	//位置参数
	Mat geometry = output[1];

	// Decode predicted bounding boxes， 对检测框进行解码，获取文本框位置方向
	//文本框位置参数
	std::vector<RotatedRect> boxes;
	//文本框置信度
	std::vector<float> confidences;
	decode(scores, geometry, confThreshold, boxes, confidences);

	// Apply non-maximum suppression procedure， 应用非极大性抑制算法
	//符合要求的文本框
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Render detections. 输出预测
	//缩放比例
	Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		RotatedRect &box = boxes[indices[i]];

		Point2f vertices[4];
		box.points(vertices);
		//还原坐标点
		for (int j = 0; j < 4; ++j)
		{
			vertices[j].x *= ratio.x;
			vertices[j].y *= ratio.y;
		}
		//画框
		for (int j = 0; j < 4; ++j)
		{
			line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
		}
	}

	// Put efficiency information. 时间
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

	return frame;
}

//模型地址
auto model = "frozen_east_text_detection.pb";
//检测图像
auto detect_image = "patient.jpg";
//输入框尺寸
auto inpWidth = 320;
auto inpHeight = 320;
//置信度阈值
auto confThreshold = 0.5;
//非极大值抑制算法阈值
auto nmsThreshold = 0.4;

int main()
{
	//读取模型
	Net net = readNet(model);
	//读取检测图像
	Mat srcImg = imread(detect_image);
	if (!srcImg.empty())
	{
		cout << "read image success!" << endl;
	}
	Mat resultImg = text_detect(srcImg, inpWidth, inpHeight, confThreshold, nmsThreshold, net);
	imshow("result", resultImg);
	waitKey();
	return 0;
}

/**
 * @brief 输出检测到的文本框相关信息
 *
 * @param scores 置信度
 * @param geometry 位置信息
 * @param scoreThresh 置信度阈值
 * @param detections 位置
 * @param confidences 分类概率
 */
void decode(const Mat &scores, const Mat &geometry, float scoreThresh, std::vector<RotatedRect> &detections, std::vector<float> &confidences)
{
	detections.clear();
	//判断是不是符合提取要求
	CV_Assert(scores.dims == 4);
	CV_Assert(geometry.dims == 4);
	CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1);
	CV_Assert(scores.size[1] == 1);
	CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]);
	CV_Assert(scores.size[3] == geometry.size[3]);

	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		//识别概率
		const float *scoresData = scores.ptr<float>(0, 0, y);
		//文本框坐标
		const float *x0_data = geometry.ptr<float>(0, 0, y);
		const float *x1_data = geometry.ptr<float>(0, 1, y);
		const float *x2_data = geometry.ptr<float>(0, 2, y);
		const float *x3_data = geometry.ptr<float>(0, 3, y);
		//文本框角度
		const float *anglesData = geometry.ptr<float>(0, 4, y);
		//遍历所有检测到的检测框
		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			//低于阈值忽略该检测框
			if (score < scoreThresh)
			{
				continue;
			}

			// Decode a prediction.
			// Multiple by 4 because feature maps are 4 time less than input image.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			//角度及相关正余弦计算
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];

			Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
			Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
			//旋转矩形，分别输入中心点坐标，图像宽高，角度
			RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			//保存检测框
			detections.push_back(r);
			//保存检测框的置信度
			confidences.push_back(score);
		}
	}
}