#include <fstream>
#include <sstream>
#include <iostream>
#include <time.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters 初始参数
 // Confidence threshold 置信度阈值
float confThreshold = 0.5;
// Non-maximum suppression threshold 非极大性抑制阈值
float nmsThreshold = 0.4;
//检测图像宽高
int inpWidth = 416;
int inpHeight = 416;
//类别参数
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
// 基于非极大性抑制去除低置信度的检测框
void postprocess(Mat& frame, const vector<Mat>& out);

// 画预测框
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// 提取输出输出层
vector<String> getOutputsNames(const Net& net);

int main()
{
	// Give the configuration and weight files for the model 模型参数文件
	String modelConfiguration = "yolov3.cfg";
	String modelWeights = "yolov3.weights";
	// Load names of classes 读取分类类名
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
	{
		classes.push_back(line);
	}

	VideoCapture cap("run.mp4");

	// Load the network 导入网络
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//仅仅使用CPU
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	Mat blob;
	clock_t start, finish;

	while (waitKey(1) < 0) {
		//读图
		//Mat frame = imread("bird.jpg");
		Mat frame;
		cap.read(frame);
		//根据需求决定是不是重置图像
		//resize(frame, frame, Size(300, 300));

		start = clock();
		// Create a 4D blob from a frame. 创建神经网络输入图像
		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network 设置输出
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers 获取输出层结果
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);
		finish = clock();

		cout << "time is " << double(finish - start) / CLOCKS_PER_SEC << endl;
		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		//输出前向传播的时间
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		imshow("result", frame);
	}
	
	//保存图像
	//imwrite("result.jpg", frame);
	waitKey(0);

	return 0;
}

/**
 * @brief Remove the bounding boxes with low confidence using non-maxima suppression 基于非极大性抑制去除边框
 *
 * @param frame 视频图像
 * @param outs 输出层结果
 */
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	//输出类
	vector<int> classIds;
	//置信度
	vector<float> confidences;
	vector<Rect> boxes;

	//遍历所有的输出层
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		//扫描所有来自网络的边界框输出，只保留具有高置信度分数的边界框。将框的类标签指定为框得分最高的类。
		//读取框
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score 获取置信度和位置参数
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			//如果大于置信度阈值
			if (confidence > confThreshold)
			{
				//获取坐标
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	//输出非极大性抑制结果，按置信度从大到小输出
	vector<int> indices;
	//非极大性抑制
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	//绘图
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		//类，置信度
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

/**
 * @brief Draw the predicted bounding box 画框
 *
 * @param classId 类别
 * @param conf 置信度
 * @param left
 * @param top
 * @param right
 * @param bottom
 * @param frame
 */
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box 在每个框左上角标上标签
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers 获取输出层
/**
 * @brief Get the Outputs Names object
 *
 * @param net
 * @return vector<String>
 */
vector<String> getOutputsNames(const Net& net)
{
	//输出
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
		{
			names[i] = layersNames[outLayers[i] - 1];
		}
	}
	return names;
}