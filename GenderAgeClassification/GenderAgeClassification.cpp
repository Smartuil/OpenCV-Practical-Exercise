#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iterator>
using namespace cv;
using namespace cv::dnn;
using namespace std;

/**
 * @brief Get the Face Box object 人脸定位
 *
 * @param net 人脸检测网络
 * @param frame 检测图像
 * @param conf_threshold 阈值
 * @return tuple<Mat, vector<vector<int>>> 元组容器，可返回多个值
 */
tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat &frame, double conf_threshold)
{
	//图像复制
	Mat frameOpenCVDNN = frame.clone();
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;
	//缩放尺寸
	double inScaleFactor = 1.0;
	//检测图大小
	Size size = Size(300, 300);
	// std::vector<int> meanVal = {104, 117, 123};
	Scalar meanVal = Scalar(104, 117, 123);

	cv::Mat inputBlob;
	inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);
	net.setInput(inputBlob, "data");
	//四维矩阵输出
	cv::Mat detection = net.forward("detection_out");
	//提取结果信息
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	vector<vector<int>> bboxes;

	for (int i = 0; i < detectionMat.rows; i++)
	{
		//预测概率
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > conf_threshold)
		{
			//左上角点，坐标被归一化
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			//右下角角点，坐标被归一化
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
			vector<int> box = { x1, y1, x2, y2 };
			//人脸坐标
			bboxes.push_back(box);
			//图像框选
			cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
		}
	}

	return make_tuple(frameOpenCVDNN, bboxes);
}

int main(void)
{
	//人脸模型
	string faceProto = "model/opencv_face_detector.pbtxt";
	string faceModel = "model/opencv_face_detector_uint8.pb";

	//年龄模型
	string ageProto = "model/age_deploy.prototxt";
	string ageModel = "model/age_net.caffemodel";

	//性别模型
	string genderProto = "model/gender_deploy.prototxt";
	string genderModel = "model/gender_net.caffemodel";

	//均值
	Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

	//年龄段标签
	vector<string> ageList = { "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)" };

	//性别标签
	vector<string> genderList = { "Male", "Female" };

	//导入网络
	Net ageNet = cv::dnn::readNet(ageProto, ageModel);
	Net genderNet = cv::dnn::readNet(genderProto, genderModel);
	Net faceNet = cv::dnn::readNetFromTensorflow(faceModel, faceProto);

	//打开摄像头
	VideoCapture cap;
	cap.open(0);
	if (cap.isOpened())
	{
		cout << "camera is opened!" << endl;
	}
	else
	{
		return 0;
	}

	int padding = 20;
	while (waitKey(1) < 0)
	{
		// read frame 读图
		Mat frame;
		cap.read(frame);
		if (frame.empty())
		{
			waitKey();
			break;
		}
		frame = imread("./images/couple1.jpg");
		//人脸坐标
		vector<vector<int>> bboxes;
		//人脸检测结果图
		Mat frameFace;
		//人脸定位
		//tie()函数解包frameFace和bboxes
		tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);
		//人脸判断
		if (bboxes.size() == 0)
		{
			cout << "No face detected, checking next frame." << endl;
			continue;
		}
		//逐个提取人脸检测
		for (auto it = begin(bboxes); it != end(bboxes); ++it)
		{
			//框选人脸
			Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2 * padding, it->at(3) - it->at(1) + 2 * padding);
			//避免人脸框选超过图像边缘
			rec.width = ((rec.x + rec.width) > frame.cols) ? (frame.cols - rec.x - 1) : rec.width;
			rec.height = ((rec.y + rec.height) > frame.rows) ? (frame.rows - rec.y - 1) : rec.height;

			// take the ROI of box on the frame,原图中提取人脸
			Mat face = frame(rec);

			//性别检测
			Mat blob;
			blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
			genderNet.setInput(blob);
			// string gender_preds; 获取前向传播softmax结果
			vector<float> genderPreds = genderNet.forward();
			// find max element index max_element用于找寻最大值
			// distance function does the argmax() work in C++ distance返回最大值和第一个值下标的距离
			int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
			//获得检测结果
			string gender = genderList[max_index_gender];
			cout << "Gender: " << gender << endl;

			//年龄识别
			ageNet.setInput(blob);
			vector<float> agePreds = ageNet.forward();
			// finding maximum indicd in the age_preds vector 找到年龄预测最大下表
			int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
			string age = ageList[max_indice_age];
			cout << "Age: " << age << endl;

			// label 输出标签
			string label = gender + ", " + age;
			//在人脸定位图上显示结果
			cv::putText(frameFace, label, Point(it->at(0), it->at(1) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
		}
		//保存结果
		imshow("Frame", frameFace);
		imwrite("out.jpg", frameFace);
	}
}