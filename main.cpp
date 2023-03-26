#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;	//缩放系数，用于将图像的每个通道乘以一个值
const Scalar meanVal(104.0, 177.0, 123.0);	//消噪均值
const float confidenceThreshold = 0.7;

//考勤记录结构体
struct Attendance_Record {
	int Seriel_Number;	//studentID
	int This_Check_In;
	int Week_Check_In;
	int Month_Check_In;
	int Semester_Check_In;
};

//创建一个 Attendance_Record 类型的 一维vector ：record ，用于存放所有人考勤统计信息。只存数据，不存说明
vector<Attendance_Record> Record;
// 表示保存的人脸照片编号
int face_serial_number;
void admin();
void Read_Attendance_Record_From_CSV();
void Save_Attendance_Record_To_CSV();
Mat cut_face(Mat, Rect);
void Take_Save_A_Face_Photo(string);
void face_detect_dnn();
bool face_match(string);

int main()
{
	Read_Attendance_Record_From_CSV();
	face_serial_number = Record.size() - 1;	//去掉第一行的说明
	int sel = 0;
	do
	{
		std::cout << "*******************\n";
		std::cout << "   1： 注册\n";
		std::cout << "   2： 开始考勤\n";
		std::cout << "   3： 进入管理员界面\n";
		std::cout << "   其他： 退出\n";
		std::cout << "********************\n";

		cin >> sel;
		switch (sel)
		{
		case 1:	face_detect_dnn(); break;	//第一部分：用户注册
		case 2:			//第二部分：考勤+考勤情况统计
		{
			int tmp_count = face_serial_number;
			bool judge = 0;
			std::cout << endl << "按 Enter 正式开始" << endl;
			waitKey(13);
			Take_Save_A_Face_Photo("face_tmp");
			while ((tmp_count--) && judge == 0)
			{
				judge = face_match(std::to_string(tmp_count));
			}
		}
		case 3:	admin();		break;		//第三部分：管理员模式，预计实现一定时间范围内的考勤情况查看与手动数据导出
		default:	std::cout << "Bye!\n";
		};
	} while (sel < 4 && sel > 0);
	return 0;
}

//实现一定时间范围内的考勤情况查看与手动数据导出
void admin()	
{
	int sel = 0;
	do
	{
		std::cout << "*******************\n";
		std::cout << "   1： 查看本周考勤情况\n";
		std::cout << "   2： 查看本月考勤情况\n";
		std::cout << "   3： 查看本学期考勤情况\n";
		std::cout << "   4： 重置今日考勤记录\n";
		std::cout << "   其他： 退出\n";
		std::cout << "********************\n";
		cin >> sel;
		switch (sel)
		{
		case 1:		//查看本周考勤情况
		{
			std::cout << "本周考勤情况如下：\n";
			std::cout << "姓名\t学号\t本周考勤次数\n";
			for (int i = 0; i < Record.size(); i++)
			{
				std::cout << Record[i].Seriel_Number << "\t" << Record[i].Week_Check_In << endl;
			}
		}break;

		case 2:		//查看本月考勤情况
		{
			std::cout << "本月考勤情况如下：\n";
			std::cout << "姓名\t学号\t本月考勤次数\n";
			for (int i = 0; i < Record.size(); i++)
			{
				std::cout << Record[i].Seriel_Number << "\t" << Record[i].Month_Check_In << endl;
			}
		}break;
		case 3:		//查看本学期考勤情况
		{
			std::cout << "本学期考勤情况如下：\n";
			std::cout << "姓名\t学号\t本月考勤次数\n";
			for (int i = 0; i < Record.size(); i++)
			{
				std::cout << Record[i].Seriel_Number << "\t" << Record[i].Semester_Check_In << endl;
			}
		}break;
		case 4:		//重置今日考勤记录
		{
			for (auto row : Record)
			{
				row.This_Check_In = 0;
				std::cout << "已重置！";
			}
		}
		}
	} while (sel < 5 && sel > 0);
}

//从csv文件中将已保存的考勤情况一行一行地读入到 Record 中。注意，在csv文件中，第一行是对应列的内容声明，如 Seriel_Number,This_Check_In,Week_Check_In,Month_Check_In,Semester_Check_In第二行开始才应该是正式的内容，对应着第一行的内容声明，如 1,0,0,0,0
void Read_Attendance_Record_From_CSV()
{
	std::ifstream infile("AttendanceRecord.csv");
	//没有就创建一个
	if (!infile.good()) {
		std::ofstream outfile("AttendanceRecord.csv");
		outfile.close();
		std::cout << "The file was not found, it has been created automatically." << std::endl;
		return;
	}
	//目标文件存在，正常读取
	else {
		std::string line;//用来存储从文件中读取的每一行
		getline(infile, line);	//跳过第一行
		while (getline(infile, line))	//从输入文件流中读取每一行数据
		{
			std::stringstream sep(line);	//sep用来处理当前行
			std::string field;		//存储当前字段的值
			Attendance_Record tmp;
			while (getline(sep, field, ','))
			{
				Attendance_Record tmp_record;
				tmp_record.Seriel_Number = atoi(field.c_str());
				getline(sep, field, ',');
				tmp_record.This_Check_In = atoi(field.c_str());
				getline(sep, field, ',');
				tmp_record.Week_Check_In = atoi(field.c_str());
				getline(sep, field, ',');
				tmp_record.Month_Check_In = atoi(field.c_str());
				getline(sep, field, ',');
				tmp_record.Semester_Check_In = atoi(field.c_str());
				tmp = tmp_record;
			}
			Record.push_back(tmp);
		}
		infile.close();
	}
}

//将Record的每一维都保存到AttendanceRecord.csv中,注意，在csv文件中，第一行是对应列的内容声明，如 Seriel_Number,This_Check_In,Week_Check_In,Month_Check_In,Semester_Check_In第二行开始才应该是正式的内容，对应着第一行的内容声明，如 1,0,0,0,0
void Save_Attendance_Record_To_CSV()
{
	std::ofstream out("AttendanceRecord.csv");
	out << "Seriel_Number" << "," << "This_Check_In" << ", " << "Week_Check_In" << "," << "Month_Check_In" << ", " << "Semester_Check_In" << "\n";
	int tmp_count = 0;
	for (const auto row : Record)
	{
		out << row.Seriel_Number << ','
			<< row.This_Check_In << ','
			<< row.Week_Check_In << ','
			<< row.Month_Check_In << ','
			<< row.Semester_Check_In;
		out << '\n';
	}
	out.close();
}

Mat cut_face(Mat img1, Rect rect)
{
	Mat img2, bg, fg;
	grabCut(img1, img2, rect, bg, fg, 1, GC_INIT_WITH_RECT);
	compare(img2, GC_PR_FGD, img2, CMP_EQ);

	Mat img3(img1.size(), CV_8UC3, Scalar(255, 255, 255));
	img1.copyTo(img3, img2);
	return img3;

}

//实现帧获取、人脸检测、获取签到人照片
void Take_Save_A_Face_Photo(string library_PATH)
{
	// 打开默认摄像头
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cerr << "无法打开摄像头\n";
		return;
	}
	cv::Mat frame;
	ostringstream ss;	//创建输出流
	namedWindow("preview", 0);
	resizeWindow("preview", Size(640, 480));

	int key = 0;
	while (key != 13)
	{
		// 读取一帧图像
		cap >> frame;

		// 检查图像是否为空
		if (frame.empty())
		{
			std::cerr << "无法读取图像\n";
			return;
		}
		// 人脸检测模型文件路径
		String det_onnx_path = "models/face_detection_yunet.onnx";
		//初始化人脸检测器 FaceDetectorYN
		Ptr<FaceDetectorYN> faceDetector;
		// 在这一帧中检测人脸
		faceDetector = FaceDetectorYN::create(det_onnx_path, "", frame.size(), 0.9f, 0.3f, 5000);
		Mat faces_1;
		faceDetector->detect(frame, faces_1);
		//如果在这一帧中未检测到人脸，则输出错误信息并继续循环
		if (faces_1.rows < 1)
		{
			std::cerr << "Cannot find a face in this frame!\n";
			continue;
		}

		//若图像不为空并且含有人脸，则显示图像
		ss << " Press enter To Save Your Photo With Face ";
		putText(frame, ss.str(), Point(5, 20), 0, 0.90, Scalar(35, 40, 133), 2, 10);
		imshow("preview", frame);
		//等待确定
		key = waitKey();
	}
	// 确定完成，保存图像
	destroyWindow("preview");
	cv::imwrite(library_PATH + "/tmp.jpg", frame);

	std::cout << "拍照成功，已保存\n";

}

// 进行人脸检测，使用dnn+TensorFlow实现人脸识别功能,实现了按照既定顺序进行人脸注册功能
//疑问:两个人的情况下为什么会优先考虑一个人而不是另一个人
void face_detect_dnn() 
{
	// 定义两个字符串变量，分别存储模型描述文件和二进制文件的路径
	//String modelDesc = "models/deploy.prototxt";
	//String modelBinary = "models/res10_300x300_ssd_iter_140000.caffemodel";
	String modelBinary = "models/opencv_face_detector_uint8.pb";
	String modelDesc = "models/opencv_face_detector.pbtxt";

	// 初始化网络，读入 包含模型权重的二进制文件 和 包含网络配置的文本文件（顺序不重要）
	//dnn::Net net = readNetFromCaffe(modelDesc, modelBinary);
	dnn::Net net = readNetFromTensorflow(modelBinary, modelDesc);

	// 设置后端为OpenCV
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	// 设置目标计算设备为CPU
	net.setPreferableTarget(DNN_TARGET_CPU);

	// 如果模型读取失败，网络为空，输出错误信息并结束
	if (net.empty())
	{
		std::cout << "无法载入网络...\n";
		return;
	}

	// 打开摄像头
	VideoCapture capture(0);
	//VideoCapture capture("2.mp4");	使用视频

	//如果打开摄像头失败，输出错误信息然后结束
	if (!capture.isOpened()) {
		std::cout << "无法载入摄像头...\n";
		return;
	}

	// 表示一帧图像
	Mat frame;
	// 表示循环次数
	int count = 0;
	long currframe = 20;
	capture.set(CAP_PROP_POS_FRAMES, currframe);

	while (capture.read(frame)) {		//读取一帧图像
		count++;
		int64 start = getTickCount();	//开始测量执行时间
		//如果当前帧为空，退出
		if (frame.empty())
		{
			break;
		}
		//waitKey(30);
		// 水平镜像调整
		// flip(frame, frame, 1);

		Mat frame0 = frame.clone();
		ostringstream ss0;	//创建输出流
		namedWindow("input_preview", 0);
		resizeWindow("input_preview", Size(640, 480));

		//声明是第几个录入的人脸
		ss0 << " Number of Cycles: " << count;
		putText(frame0, ss0.str(), Point(200, 20), 0, 0.90, Scalar(35, 40, 133), 2, 10);
		ss0.str("");		//清空字符串流
		ss0 << "Press backspace To Take ANOTHER Photo";
		putText(frame0, ss0.str(), Point(5, 50), 0, 0.85, Scalar(0, 0, 255), 2, 10);
		ss0.str("");		//清空字符串流
		ss0 << "Press other keys To Continue";
		putText(frame0, ss0.str(), Point(10, 80), 0, 0.85, Scalar(133, 0, 255), 2, 10);
		ss0.str("");		//清空字符串流
		imshow("input_preview", frame0);
		int key0 = waitKey();
		if (key0 != 8)		//照片没问题，继续
		{
			destroyWindow("input_preview");
		}
		if (key0 == 8)		//Backspace : 照片需更换
			continue;

		// 判断是否需要对图像进行调整
		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);

		// 将当前帧的数据转换为网络所需的blob格式
		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight), meanVal, false, false);

		//设置网络的输入，输入网络层名为 data
		net.setInput(inputBlob, "data");
		// 获取网络的输出，名为 detection_out
		Mat detection = net.forward("detection_out");

		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		// 计算得到本次模型推理所用时间
		double time = net.getPerfProfile(layersTimings) / freq;

		// 储存检测结果
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


		//遍历检测矩阵的每一行，并检查每一行的置信度值是否高于阈值。如果是，在检测到的人脸周围绘制一个矩形，并用其置信度值标记在矩形旁边
		ostringstream ss;	//创建输出流
		Rect tmprect;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			// 置信度 0～1之间
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				rectangle(frame, object, Scalar(0, 255, 0));
				tmprect = object;
				ss << confidence;
				String conf(ss.str());
				String label = "Face: " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), FILLED);
				putText(frame, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		float fps = getTickFrequency() / (getTickCount() - start);

		//int key = waitKey(1);

		//声明是第几个录入的人脸
		ss.str("");
		ss << "Seriel Number : " << ++face_serial_number;
		putText(frame, ss.str(), Point(200, 20), 0, 0.90, Scalar(35, 40, 133), 2, 10);
		ss.str("");		//清空字符串流
		//ss << "FPS: " << fps << " ; inference time: " << time << " ms.";
		ss << "PRESS enter TO save ; PRESS esc TO exit";
		putText(frame, ss.str(), Point(5, 50), 0, 0.85, Scalar(0, 0, 255), 2, 10);

		namedWindow("dnn_face_detection_result", 0);
		resizeWindow("dnn_face_detection_result", Size(640, 480));
		imshow("dnn_face_detection_result", frame);//将人脸检测的结果展示出来


		int key = waitKey();
		Mat face;
		char* save_path = "face_sign_library/face0";

		if (key == 13)	//Enter:拍照
		{
			char* facename = new char[strlen(save_path) + sizeof(face_serial_number) + 1];
			sprintf(facename, "%s%d.jpg", save_path, face_serial_number);
			face = cut_face(frame, tmprect);
			cv::imwrite(facename, face);
			destroyWindow("dnn_face_detection_result");

			//向 Record 添加第 face_serial_number 行，并将除serial_number以外的成员初始化为0
			Attendance_Record tmp_record;
			tmp_record.Seriel_Number = face_serial_number;
			tmp_record.This_Check_In = 0;
			tmp_record.Week_Check_In = 0;
			tmp_record.Month_Check_In = 0;
			tmp_record.Semester_Check_In = 0;
			Record.push_back(tmp_record);

			//向 AttendanceRecord.csv 中覆盖写入用户信息
			Save_Attendance_Record_To_CSV();

		}
		if (key == 27)	//ESC:退出
		{
			count--;
			destroyWindow("dnn_face_detection_result");
			break;
		}
	}
	capture.release();	//关闭摄像头
	std::cout << "录入人脸总数:" << count << endl;
}

bool face_match(string serial_number)
{	// 人脸检测模型文件路径
	String det_onnx_path = "models/face_detection_yunet.onnx";
	///人脸识别模型文件的路径
	String reg_onnx_path = "models/face_recognition_sface.onnx";

	string image1_path = "face_tmp/tmp.jpg";		//待对比的人脸
	string image2_path = "face_sign_library/face0" + serial_number + ".jpg";	//人脸库中的人脸

	Mat image1 = imread(image1_path);
	Mat image2 = imread(image2_path);

	//定义阈值
	float score_thresh = 0.9f;
	float nms_thresh = 0.3f;
	double cosine_similar_thresh = 0.363;
	double l2norm_similar_thresh = 1.128;
	int top_k = 5000;

	//初始化人脸检测器 FaceDetectorYN
	Ptr<FaceDetectorYN> faceDetector;


	// 在第一张照片中检测人脸
	faceDetector = FaceDetectorYN::create(det_onnx_path, "", image1.size(), score_thresh, nms_thresh, top_k);
	Mat faces_1;
	faceDetector->detect(image1, faces_1);
	/*
	//如果在第一张照片中未检测到人脸，则输出错误信息并退出函数
	if (faces_1.rows < 1)
	{
		std::cerr << "Cannot find a face in " << image1_path << "\n";
		return 0;
	}
	*/

	//在第二张照片中检测人脸
	faceDetector = FaceDetectorYN::create(det_onnx_path, "", image2.size(), score_thresh, nms_thresh, top_k);
	Mat faces_2;
	faceDetector->detect(image2, faces_2);
	// 如果在第二张照片中未检测到人脸，则输出错误信息并退出函数
	if (faces_2.rows < 1)
	{
		std::cerr << "Cannot find a face in  " << image2_path << "\n";
		return 0;
	}

	//初始化人脸识别器 FaceRecognizerSF
	Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");

	// 对两张照片中的人脸进行对齐和裁剪，并保存在变量 aligned_face1、aligned_face2 中
	Mat aligned_face1, aligned_face2;
	faceRecognizer->alignCrop(image1, faces_1.row(0), aligned_face1);
	faceRecognizer->alignCrop(image2, faces_2.row(0), aligned_face2);

	// 分别计算第一张和第二张照片中人脸的特征向量
	Mat feature1, feature2;
	faceRecognizer->feature(aligned_face1, feature1);
	feature1 = feature1.clone();
	faceRecognizer->feature(aligned_face2, feature2);
	feature2 = feature2.clone();

	double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
	double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);

	/*
	if (cos_score >= cosine_similar_thresh)
	{
		std::cout << "SAME IDENTITY!";
	}
	else
	{
		std::cout << "DIFFERENT!";
	}
	std::cout << " 余弦距离: " << cos_score << ", 阈值: " << cosine_similar_thresh << ".  （值越大，相似度越高，最大值为 1.0）\n";

	if (L2_score <= l2norm_similar_thresh)
	{
		std::cout << "SAME IDENTITY!";
	}
	else
	{
		std::cout << "DIFFERENT!";
	}
	std::cout << " L2范数距离: " << L2_score << ", 阈值: " << l2norm_similar_thresh << ". （值越小，相似度越高，最小值为 0.0）\n";

	*/

	std::cout << " 余弦距离: " << cos_score << ", 阈值: " << cosine_similar_thresh << ".  （值越大，相似度越高，最大值为 1.0）\n";
	std::cout << " L2范数距离: " << L2_score << ", 阈值: " << l2norm_similar_thresh << ". （值越小，相似度越高，最小值为 0.0）\n";

	//从库中找到匹配
	if (cos_score >= cosine_similar_thresh || L2_score <= l2norm_similar_thresh)
	{
		std::cout << "恭喜！考勤成功！" << endl;

		//将Record的第Serial_Number行的元素除学号外都自增1
		Record[stoi(serial_number)].This_Check_In += 1;
		Record[stoi(serial_number)].Week_Check_In += 1;
		Record[stoi(serial_number)].Month_Check_In += 1;
		Record[stoi(serial_number)].Semester_Check_In += 1;

		//向csv文件中覆盖写入数据
		Save_Attendance_Record_To_CSV();
		return 1;
	}
	//人脸不匹配
	else {
		if (stoi(serial_number) == 0)	std::cout << "考勤失败，请重试！";
		return 0;
	}
}