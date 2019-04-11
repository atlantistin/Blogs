#pragma comment(lib, "./tensorflow.lib")
#include "03_cc_unet_interface.h"

// public --------------------------------------------------
// -------------------------构造函数-------------------------
UnetInterface::UnetInterface(string model_path)
{
	// 配置模型的输入输出节点名字
	inp_tensor_name = "input_1:0";
	out_tensor_name = "conv2d_24/Sigmoid:0";

	// 加载模型到计算图
	Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!status_load.ok()) {
		cout << "ERROR: Loading model failed..." << endl;
		cout << model_path << status_load.ToString() << "\n";
	}

	// 创建会话
	NewSession(SessionOptions(), &session);
	Status status_create = session->Create(graphdef);
	if (!status_create.ok()) {
		cout << "ERROR: Creating graph in session failed.." << status_create.ToString() << endl;
	}
	else {
		cout << "----------- Successfully created session and load graph -------------" << endl;
	}

}

// -------------------------模型计算-------------------------
int UnetInterface::predict(string img_path, Mat& dstImage)
{
	// 读取图像 -> 输入图像
	Mat srcImage = imread(img_path, 0);  // 以灰度图模式打开
	if (srcImage.empty()) {  // 校验是否正常打开待操作图像!
		cout << "can't open the image!!!!!!!" << endl;
		return -1;
	}
	imgPreprocessing(srcImage);  // 预处理
	// 输入图像 -> 输入张量
	convertCVMatToTensor();
	// 输入张量 -> 输出张量
	Status status_run = session->Run({ {this->inp_tensor_name, inpTensor} }, { out_tensor_name }, {}, &outTensor);
	if (!status_run.ok()) {
		cout << "ERROR: RUN failed..." << std::endl;
		cout << status_run.ToString() << "\n";
		return -1;
	}
	// 输出张量 -> 输出图像
	convertTensorToCVMat();
	outImage.copyTo(dstImage);
}

// private ------------------------------------------------
// -------------------------图像尺寸-------------------------
void UnetInterface::imgPreprocessing(Mat& srcImage)
{
	resize(srcImage, inpImage, cv::Size(img_side, img_side));  // 尺寸缩放
}

// -------------------------将图像拷贝到输入Tensor------------
void UnetInterface::convertCVMatToTensor()
{
	inpImage.copyTo(inpFloatMat);  // 拷贝
	inpFloatMat.convertTo(inpFloatMat, CV_32FC1);  // 类型转化
	inpFloatMat = inpFloatMat / 255;  // 数值归一化

	float *p = (&inpTensor)->flat<float>().data();  //创建指向inpTensor内容的指针
	cv::Mat tensorMat(img_side, img_side, CV_32FC1, p);  // 创建一个与inpTensor地址绑定的tensorMat
	inpFloatMat.convertTo(tensorMat, CV_32FC1);  // 通过拷贝inpFloatMat至tensorMat来填充inpTensor
}

// -------------------------将输出Tensor转化回图像------------
void UnetInterface::convertTensorToCVMat()
{
	Tensor *p = &outTensor[0];  // 创建指向outTensor的指针
	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& result = p->flat<float>();
	const long count = result.size();  // 计算总字节数并打印信息
	cout << "The outTensor's size: " << result.size() << "\n" << endl;
	// 迭代赋值
	Mat_<uchar>::iterator iter = outImage.begin<uchar>();
	Mat_<uchar>::iterator over = outImage.end<uchar>();
	for (int i = 0; iter != over; iter++) {
		float tmp = result(i);
		*iter = (int)(tmp * 255);
		i++;
	}
}

/*
--------------------------------------------------------------------------------------------------------------------
*/

int main(int aargc, char** argv)
{
	UnetInterface unet = UnetInterface("./data/unet_membrane.pb");
	Mat dstImage;  // 定义输出图像
	unet.predict("./data/0.png", dstImage);
	imwrite("./data/0.jpg", dstImage);
	return 0;
}
