#pragma comment(lib, "./tensorflow.lib")
#include "03_cc_unet_interface.h"

// public --------------------------------------------------
// -------------------------���캯��-------------------------
UnetInterface::UnetInterface(string model_path)
{
	// ����ģ�͵���������ڵ�����
	inp_tensor_name = "input_1:0";
	out_tensor_name = "conv2d_24/Sigmoid:0";

	// ����ģ�͵�����ͼ
	Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!status_load.ok()) {
		cout << "ERROR: Loading model failed..." << endl;
		cout << model_path << status_load.ToString() << "\n";
	}

	// �����Ự
	NewSession(SessionOptions(), &session);
	Status status_create = session->Create(graphdef);
	if (!status_create.ok()) {
		cout << "ERROR: Creating graph in session failed.." << status_create.ToString() << endl;
	}
	else {
		cout << "----------- Successfully created session and load graph -------------" << endl;
	}

}

// -------------------------ģ�ͼ���-------------------------
int UnetInterface::predict(string img_path, Mat& dstImage)
{
	// ��ȡͼ�� -> ����ͼ��
	Mat srcImage = imread(img_path, 0);  // �ԻҶ�ͼģʽ��
	if (srcImage.empty()) {  // У���Ƿ������򿪴�����ͼ��!
		cout << "can't open the image!!!!!!!" << endl;
		return -1;
	}
	imgPreprocessing(srcImage);  // Ԥ����
	// ����ͼ�� -> ��������
	convertCVMatToTensor();
	// �������� -> �������
	Status status_run = session->Run({ {this->inp_tensor_name, inpTensor} }, { out_tensor_name }, {}, &outTensor);
	if (!status_run.ok()) {
		cout << "ERROR: RUN failed..." << std::endl;
		cout << status_run.ToString() << "\n";
		return -1;
	}
	// ������� -> ���ͼ��
	convertTensorToCVMat();
	outImage.copyTo(dstImage);
}

// private ------------------------------------------------
// -------------------------ͼ��ߴ�-------------------------
void UnetInterface::imgPreprocessing(Mat& srcImage)
{
	resize(srcImage, inpImage, cv::Size(img_side, img_side));  // �ߴ�����
}

// -------------------------��ͼ�񿽱�������Tensor------------
void UnetInterface::convertCVMatToTensor()
{
	inpImage.copyTo(inpFloatMat);  // ����
	inpFloatMat.convertTo(inpFloatMat, CV_32FC1);  // ����ת��
	inpFloatMat = inpFloatMat / 255;  // ��ֵ��һ��

	float *p = (&inpTensor)->flat<float>().data();  //����ָ��inpTensor���ݵ�ָ��
	cv::Mat tensorMat(img_side, img_side, CV_32FC1, p);  // ����һ����inpTensor��ַ�󶨵�tensorMat
	inpFloatMat.convertTo(tensorMat, CV_32FC1);  // ͨ������inpFloatMat��tensorMat�����inpTensor
}

// -------------------------�����Tensorת����ͼ��------------
void UnetInterface::convertTensorToCVMat()
{
	Tensor *p = &outTensor[0];  // ����ָ��outTensor��ָ��
	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& result = p->flat<float>();
	const long count = result.size();  // �������ֽ�������ӡ��Ϣ
	cout << "The outTensor's size: " << result.size() << "\n" << endl;
	// ������ֵ
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
	Mat dstImage;  // �������ͼ��
	unet.predict("./data/0.png", dstImage);
	imwrite("./data/0.jpg", dstImage);
	return 0;
}
