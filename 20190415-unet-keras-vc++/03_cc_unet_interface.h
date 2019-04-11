#pragma once
#define COMPILER_MSVC
#define NOMINMAX
#include <fstream>
#include <utility>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"

#include "opencv2/opencv.hpp"

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using namespace tensorflow;
using namespace tensorflow::ops;

using namespace std;
using namespace cv;

// 定义UnetInterface类
class UnetInterface
{
public:
	/*
	---------------------变量---------------------
	*/
	UnetInterface(string model_path);
	/*
	---------------------方法---------------------
	*/
	int predict(string img_path, Mat& dstImage);

private:
	/*
	---------------------变量---------------------
	*/
	string inp_tensor_name;  // 输入节点名字
	string out_tensor_name;  // 输出节点名字

	Session* session;  // 定义session
	GraphDef graphdef;  // 定义graph

	// 定义输入输入图像
	int img_side = 256;  // 输入尺寸
	Mat inpImage = Mat(img_side, img_side, CV_8UC1);
	Mat outImage = Mat(img_side, img_side, CV_8UC1);

	// 定义输入输出张量
	cv::Mat inpFloatMat;  // inpImage -> inpFloatMat
	Tensor inpTensor = Tensor(DT_FLOAT, TensorShape({ 1, img_side, img_side, 1 }));  // 输入张量
	vector<tensorflow::Tensor> outTensor;

	/*
	---------------------方法---------------------
	*/
	void imgPreprocessing(Mat& src);
	void convertCVMatToTensor();
	void convertTensorToCVMat();
};
