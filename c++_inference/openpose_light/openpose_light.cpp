// r_lstm_cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <assert.h>
#include <onnxruntime_c_api.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;

using namespace chrono;

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const int HEIGHT = 320;
const int WIDTH = 320;

// size of paf 19x2 (x and y)
const int OUTPUT_PAF = 19;
// size of heatmap 18 joints
const int OUTPUT_HEATMAP = 18;
const int BLANK_INDEX = 0;


//*****************************************************************************
// recognition class
class OnnxRecognition
{
private:
	OrtSession* session;
	OrtEnv* env;
	OrtSessionOptions* session_options;

#ifdef _WIN32
	const wchar_t* model_path = L"D:/c++/openpose_light/_models/bodypose_light.onnx";
#else
	const char* model_path = "D:/c++/openpose_light/_models/bodypose_light.onnx";
#endif

public:
	OnnxRecognition() {
		// create env
		CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

		// initialize session options if needed
		CheckStatus(g_ort->CreateSessionOptions(&session_options));
		g_ort->SetIntraOpNumThreads(session_options, 50);
		g_ort->SetInterOpNumThreads(session_options, 50);

		// Sets graph optimization level
		g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
		CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));
		printf("Using Onnxruntime C API\n");
	}

	~OnnxRecognition() {
		g_ort->ReleaseSession(session);
		g_ort->ReleaseSessionOptions(session_options);
		g_ort->ReleaseEnv(env);
	}

	void CheckStatus(OrtStatus* status)
	{
		if (status != NULL) {
			const char* msg = g_ort->GetErrorMessage(status);
			fprintf(stderr, "%s\n", msg);
			g_ort->ReleaseStatus(status);
		}
	}


	void detectPose(vector<float> img) 
	/// <summary>
	/// get image -> export the 
	/// </summary>
	/// <param name="img"></param>
	/// <returns></returns>
	{
		size_t input_tensor_size = HEIGHT * WIDTH * 3;
		int64_t input_node_dims[] = { 1, 3, HEIGHT, WIDTH};
		std::vector<const char*> output_node_names = { "pafs", "heatmaps"};
		std::vector<const char*> input_node_names = { "data" };

		// create input tensor object from data values
		OrtMemoryInfo* memory_info;
		CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
		OrtValue* input_tensor = NULL;
		CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, img.data(), input_tensor_size * sizeof(float), input_node_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
		int is_tensor;
		g_ort->IsTensor(input_tensor, &is_tensor);
		assert(is_tensor);
		g_ort->ReleaseMemoryInfo(memory_info);

		// score model & input tensor, get back output tensor
		OrtValue* output_tensor [2];
		g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 2, &output_tensor);


		float(*output_paf)[OUTPUT_PAF*2][HEIGHT][WIDTH];
		float(*output_heatmap)[OUTPUT_HEATMAP * 2][HEIGHT][WIDTH];
		g_ort->GetTensorMutableData(output_tensor[0], (void**)&output_paf);
		g_ort->GetTensorMutableData(output_tensor[1], (void**)&output_heatmap);
		
		// get export 2 output here 



		// release mem for tensor
		g_ort->ReleaseValue(input_tensor);
		g_ort->ReleaseValue(output_tensor[0]);
		g_ort->ReleaseValue(output_tensor[1]);

		// TODO: remember to release mem for output paf and heatmap
	}
};

//*****************************************************************************
// helper function to check for status


// function for resize and padding
cv::Mat GetFitRatioImage(const cv::Mat& img)
/// <summary>
/// change mat image -> 320 x 320
/// </summary>
/// <param name="img"></param>
/// <returns></returns>
{
	int width = img.cols,
		height = img.rows;
	float scale = 0.0;

	if (width > height) {
		scale = float(WIDTH) / width;
		width = WIDTH;
		height = (int)(height * scale);
	}
	else {
		scale = float(HEIGHT) / height;
		width = (int)(width * scale);
		height = HEIGHT;
	}

	cv::Rect roi;

	// resize image -> 320x320
	roi.y = 0;
	roi.x = 0;
	roi.height = height;
	roi.width = width;

	cv::Mat square = cv::Mat::zeros(HEIGHT, WIDTH, img.type());
	cv::Mat resized_img;
	cv::resize(img, resized_img, roi.size());


	// paste to image
	roi.x = (int)((WIDTH - width) / 2);
	roi.y = (int)((HEIGHT - height) / 2);
	resized_img.copyTo(square(roi));


	return square;
}

vector<float> GetFloatVectorFromMat(cv::Mat& img) {

	Mat chans[3];
	split(img, chans);
	vector<uchar> V;

	V.assign(chans[0].data, chans[0].data + chans[0].total() * chans[0].channels());
	V.insert(V.end(), chans[1].data, chans[1].data + chans[1].total() * chans[1].channels());
	V.insert(V.end(), chans[2].data, chans[2].data + chans[2].total() * chans[2].channels());


	vector<float> result(V.begin(), V.end());

	for (int i = 0; i < result.size();i++) {
		result[i] = (float)result[i] / 255;
		result[i] = result[i] - 0.5f;
	}

	return result;
}

int main(int argc, char* argv[]) {
	// create recognizor
	OnnxRecognition* recognizor = new OnnxRecognition();

	// read image 320 x 320
	Mat image;
	image = imread("D:/ocr_code/r_lstm/data/1.jpg");   // Read the file
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	image = GetFitRatioImage(image);

	// start to count time 
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	recognizor->detectPose(GetFloatVectorFromMat(image));
	
	// finish to count time
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	std::cout << "It took me " << time_span.count() << " seconds.";

	// release mem for the model.
	delete recognizor;

	return 0;
}