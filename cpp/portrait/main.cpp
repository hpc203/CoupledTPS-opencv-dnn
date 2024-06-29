#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include"tps2flow.h"
#include "upsample.h"
#include "grid_sample.h"
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace std;
using namespace dnn;


class CoupledTPS_PortraitNet
{
public:
	CoupledTPS_PortraitNet(string modelpatha, string modelpathb);
	Mat detect(const Mat srcimg, const int iter_num);
private:
	const int input_height = 384;
	const int input_width = 512;
	const int grid_h = 7;
	const int grid_w = 10;
	Mat rigid_mesh;
	Mat grid;
	Mat W_inv;
	
	Net feature_extractor;
    Net regressNet;
};

CoupledTPS_PortraitNet::CoupledTPS_PortraitNet(string modelpatha, string modelpathb)
{
	this->feature_extractor = readNet(modelpatha);
    this->regressNet = readNet(modelpathb);
    
	get_norm_rigid_mesh_inv_grid(this->rigid_mesh, this->grid, this->W_inv, this->input_height, this->input_width, this->grid_h, this->grid_w);
}

Mat CoupledTPS_PortraitNet::detect(const Mat srcimg, const int iter_num)
{
	const float scale_x = (float)srcimg.cols / this->input_width;
	const float scale_y = (float)srcimg.rows / this->input_height;
	Mat input = srcimg.clone();
	input.convertTo(input, CV_32FC3, 1.0 / 255.0, -1.0);

	Mat img;
	resize(srcimg, img, Size(this->input_width, this->input_height), INTER_LINEAR);
	img.convertTo(img, CV_32FC3, 1.0 / 127.5, -1.0);
	Mat blob = blobFromImage(img);

	this->feature_extractor.setInput(blob);
	vector<Mat> feature_oris;
	this->feature_extractor.forward(feature_oris, this->feature_extractor.getUnconnectedOutLayersNames());   
    Mat feature = feature_oris[0].clone();

	int shape[4] = {1, 2, this->input_height, this->input_width};
	Mat flow = cv::Mat::zeros(4, shape, CV_32FC1); 
	vector<Mat> flow_list;
    for(int i=0;i<iter_num;i++)
    {
        this->regressNet.setInput(feature);
        vector<Mat> mesh_motions;
        this->regressNet.forward(mesh_motions, this->regressNet.getUnconnectedOutLayersNames());

		const float* offset = (float*)mesh_motions[0].data;
		Mat tp;
		get_ori_rigid_mesh_tp(this->rigid_mesh, tp, offset, this->input_height, this->input_width);
	
		Mat T = W_inv * tp;   ////_solve_system
		T = T.t();    ////舍弃batchsize

		Mat T_g = T * this->grid;   
		Mat delta_flow;
		_transform(T_g, this->grid, this->input_height, this->input_width, delta_flow);

		if(i==0)
		{
			flow += delta_flow;
		}
		else
		{	
			Mat warped_flow;
			warp_with_flow(flow, delta_flow, warped_flow);
			flow = delta_flow + warped_flow;
		}
		flow_list.emplace_back(flow.clone());

		if(i<(iter_num-1))
		{
			const int fea_h = feature.size[2];
			const int fea_w = feature.size[3];
			const float scale_h = (float)fea_h/flow.size[2];
			const float scale_w = (float)fea_w/flow.size[3];
			std::optional<double> scales_h = std::optional<double>(scale_h);
			std::optional<double> scales_w = std::optional<double>(scale_w);
			Mat down_flow;
			UpSamplingBilinear(flow, down_flow, fea_h, fea_w, true, scales_h, scales_w);

			for(int h=0;h<fea_h;h++)
			{
				for(int w=0;w<fea_w;w++)
				{
					down_flow.ptr<float>(0, 0, h)[w] *= scale_w;
					down_flow.ptr<float>(0, 1, h)[w] *= scale_h;
				}
			}
			feature.release();
			warp_with_flow(feature_oris[0], down_flow, feature);
		}
    }
	
	Mat flow_x, flow_y;
	split2xy(flow_list[iter_num-1], flow_x, flow_y);
	Mat predflow_x;
	resize(flow_x, predflow_x, Size(srcimg.cols, srcimg.rows));
	predflow_x *= scale_x;
	Mat predflow_y;
	resize(flow_y, predflow_y, Size(srcimg.cols, srcimg.rows));
	predflow_y *= scale_y;

	Mat pred_out = flow_mesh(predflow_x, predflow_y, input, srcimg.rows, srcimg.cols);
	return pred_out;
}


int main()
{
	CoupledTPS_PortraitNet mynet("weights/portrait/feature_extractor.onnx", "weights/portrait/regressnet.onnx");
    const int iter_num = 1;
	string imgpath = "testimgs/portrait/0083_mi9.jpg";
	Mat srcimg = imread(imgpath);

	Mat pred_out = mynet.detect(srcimg, iter_num);
	///如果预测图跟输入原图的高宽不一致，请reszie到一致
	Mat combine_img;
	if (srcimg.rows >= srcimg.cols)
	{
		copyMakeBorder(pred_out, pred_out, 0, 0, 20, 0, BORDER_CONSTANT, Scalar(255, 255, 255));
		hconcat(srcimg, pred_out, combine_img);
	}
	else
	{
		copyMakeBorder(pred_out, pred_out, 20, 0, 0, 0, BORDER_CONSTANT, Scalar(255, 255, 255));
		vconcat(srcimg, pred_out, combine_img);
	}
	///imwrite("cpp-opencv-dnn-CoupledTPS_Portrait.jpg", combine_img);
	namedWindow("opencv-dnn-CoupledTPS_Portrait", WINDOW_NORMAL);
	imshow("opencv-dnn-CoupledTPS_Portrait", combine_img);

	// namedWindow("srcimg", WINDOW_NORMAL);
	// imshow("srcimg", srcimg);
	// namedWindow("pred_out", WINDOW_NORMAL);
	// imshow("pred_out", pred_out);
	waitKey(0);
	destroyAllWindows();
}