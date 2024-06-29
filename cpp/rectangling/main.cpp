#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <string>
#include <math.h>
#include "tps2flow.h"
#include "upsample.h"
#include "grid_sample.h"
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace std;
using namespace dnn;


class CoupledTPS_RectanglingNet
{
public:
	CoupledTPS_RectanglingNet(string modelpatha, string modelpathb);
	Mat detect(const Mat srcimg, const Mat mask_img, const int iter_num);
private:
	const int input_height = 384;
	const int input_width = 512;
	const int grid_h = 6;
	const int grid_w = 8;
	Mat grid;
	Mat W_inv;
	
	Net feature_extractor;
    Net regressNet;
};

CoupledTPS_RectanglingNet::CoupledTPS_RectanglingNet(string modelpatha, string modelpathb)
{
	this->feature_extractor = readNet(modelpatha);
    this->regressNet = readNet(modelpathb);
    
	get_norm_rigid_mesh_inv_grid(this->grid, this->W_inv, this->input_height, this->input_width, this->grid_h, this->grid_w);
}

Mat CoupledTPS_RectanglingNet::detect(const Mat srcimg, const Mat mask_img, const int iter_num)
{
	Mat img;
	resize(srcimg, img, Size(this->input_width, this->input_height), INTER_LINEAR);
	img.convertTo(img, CV_32FC3, 1.0 / 127.5, -1.0);
	Mat input_tensor = blobFromImage(img);
	
	Mat mask;
	resize(mask_img, mask, Size(this->input_width, this->input_height), INTER_LINEAR);
	mask.convertTo(mask, CV_32FC3, 1.0 / 255.0);
	Mat mask_tensor = blobFromImage(mask);

	this->feature_extractor.setInput(input_tensor, "inputa");
	this->feature_extractor.setInput(mask_tensor, "inputb");

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
		get_ori_rigid_mesh_tp(tp, offset, this->input_height, this->input_width, this->grid_h, this->grid_w);
		
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
	Mat correction_final;
	warp_with_flow(input_tensor, flow_list[iter_num-1], correction_final);

	Mat correction_img = convert4dtoimage(correction_final);
	return correction_img;
}



int main()
{
	CoupledTPS_RectanglingNet mynet("weights/rectangling/feature_extractor.onnx", "weights/rectangling/regressnet.onnx");
    const int iter_num = 3;
	string imgpath = "testimgs/rectangling/input/00474.jpg";
	string maskpath = "testimgs/rectangling/mask/00474.jpg";
	Mat srcimg = imread(imgpath);
	Mat mask_img = imread(maskpath);

	Mat pred_out = mynet.detect(srcimg, mask_img, iter_num);
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
	///imwrite("cpp-opencv-dnn-CoupledTPS_Rectangling.jpg", combine_img);
	namedWindow("opencv-dnn-CoupledTPS_Rectangling", WINDOW_NORMAL);
	imshow("opencv-dnn-CoupledTPS_Rectangling", combine_img);

	// namedWindow("srcimg", WINDOW_NORMAL);
	// imshow("srcimg", srcimg);
	// namedWindow("pred_out", WINDOW_NORMAL);
	// imshow("pred_out", pred_out);
	waitKey(0);
	destroyAllWindows();
}