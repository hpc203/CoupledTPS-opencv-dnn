#include <iostream>
#include"tps2flow.h"

using namespace cv;
using namespace std;


void get_norm_rigid_mesh_inv_grid(Mat& grid, Mat& W_inv, const int input_height, const int input_width, const int grid_h, const int grid_w)
{
	float interval_x = input_width / grid_w;
	float interval_y = input_height / grid_h;
	const int h = grid_h + 1;
	const int w = grid_w + 1;
	const int length = h * w;
	Mat norm_rigid_mesh(length, 2, CV_32FC1);
	Mat W(length + 3, length + 3, CV_32FC1);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const int row_ind = i * w + j;
			const float x = (j * interval_x)*2.0 / float(input_width) - 1.0;
			const float y = (i * interval_y)*2.0 / float(input_height) - 1.0;
			
			W.at<float>(row_ind, 0) = 1;
			W.at<float>(row_ind, 1) = x;
			W.at<float>(row_ind, 2) = y;

			W.at<float>(length, 3 + row_ind) = 1;
			W.at<float>(length + 1, 3 + row_ind) = x;
			W.at<float>(length + 2, 3 + row_ind) = y;

			norm_rigid_mesh.at<float>(row_ind, 0) = x;
			norm_rigid_mesh.at<float>(row_ind, 1) = y;
		}
	}
	for (int i = 0; i < length; i++)
	{
		for (int j = 0;j < length; j++)
		{
			const float d2_ij = powf(W.at<float>(i, 0) - W.at<float>(j, 0), 2.0) + powf(W.at<float>(i, 1) - W.at<float>(j, 1), 2.0) + powf(W.at<float>(i, 2) - W.at<float>(j, 2), 2.0);
			W.at<float>(i, 3 + j) = d2_ij * logf(d2_ij + 1e-9);
		}
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			W.at<float>(length + i, j) = 0;
		}
	}

	W_inv = W.inv();

	interval_x = 2.0 / (input_width - 1);
	interval_y = 2.0 / (input_height - 1);
	const int grid_width = input_height * input_width;

	///Mat grid(length + 3, grid_width, CV_32FC1);
	grid.create(length + 3, grid_width, CV_32FC1);
	for (int i = 0; i < input_height; i++)
	{
		for (int j = 0; j < input_width; j++)
		{
			const float x = -1.0 + j * interval_x;
			const float y = -1.0 + i * interval_y;
			const int col_ind = i * input_width + j;
			grid.at<float>(0, col_ind) = 1;
			grid.at<float>(1, col_ind) = x;
			grid.at<float>(2, col_ind) = y;
		}
	}
	
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < grid_width; j++)
		{
			const float d2_ij = powf(norm_rigid_mesh.at<float>(i, 0) - grid.at<float>(1, j), 2.0) + powf(norm_rigid_mesh.at<float>(i, 1) - grid.at<float>(2, j), 2.0);
			grid.at<float>(3 + i, j) = d2_ij * logf(d2_ij + 1e-9);
		}
	}
	norm_rigid_mesh.release();
}

void get_ori_rigid_mesh_tp(Mat& tp, const float* offset, const int input_height, const int input_width, const int grid_h, const int grid_w)
{
	const float interval_x = input_width / grid_w;
	const float interval_y = input_height / grid_h;
	const int h = grid_h + 1;
	const int w = grid_w + 1;
	const int length = h * w;
	tp.create(length + 3, 2, CV_32FC1);
	
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const int row_ind = i * w + j;
			const float x = j * interval_x + offset[row_ind * 2];
			const float y = i * interval_y + offset[row_ind * 2 + 1];
			tp.at<float>(row_ind, 0) = x*2.0 / float(input_width) - 1.0;
			tp.at<float>(row_ind, 1) = y*2.0 / float(input_height) - 1.0;
		}

	}
	for (int i = 0; i < 3; i++)
	{
		tp.at<float>(length + i, 0) = 0;
		tp.at<float>(length + i, 1) = 0;
	}
}

void _transform(Mat T_g, Mat grid, const int h, const int w, Mat& flow)    ////T_g的形状是(2, 196608), grid的形状是(85, 196608)
{
	vector<int> out_dims = {1, 2, h, w};
	flow.create(out_dims, CV_32FC1);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const int col_ind = i * w + j;
			flow.ptr<float>(0, 0, i)[j] = (T_g.at<float>(0, col_ind) - grid.at<float>(1, col_ind)) * w*0.5;   /////对于四维Mat的索引，不能使用at函数，因为不支持超过三维的Mat, 大于四维的Mat，既不能使用at，也不能使用ptr访问元素
			flow.ptr<float>(0, 1, i)[j] = (T_g.at<float>(1, col_ind) - grid.at<float>(2, col_ind)) * h*0.5;   /////对于四维Mat的索引，不能使用at函数，因为不支持超过三维的Mat, 大于四维的Mat，既不能使用at，也不能使用ptr访问元素
		}
	}
}

