#include <iostream>
#include"tps2flow.h"

using namespace cv;
using namespace std;


void get_norm_rigid_mesh_inv_grid(Mat& rigid_mesh, Mat& grid, Mat& W_inv, const int input_height, const int input_width, const int grid_h, const int grid_w)
{
	float interval_x = input_width / grid_w;
	float interval_y = input_height / grid_h;
	const int h = grid_h + 1;
	const int w = grid_w + 1;

	/*python里的
	grid_index = (ori_pt[:, :, 1] >=(192-75) ) & (ori_pt[:, :, 1] <= (192+75))
    grid_index = grid_index & (ori_pt[:, :, 0] >= (256-100)) & (ori_pt[:, :, 0] <= (256+100))
    grid_index = ~grid_index
    grid = ori_pt[grid_index]
	*/
	vector<Point2f> keep_pts;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const float x = j * interval_x;
			const float y = i * interval_y;
			if((y>=117 && y<=267) && (x>=156 && x<=356))
			{
				continue;
			}
			keep_pts.push_back(Point2f(x, y));
		}
	}

	const int length = keep_pts.size();
	rigid_mesh.create(length, 2, CV_32FC1);
	Mat W(length + 3, length + 3, CV_32FC1);
	for (int row_ind = 0; row_ind < length; row_ind++)
	{
		rigid_mesh.at<float>(row_ind, 0) = keep_pts[row_ind].x;
		rigid_mesh.at<float>(row_ind, 1) = keep_pts[row_ind].y;
		const float x = keep_pts[row_ind].x * 2.0 / float(input_width) - 1.0;
		const float y = keep_pts[row_ind].y * 2.0 / float(input_height) - 1.0;
		
		W.at<float>(row_ind, 0) = 1;
		W.at<float>(row_ind, 1) = x;
		W.at<float>(row_ind, 2) = y;

		W.at<float>(length, 3 + row_ind) = 1;
		W.at<float>(length + 1, 3 + row_ind) = x;
		W.at<float>(length + 2, 3 + row_ind) = y;
		
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
			const float x = keep_pts[i].x * 2.0 / float(input_width) - 1.0;
			const float y = keep_pts[i].y * 2.0 / float(input_height) - 1.0;
			const float d2_ij = powf(x - grid.at<float>(1, j), 2.0) + powf(y - grid.at<float>(2, j), 2.0);
			grid.at<float>(3 + i, j) = d2_ij * logf(d2_ij + 1e-9);
		}
	}

}

void get_ori_rigid_mesh_tp(Mat rigid_mesh, Mat& tp, const float* offset, const int input_height, const int input_width)
{
	const int length = rigid_mesh.rows;
	tp.create(length + 3, 2, CV_32FC1);
	for (int row_ind = 0; row_ind < length; row_ind++)
	{
		tp.at<float>(row_ind, 0) = (rigid_mesh.at<float>(row_ind, 0) + offset[row_ind * 2])*2.0 / float(input_width) - 1.0;
		tp.at<float>(row_ind, 1) = (rigid_mesh.at<float>(row_ind, 1) + offset[row_ind * 2+1])*2.0 / float(input_height) - 1.0;

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

Mat flow_mesh(Mat predflow_x, Mat predflow_y, Mat input, const int ori_height, const int ori_width)
{
	Mat mesh_x(ori_height, ori_width, CV_32FC1);
	Mat mesh_y(ori_height, ori_width, CV_32FC1);
	for (int i = 0; i < ori_height; i++)
	{
		for (int j = 0; j < ori_width; j++)
		{
			mesh_x.at<float>(i, j) = predflow_x.at<float>(i, j) + j;
			mesh_y.at<float>(i, j) = predflow_y.at<float>(i, j) + i;
		}
	}
	Mat pred_out;
	cv::remap(input, pred_out, mesh_x, mesh_y, INTER_LINEAR);
	pred_out = (pred_out + 1.0) * 255.0;
	pred_out.convertTo(pred_out, CV_8UC3);
	return pred_out;
}

void split2xy(Mat flow, Mat& flow_x, Mat& flow_y)
{
	const int h = flow.size[2];
	const int w = flow.size[3];
	flow_x.create(h, w, CV_32FC1);
	flow_y.create(h, w, CV_32FC1);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			flow_x.at<float>(i, j) = flow.ptr<float>(0, 0, i)[j];
			flow_y.at<float>(i, j) = flow.ptr<float>(0, 1, i)[j];
		}
	}
}