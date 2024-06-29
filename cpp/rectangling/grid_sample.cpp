# include "grid_sample.h"

using namespace cv;
using namespace std;

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

float SAFE_GET(Mat input, int x, int y, int n, int c, int H, int W)
{
	 if(x >= 0 && x < W && y >=0 && y < H)
	 {
		return input.ptr<float>(n, c, y)[x];
		/*auto shape = input.size;
		float pix = *((float*)input.data + n*shape[1]*shape[2]*shape[3] + c*shape[2]*shape[3] + y*shape[3] + x);
		return pix;*/
	 }
	 else
	 {
		return 0;
	 }
}

#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))


void GridSamplerBilinear(Mat input, Mat grid, Mat& output, int padding_mode, bool align_corners)
{
	int N = input.size[0];
	int C = input.size[1];
	int IH = input.size[2];
	int IW = input.size[3];
	int H = grid.size[1];
	int W = grid.size[2];
	vector<int> out_dims = {N, C, H, W};
	output.create(out_dims, CV_32FC1);
	/*int sizes[4] = {N, C, H, W};
	output.create(4, sizes, CV_32FC1);*/    ///也可以这么创建


	int n, h, w, c;
	for (n = 0; n < N; ++n) 
	{
		for (h = 0; h < H; ++h) 
		{
			for (w = 0; w < W; ++w) 
			{
				// get the corresponding input x, y co-ordinates from grid
				float ix = grid.ptr<float>(0, h, w)[0];  ////batchsize=1
				float iy = grid.ptr<float>(0, h, w)[1];  ////batchsize=1

				// normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
				if (align_corners) 
				{
					ix = ((ix + 1) / 2) * (IW-1);
					iy = ((iy + 1) / 2) * (IH-1);
				} 
				else 
				{
					ix = ((ix + 1) * IW - 1) / 2;
					iy = ((iy + 1) * IH - 1) / 2;
				}

				// get NE, NW, SE, SW pixel values from (x, y)
				int ix_nw = floor(ix);
				int iy_nw = floor(iy);
				int ix_ne = ix_nw + 1;
				int iy_ne = iy_nw;
				int ix_sw = ix_nw;
				int iy_sw = iy_nw + 1;
				int ix_se = ix_nw + 1;
				int iy_se = iy_nw + 1;

				// get surfaces to each neighbor:
				float nw = (ix_se - ix)    * (iy_se - iy);
				float ne = (ix    - ix_sw) * (iy_sw - iy);
				float sw = (ix_ne - ix)    * (iy    - iy_ne);
				float se = (ix    - ix_nw) * (iy    - iy_nw);

				if (padding_mode==1)
				{
					// clip coordinates to image borders
					CLIP_COORDINATES(ix_nw, ix_nw, IW);
					CLIP_COORDINATES(iy_nw, iy_nw, IH);
					CLIP_COORDINATES(ix_ne, ix_ne, IW);
					CLIP_COORDINATES(iy_ne, iy_ne, IH);
					CLIP_COORDINATES(ix_sw, ix_sw, IW);
					CLIP_COORDINATES(iy_sw, iy_sw, IH);
					CLIP_COORDINATES(ix_se, ix_se, IW);
					CLIP_COORDINATES(iy_se, iy_se, IH);
				}

                // calculate bilinear weighted pixel value and set output pixel
                for (c = 0; c < C; ++c) 
                {
					//   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
					// + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
					float nw_val = SAFE_GET(input, ix_nw, iy_nw, n, c, IH, IW);
					float ne_val = SAFE_GET(input, ix_ne, iy_ne, n, c, IH, IW);
					float sw_val = SAFE_GET(input, ix_sw, iy_sw, n, c, IH, IW);
					float se_val = SAFE_GET(input, ix_se, iy_se, n, c, IH, IW);
					float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
					float* pdata = (float*)output.data;
					pdata[n*C*H*W + c*H*W + h*W + w] = out_val;   /////对于四维Mat的索引，不能使用at函数，因为不支持超过三维的Mat, 大于四维的Mat，既不能使用at，也不能使用ptr访问元素
					////output.at<float>(n, c, h, w) = out_val;   /////对于四维Mat的索引，不能使用at函数，因为不支持超过三维的Mat, 大于四维的Mat，既不能使用at，也不能使用ptr访问元素
                }
			}
		}
	}
}


void warp_with_flow(Mat img, Mat flow, Mat& warped_img)
{
	const int H = img.size[2];
	const int W = img.size[3];
	const int flow_h = flow.size[2];
	const int flow_w = flow.size[3];
	
	int shape[4] = {1, flow_h, flow_w, 2};
	Mat target_coord_wh = cv::Mat::zeros(4, shape, CV_32FC1); 
	for(int i=0;i<H;i++)
	{
		for(int j=0;j<W;j++)
		{
			target_coord_wh.ptr<float>(0, i, j)[0] = (j + flow.ptr<float>(0, 0, i)[j]) * 2.0 / W - 1.0;
			target_coord_wh.ptr<float>(0, i, j)[1] = (i + flow.ptr<float>(0, 1, i)[j]) * 2.0 / H - 1.0;
		}
	}

	GridSamplerBilinear(img, target_coord_wh, warped_img, 0, true);
}

Mat convert4dtoimage(Mat blob)
{
	const int H = blob.size[2];
	const int W = blob.size[3];
	const int area = H * W;
	const float* pdata = (float*)blob.data;
	Mat output(H, W, CV_32FC3);
	for(int i=0;i<H;i++)
	{
		for(int j=0;j<W;j++)
		{
			const int idx = i*W+j;
			float pix_r = (pdata[idx] + 1) * 127.5;
			float pix_g = (pdata[area + idx] + 1) * 127.5;
			float pix_b = (pdata[2*area + idx] + 1) * 127.5;
			output.at<Vec3f>(i, j) = Vec3f(pix_r, pix_g, pix_b);

			/*output.ptr<Vec3f>(i, j)[0] = (blob.ptr<float>(0, 0, i)[j] + 1) * 127.5;
			output.ptr<Vec3f>(i, j)[1] = (blob.ptr<float>(0, 1, i)[j] + 1) * 127.5;
			output.ptr<Vec3f>(i, j)[2] = (blob.ptr<float>(0, 2, i)[j] + 1) * 127.5;*/    /////这么写，结果图是有蓝色，不正常
		}
	}
	output.convertTo(output, CV_8UC3);
	return output;
}