# ifndef GRIDSAMPLE
# define GRIDSAMPLE
#include <opencv2/core.hpp>

void GridSamplerBilinear(cv::Mat input, cv::Mat grid, cv::Mat& output, int padding_mode, bool align_corners);

void warp_with_flow(cv::Mat img, cv::Mat flow, cv::Mat& warped_img);


#endif