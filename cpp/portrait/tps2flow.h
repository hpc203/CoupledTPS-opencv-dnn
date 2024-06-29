# ifndef TPS2FLOW
# define TPS2FLOW
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


void get_norm_rigid_mesh_inv_grid(cv::Mat& rigid_mesh, cv::Mat& grid, cv::Mat& W_inv, const int input_height, const int input_width, const int grid_h, const int grid_w);
void get_ori_rigid_mesh_tp(cv::Mat rigid_mesh, cv::Mat& tp, const float* offset, const int input_height, const int input_width);
void _transform(cv::Mat T_g, cv::Mat grid, const int h, const int w, cv::Mat& flow);
cv::Mat flow_mesh(cv::Mat predflow_x, cv::Mat predflow_y, cv::Mat input, const int ori_height, const int ori_width);
void split2xy(cv::Mat flow, cv::Mat& flow_x, cv::Mat& flow_y);
#endif
