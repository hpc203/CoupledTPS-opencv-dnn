# include "upsample.h"

using namespace cv;
using namespace std;


void UpSamplingBilinear(Mat input, Mat& output, int outputHeight, int outputWidth, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w)
{
    int nbatch = input.size[0];
    int channels = input.size[1];
    int inputHeight = input.size[2];
    int inputWidth = input.size[3];
    int out_dims[4] = {nbatch, channels, outputHeight, outputWidth};
	output.create(4, out_dims, CV_32FC1);
    //////想要使得程序健壮，可以添加shapeCheck函数，检查输入输出的维度是否正确

    float *idata = (float*)input.data;
    float *odata = (float*)output.data;
    channels = nbatch * channels;
    // special case: just copy
    if (inputHeight == outputHeight && inputWidth == outputWidth)
    {
        input.copyTo(output);   ////opencv的Mat直接拷贝
        /*for (int h2 = 0; h2 < outputHeight; ++h2) 
        {
            const int h1 = h2;
            for (int w2 = 0; w2 < outputWidth; ++w2) 
            {
                const int w1 = w2;
                const float* pos1 = &idata[h1 * inputWidth + w1];
                float* pos2 = &odata[h2 * outputWidth + w2];
                for (int c = 0; c < channels; ++c) 
                {
                    pos2[0] = pos1[0];
                    pos1 += inputWidth * inputHeight;
                    pos2 += outputWidth * outputHeight;
                }
            }
        }*/
        return;
    }

    /*const float rheight =(outputHeight > 1) ? (float)(inputHeight - 1)/(outputHeight - 1) : 0.f;
    const float rwidth = (outputWidth > 1) ? (float)(inputWidth - 1) / (outputWidth - 1) : 0.f;*/
    const auto rheight = area_pixel_compute_scale<float>(inputHeight, outputHeight, align_corners, scales_h);
    const auto rwidth = area_pixel_compute_scale<float>(inputWidth, outputWidth, align_corners, scales_w);

    for (int h2 = 0; h2 < outputHeight; ++h2) 
    {
        ////const float h1r = rheight * h2;
        const float h1r = area_pixel_compute_source_index<float>(rheight, h2, align_corners, /*cubic=*/false);
        const int h1 = h1r;
        const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < outputWidth; ++w2) 
        {
            ///const float w1r = rwidth * w2;
            const float w1r = area_pixel_compute_source_index<float>(rwidth, w2, align_corners, /*cubic=*/false);
            const int w1 = w1r;
            const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = (float)1. - w1lambda;
            const float* pos1 = &idata[h1 * inputWidth + w1];
            float* pos2 = &odata[h2 * outputWidth + w2];
            for (int c = 0; c < channels; ++c) 
            {
                pos2[0] = h0lambda * (w0lambda * pos1[0]+ w1lambda * pos1[w1p])
                        + h1lambda * (w0lambda * pos1[h1p * inputWidth]
                        + w1lambda * pos1[h1p * inputWidth + w1p]);
                pos1 += inputWidth * inputHeight;
                pos2 += outputWidth * outputHeight;
            }
        }
    }
}


void upsample_forward(Mat inp, Mat& out, int outHeight, int outWidth, string interpolation, bool alignCorners, bool halfPixelCenters)
{
    float scaleWidth, scaleHeight;
    if (alignCorners && outHeight > 1)
            scaleHeight = static_cast<float>(inp.size[2] - 1) / (outHeight - 1);
        else
            scaleHeight = static_cast<float>(inp.size[2]) / outHeight;

        if (alignCorners && outWidth > 1)
            scaleWidth = static_cast<float>(inp.size[3] - 1) / (outWidth - 1);
        else
            scaleWidth = static_cast<float>(inp.size[3]) / outWidth;
    int depth = inp.depth();
    if (interpolation == "nearest")
    {
        const int inpHeight = inp.size[2];
        const int inpWidth = inp.size[3];
        const int inpSpatialSize = inpHeight * inpWidth;
        const int outSpatialSize = outHeight * outWidth;
        const int numPlanes = inp.size[0] * inp.size[1];
        CV_Assert_N(inp.isContinuous(), out.isContinuous());

        Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
        Mat outPlanes = out.reshape(1, numPlanes * outHeight);

        float heightOffset = 0.0f;
        float widthOffset = 0.0f;

        if (halfPixelCenters)
        {
            heightOffset = 0.5f * scaleHeight;
            widthOffset = 0.5f * scaleWidth;
        }

        if (depth == CV_8S)
        {
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = y * scaleHeight + heightOffset;
                int y0 = halfPixelCenters ? std::floor(input_y) : lroundf(input_y);
                y0 = std::min(y0, inpHeight - 1);

                const int8_t* inpData_row = inpPlanes.ptr<int8_t>(y0);

                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = x * scaleWidth + widthOffset;
                    int x0 = halfPixelCenters ? std::floor(input_x) : lroundf(input_x);
                    x0 = std::min(x0, inpWidth - 1);

                    int8_t* outData = outPlanes.ptr<int8_t>(y, x);
                    const int8_t* inpData_row_c = inpData_row;

                    for (int c = 0; c < numPlanes; ++c)
                    {
                        *outData = inpData_row_c[x0];

                        inpData_row_c += inpSpatialSize;
                        outData += outSpatialSize;
                    }
                }
            }
        }
        else
        {
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = y * scaleHeight + heightOffset;
                int y0 = halfPixelCenters ? std::floor(input_y) : lroundf(input_y);
                y0 = std::min(y0, inpHeight - 1);

                const float* inpData_row = inpPlanes.ptr<float>(y0);

                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = x * scaleWidth + widthOffset;
                    int x0 = halfPixelCenters ? std::floor(input_x) : lroundf(input_x);
                    x0 = std::min(x0, inpWidth - 1);

                    float* outData = outPlanes.ptr<float>(y, x);
                    const float* inpData_row_c = inpData_row;

                    for (int c = 0; c < numPlanes; ++c)
                    {
                        *outData = inpData_row_c[x0];

                        inpData_row_c += inpSpatialSize;
                        outData += outSpatialSize;
                    }
                }
            }
        }
    }
    else if (interpolation == "bilinear" || interpolation == "opencv_linear")
    {
        const int inpHeight = inp.size[2];
        const int inpWidth = inp.size[3];
        const int inpSpatialSize = inpHeight * inpWidth;
        const int outSpatialSize = outHeight * outWidth;
        const int numPlanes = inp.size[0] * inp.size[1];
        CV_Assert_N(inp.isContinuous(), out.isContinuous());

        Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
        Mat outPlanes = out.reshape(1, numPlanes * outHeight);
        if (depth == CV_8S)
        {
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = halfPixelCenters ? std::max((y + 0.5f) * scaleHeight - 0.5f, 0.0f) : y * scaleHeight;
                int y0 = static_cast<int>(input_y);
                const int8_t* inpData_row0 = inpPlanes.ptr<int8_t>(y0);
                const int8_t* inpData_row1 = inpPlanes.ptr<int8_t>(std::min(y0 + 1, inpHeight - 1));
                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = halfPixelCenters ? std::max((x + 0.5f) * scaleWidth - 0.5f, 0.0f) : x * scaleWidth;
                    int x0 = static_cast<int>(input_x);
                    int x1 = std::min(x0 + 1, inpWidth - 1);

                    int8_t* outData = outPlanes.ptr<int8_t>(y, x);
                    const int8_t* inpData_row0_c = inpData_row0;
                    const int8_t* inpData_row1_c = inpData_row1;
                    for (int c = 0; c < numPlanes; ++c)
                    {
                        *outData = static_cast<int8_t>(inpData_row0_c[x0] +
                            (input_y - y0) * (inpData_row1_c[x0] - inpData_row0_c[x0]) +
                            (input_x - x0) * (inpData_row0_c[x1] - inpData_row0_c[x0] +
                            (input_y - y0) * (inpData_row1_c[x1] - inpData_row0_c[x1] - inpData_row1_c[x0] + inpData_row0_c[x0])));

                        inpData_row0_c += inpSpatialSize;
                        inpData_row1_c += inpSpatialSize;
                        outData += outSpatialSize;
                    }
                }
            }
        }
        else
        {
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = y * scaleHeight;
                int y0 = static_cast<int>(input_y);
                const float* inpData_row0 = inpPlanes.ptr<float>(y0);
                const float* inpData_row1 = inpPlanes.ptr<float>(std::min(y0 + 1, inpHeight - 1));
                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = x * scaleWidth;
                    int x0 = static_cast<int>(input_x);
                    int x1 = std::min(x0 + 1, inpWidth - 1);

                    float* outData = outPlanes.ptr<float>(y, x);
                    const float* inpData_row0_c = inpData_row0;
                    const float* inpData_row1_c = inpData_row1;
                    for (int c = 0; c < numPlanes; ++c)
                    {
                        *outData = inpData_row0_c[x0] +
                            (input_y - y0) * (inpData_row1_c[x0] - inpData_row0_c[x0]) +
                            (input_x - x0) * (inpData_row0_c[x1] - inpData_row0_c[x0] +
                            (input_y - y0) * (inpData_row1_c[x1] - inpData_row0_c[x1] - inpData_row1_c[x0] + inpData_row0_c[x0]));

                        inpData_row0_c += inpSpatialSize;
                        inpData_row1_c += inpSpatialSize;
                        outData += outSpatialSize;
                    }
                }
            }
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "Unknown interpolation: " + interpolation);
    }
}