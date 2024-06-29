# ifndef UPSAMPLE
# define UPSAMPLE
#include <iostream>
#include <optional>
#include <opencv2/core.hpp>


template <typename scalar_t>
inline scalar_t compute_scales_value(
    const std::optional<double> scale,
    int64_t input_size,
    int64_t output_size) {
      // see Note [compute_scales_value]
      // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
      return (scale.has_value() && scale.value() > 0.)
          ? static_cast<scalar_t>(1.0 / scale.value())
          : (static_cast<scalar_t>(input_size) / output_size);
}

template <typename scalar_t>
inline scalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners,
    const std::optional<double> scale) {
  // see Note [area_pixel_compute_scale]
  if(align_corners) {
    if(output_size > 1) {
      return static_cast<scalar_t>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<scalar_t>(0);
    }
  } else {
    return compute_scales_value<scalar_t>(scale, input_size, output_size);
  }
}

template <typename scalar_t>
inline scalar_t area_pixel_compute_source_index(
    scalar_t scale,
    int64_t dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t src_idx = scale * (dst_index + static_cast<scalar_t>(0.5)) -
        static_cast<scalar_t>(0.5);
    // [Note] Follow Opencv resize logic:
    // We allow negative src_idx here and later will use
    //   dx = src_idx - floorf(src_idx)
    // to compute the "distance"(which affects weights).
    // For linear modes, weight distribution doesn't matter
    // for negative indices as they use 2 pixels to interpolate.
    // For example, [-1, 0], they both use pixel 0 value so it
    // doesn't affect if we bound the src_idx to 0 or not.
    // TODO: Our current linear mode impls use unbound indices
    // where we should and then remove this cubic flag.
    // This matters in cubic mode, as we might need [-1, 0, 1, 2]
    // to interpolate and the weights can be affected.
    return (!cubic && src_idx < static_cast<scalar_t>(0)) ? scalar_t(0)
                                                          : src_idx;
  }
}


void UpSamplingBilinear(cv::Mat input, cv::Mat& output, int outputHeight, int outputWidth, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w);
void upsample_forward(cv::Mat inp, cv::Mat& out, int outHeight, int outWidth, std::string interpolation, bool alignCorners, bool halfPixelCenters);

#endif