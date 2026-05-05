/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/image_utils.cc
 */
#include "vlm_utils.h"

#include <algorithm>
#include <cmath>

namespace mlc {
namespace llm {

namespace {

int RoundToMultipleNearestEven(int value, int factor) {
  int quotient = value / factor;
  int remainder = value % factor;
  if (2 * remainder < factor) {
    return quotient * factor;
  }
  if (2 * remainder > factor) {
    return (quotient + 1) * factor;
  }
  return ((quotient % 2) == 0 ? quotient : quotient + 1) * factor;
}

void Qwen35SmartResize(int height, int width, int* p_target_height, int* p_target_width) {
  constexpr int factor = 32;        // patch_size(16) * spatial_merge_size(2)
  constexpr int min_pixels = 65536;
  constexpr int max_pixels = 16777216;
  TVM_FFI_ICHECK_GT(height, 0);
  TVM_FFI_ICHECK_GT(width, 0);
  TVM_FFI_ICHECK_LE(static_cast<double>(std::max(height, width)) / std::min(height, width), 200.0)
      << "Image aspect ratio is too large: height=" << height << ", width=" << width;

  int resized_height = std::max(factor, RoundToMultipleNearestEven(height, factor));
  int resized_width = std::max(factor, RoundToMultipleNearestEven(width, factor));
  if (static_cast<double>(resized_height) * resized_width > max_pixels) {
    double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
    resized_height = static_cast<int>(std::floor(height / beta / factor)) * factor;
    resized_width = static_cast<int>(std::floor(width / beta / factor)) * factor;
  } else if (static_cast<double>(resized_height) * resized_width < min_pixels) {
    double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
    resized_height = static_cast<int>(std::ceil(height * beta / factor)) * factor;
    resized_width = static_cast<int>(std::ceil(width * beta / factor)) * factor;
  }

  *p_target_height = resized_height;
  *p_target_width = resized_width;
}

}  // namespace

void CalculateResizeShape(tvm::runtime::Tensor image_data, std::string model_type,
                          int* p_target_height, int* p_target_width) {
  TVM_FFI_ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  int height = image_data->shape[1];
  int width = image_data->shape[2];

  if ("qwen3_5" == model_type) {
    Qwen35SmartResize(height, width, p_target_height, p_target_width);
  } else if ("phi3_v" == model_type) {
    const int hd_num = 4;
    double ratio = static_cast<double>(width) / height;
    int scale = 1;
    while (scale * std::ceil(scale / ratio) <= hd_num) {
      scale += 1;
    }
    scale -= 1;
    *p_target_width = static_cast<int>(scale * 336);
    *p_target_height = static_cast<int>(*p_target_width / ratio);
  }
}

void CalculatePadShape(tvm::runtime::Tensor image_data, std::string model_type, int* p_pad_height,
                       int* p_pad_width) {
  TVM_FFI_ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  if ("phi3_v" == model_type) {
    int resized_height = 0, resized_width = 0;
    CalculateResizeShape(image_data, model_type, &resized_height, &resized_width);
    int tar = (int)(ceil(resized_height / 336.0) * 336);
    int top_padding = (int)((tar - resized_height) / 2);
    int bottom_padding = tar - resized_height - top_padding;
    TVM_FFI_ICHECK_EQ(tar, resized_height + top_padding + bottom_padding)
        << "Padding size not equal!";
    *p_pad_height = tar;
    *p_pad_width = resized_width;
  }
}

void CalculateCropShape(tvm::runtime::Tensor image_data, std::string model_type, int* p_crop_height,
                        int* p_crop_width) {
  TVM_FFI_ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  if ("qwen3_5" == model_type) {
    int resized_height = 0, resized_width = 0;
    CalculateResizeShape(image_data, model_type, &resized_height, &resized_width);
    *p_crop_height = resized_height / 16;
    *p_crop_width = resized_width / 16;
  } else if ("phi3_v" == model_type) {
    int pad_h = 0, pad_w = 0;
    CalculatePadShape(image_data, model_type, &pad_h, &pad_w);
    *p_crop_height = pad_h / 336;
    *p_crop_width = pad_w / 336;
  }
}

}  // namespace llm
}  // namespace mlc
