#include "support/vlm_utils.h"

#include <gtest/gtest.h>
#include <tvm/runtime/tensor.h>

namespace mlc {
namespace llm {

TEST(VLMUtilsTest, Qwen35ResizeUsesDefaultVisionConfig) {
  tvm::runtime::Tensor image =
      tvm::runtime::Tensor::Empty({1, 80, 80, 3}, {kDLUInt, 8, 1}, {kDLCPU, 0});
  int resized_height = 0;
  int resized_width = 0;
  int crop_height = 0;
  int crop_width = 0;

  CalculateQwen35ResizeShape(image, Qwen35ImageResizeConfig(), &resized_height, &resized_width);
  CalculateQwen35CropShape(image, Qwen35ImageResizeConfig(), &crop_height, &crop_width);

  EXPECT_EQ(resized_height, 256);
  EXPECT_EQ(resized_width, 256);
  EXPECT_EQ(crop_height, 16);
  EXPECT_EQ(crop_width, 16);
}

TEST(VLMUtilsTest, Qwen35ResizeUsesModelVisionConfig) {
  tvm::runtime::Tensor image =
      tvm::runtime::Tensor::Empty({1, 80, 80, 3}, {kDLUInt, 8, 1}, {kDLCPU, 0});
  Qwen35ImageResizeConfig config;
  config.patch_size = 8;
  config.spatial_merge_size = 2;
  config.min_pixels = 4096;
  config.max_pixels = 16777216;
  int resized_height = 0;
  int resized_width = 0;
  int crop_height = 0;
  int crop_width = 0;

  CalculateQwen35ResizeShape(image, config, &resized_height, &resized_width);
  CalculateQwen35CropShape(image, config, &crop_height, &crop_width);

  EXPECT_EQ(resized_height, 80);
  EXPECT_EQ(resized_width, 80);
  EXPECT_EQ(crop_height, 10);
  EXPECT_EQ(crop_width, 10);
}

}  // namespace llm
}  // namespace mlc
