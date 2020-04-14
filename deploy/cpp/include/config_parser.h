//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "yaml-cpp/yaml.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace PaddleDetection {

// Inference model configuration parser
class ConfigPaser {
 public:
  ConfigPaser() {}

  ~ConfigPaser() {}

  std::string rename_model_arch(const std::string& arch) {
    if (arch.find("YOLO") != std::string::npos) {
      return "YOLO";
    }
    if (arch.find("RCNN") != std::string::npos) {
      return "RCNN";
    }
    if (arch.find("RetinaNet") != std::string::npos) {
      return "RetinaNet";
    }
    if (arch.find("SSD") != std::string::npos) {
      return "SSD";
    }
    return "UNKNOWN";
  }

  bool load_config(const std::string& model_dir,
                   const std::string& cfg = "model.yml") {
    // Load as a YAML::Node
    YAML::Node config;
    config = YAML::LoadFile(model_dir + OS_PATH_SEP + cfg);

    // Get runtime mode : fluid, trt_int8, trt_fp16, trt_fp32
    if (config["mode"].IsDefined()) {
      mode_ = config["mode"].as<std::string>();
    } else {
      mode_ = "fluid";
    }

    // Get model arch : YOLO, SSD, RetinaNet, RCNN
    if (config["Model"].IsDefined()) {
      arch_ = rename_model_arch(config["Model"].as<std::string>());
    } else {
      std::cerr << "Please set model arch,"
                << "support value : YOLO, SSD, RetinaNet, RCNN."
                << std::endl;
      return false;
    }

    // Get min_subgraph_size for tensorrt
    if (config["min_subgraph_size"].IsDefined()) {
      min_subgraph_size_ = config["min_subgraph_size"].as<int>();
    } else {
      min_subgraph_size_ = 3;
    }
    // Get draw_threshold for visualization
    if (config["draw_threshold"].IsDefined()) {
      draw_threshold_ = config["draw_threshold"].as<float>();
    } else {
      draw_threshold_ = 0.5;
    }

    // Get with_background
    if (config["with_background"].IsDefined()) {
      with_background_ = config["with_background"].as<bool>();
    } else {
      with_background_ = false;
    }
    // Get Preprocess for preprocessing
    if (config["Transforms"].IsDefined()) {
      preprocess_info_ = config["Transforms"];
    } else {
      std::cerr << "Please set Transforms." << std::endl;
      return false;
    }
    // Get label_list for visualization
    if (config["_Attributes"].IsDefined()) {
      auto attrs = config["_Attributes"];
      label_list_ = attrs["labels"].as<std::vector<std::string>>();
    } else {
      std::cerr << "Please set _Attributes.labels." << std::endl;
      return false;
    }

    return true;
  }
  std::string mode_;
  float draw_threshold_;
  std::string arch_;
  int min_subgraph_size_;
  bool with_background_;
  YAML::Node preprocess_info_;
  std::vector<std::string> label_list_;
};

}  // namespace PaddleDetection

