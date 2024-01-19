//
// Created by Enrico Stauss on 18.01.24.
//

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "iou_box3d/iou_box3d.h"

// CPU implementation
at::Tensor NMSCpu(
    const at::Tensor &labels,
    const at::Tensor &scores,
    const at::Tensor &boxes,
    const double iou_threshold,
    const double distance_threshold);

// CUDA implementation
//std::tuple <at::Tensor, at::Tensor> NMSCuda(
//        const at::Tensor &labels,
//        const at::Tensor &scores,
//        const at::Tensor &boxes);

// Main Entrypoint
inline at::Tensor NMS(
    const at::Tensor &labels,
    const at::Tensor &scores,
    const at::Tensor &boxes,
    const double iou_threshold,
    const double distance_threshold) {
    return NMSCpu(labels.contiguous(), scores.contiguous(), boxes.contiguous(), iou_threshold, distance_threshold);
}



//if (labels.is_cuda() || (scores.is_cuda() || boxes.is_cuda()) {
//#ifdef WITH_CUDA
//    CHECK_CUDA(labels);
//    CHECK_CUDA(scores);
//    CHECK_CUDA(boxes);
//    //return NMSCuda(labels.contiguous(), scores.contiguous(), boxes.contiguous());
//    return 0;
//#else
//        AT_ERROR("Not compiled with GPU support.");
//#endif
//    }
//    return NMSCpu(labels.contiguous(), scores.contiguous(), boxes.contiguous());
//}