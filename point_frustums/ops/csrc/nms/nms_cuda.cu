//
// Created by Enrico Stauss on 19.01.24.
//
#include "iou_box3d/iou_utils.cuh"
#include "nms/nms_utils.cuh"
#include "utils/float_math.cuh"
#include <math.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/tuple.h>

// For the CPU implementation, the strategy is:
// 1. Get the index that sorts the score (and corresponding labels/boxes) in
// ascending order
// 2. Precalculate centers, tris, planes and volumes for each box
// 3. Nested loop:
// 3.1 Outer loop: Iterate over boxes as sorted by increasing score
// 3.2 Inner loop: Iterate over all other boxes (with higher score)
//  - Evaluate if the label matches (continue otherwise)
//  - Evaluate if the scaled center distance is lower than the distance
//  threshold (continue otherwise)
//  - Evaluate the IoU and check if the threshold is surpassed. If so, the box
//  indexed by the outer loop is not kept.

// The CUDA implementation cannot parallelize the iterative strategy. Instead,
// the IoU is evaluated for all boxes that are of the same class, score higher
// and are close enough. If the IoU surpasses the threshold we increment the
// duplicate count by one. The exact count is of no relevance but if the count
// is N > 0, we know that the box has N better scoring alternatives and can be
// removed.

template <typename T>
__global__ void BoxesMetaKernel(
    const at::PackedTensorAccessor32<T, 3, at::RestrictPtrTraits> boxes,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> boxes_parameters
) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= boxes.size(0)) {
    return; // Return early if index is out of bounds
  }

  // Reinterpret the underlying memory in the adequate types
  BoxParameters *box_parameters = reinterpret_cast<BoxParameters *>(boxes_parameters[tid].data());

  // Evaluate and write {center [N, 3], tris [N, NUM_TRIS, 4, 3], planes [N, 4, 3], volume [N,]}.
  // Evaluate the box center (float3) and write to the pre-allocated tensor elements.
  box_parameters->center = BoxCenter(boxes[tid]);
  // Evaluate and directly write the box tris (float: [4, 3]) into the pre-allocated tensor elements.
  GetBoxTris(boxes[tid], box_parameters->triangles);
  // Evaluate and directly write the box planes (float: [4, 3]) into the pre-allocated tensor elements.
  GetBoxPlanes(boxes[tid], box_parameters->planes);
  // Evaluate the box volume (float) and write to the pre-allocated tensor element.
  box_parameters->volume = BoxVolume(box_parameters->triangles, box_parameters->center, NUM_TRIS);
}

template <typename T>
__global__ void NMSDuplicateKernel(
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> boxes_label,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> boxes_sort_idx,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> boxes_parameters, double iou_threshold,
    double distance_threshold, at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> duplicate_count
) {

  // Parallelize over all possible combinations by executing a 2D grid
  const int tid_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_j = blockIdx.y * blockDim.y + threadIdx.y;

  // Return if out of bounds.
  const size_t n_boxes = boxes_label.size(0);
  if (tid_i >= n_boxes || tid_j >= n_boxes) {
    return;
  }

  // Check if index i is greater or equal to index j; if so, return early (box j
  // is no higher-scoring duplicate).
  if (tid_i >= tid_j) {
    return;
  }

  // Get the index of the sorted boxes
  const int i = boxes_sort_idx[tid_i];
  const int j = boxes_sort_idx[tid_j];

  // Check if boxes i and j are of the same class; return early if not.
  if (boxes_label[i] != boxes_label[j]) {
    return;
  }

  // From here on, I will need to access the raw data of center, tris, planes
  // and volume Reinterpret the underlying memory in the adequate types
  const BoxParameters *box_i_parameters = reinterpret_cast<BoxParameters *>(boxes_parameters[i].data());
  const BoxParameters *box_j_parameters = reinterpret_cast<BoxParameters *>(boxes_parameters[j].data());

  // Evaluate the scaled center distance; return early if the (lower) threshold
  // is exceeded.
  if ((norm(*box_i_parameters->center - *box_j_parameters->center) / box_i_parameters->volume) > distance_threshold) {
    return;
  }

  // Evaluate the IoU between box i and j.
  thrust::tuple<float, float> result = GetIntersectionAndUnion(
      box_i_parameters->center, box_j_parameters->center, box_i_parameters->triangles, box_j_parameters->triangles,
      box_i_parameters->planes, box_j_parameters->planes, box_i_parameters->volume, box_j_parameters->volume
  );
  const float iou = thrust::get<1>(result);
  // Increment the count for box i by 1 if the IoU between boxes i and j exceeds
  // the threshold.
  if (iou >= iou_threshold) {
    duplicate_count[i][j] = 1;
  }
}

at::Tensor NMSCuda(
    const at::Tensor &labels, const at::Tensor &scores, const at::Tensor &boxes, const double iou_threshold,
    const double distance_threshold
) {

  // Check inputs are on the same device
  at::TensorArg labels_t{labels, "labels", 1}, scores_t{scores, "scores", 2}, boxes_t{boxes, "boxes", 3};
  at::CheckedFrom c = "NMSCuda";
  at::checkAllSameGPU(c, {boxes_t, scores_t, labels_t});
  at::checkAllSameType(c, {boxes_t, scores_t});

  // Set the device for the kernel launch based on the device of boxes1
  at::cuda::CUDAGuard device_guard(boxes.device());

  const size_t blocks = 256;
  const size_t threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(boxes.size(0) == scores.size(0), "The number of boxes must equal the number of scores");
  TORCH_CHECK(labels.size(0) == scores.size(0), "The number of labels must equal the number of scores");
  TORCH_CHECK((boxes.size(1) == 8 && boxes.size(2) == 3), "Boxes must be of shape (N, 8, 3)");

  const int64_t N_BOXES = boxes.size(0);

  // 0. Sort everything by ascending score.
  auto sorting_index = scores.argsort().to(at::kInt);

  // 1. Evaluate center, tris, planes and volume of all boxes
  auto options_float = boxes.options().dtype(at::kFloat);
  auto boxes_parameters = at::empty({N_BOXES, 3 + N_FLOATS_PER_TRIANGLE + N_FLOATS_PER_PLANE + 1}, options_float);
  BoxesMetaKernel<<<blocks, threads, 0, stream>>>(
      boxes.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
      boxes_parameters.packed_accessor32<float, 2, at::RestrictPtrTraits>()
  );

  // 2. Initialize an all-zero vector that will hold the duplicate count for
  // each box (0 indicates a box not having better duplicate representations)
  // and execute the kernel to accumulate the duplicate count
  // auto duplicate_count = at::zeros_like(labels, at::kInt);
  auto is_duplicate = at::zeros({N_BOXES, N_BOXES}, boxes.options().dtype(at::kInt));

  dim3 blocksPerGrid(blocks, blocks);
  dim3 threadsPerBlock(16, 16);

  cudaStreamSynchronize(stream);
  NMSDuplicateKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      labels.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      sorting_index.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      boxes_parameters.packed_accessor32<float, 2, at::RestrictPtrTraits>(), iou_threshold, distance_threshold,
      is_duplicate.packed_accessor32<int, 2, at::RestrictPtrTraits>()
  );
  AT_CUDA_CHECK(cudaGetLastError());

  // 3. Return count==0 boolean mask to indicate which boxes to keep
  auto duplicate_count = is_duplicate.sum(1);
  return at::eq(duplicate_count, at::zeros_like(duplicate_count));
}
