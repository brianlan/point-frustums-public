//
// Created by Enrico Stauss on 18.01.24.
//
#include <torch/extension.h>
#include <torch/torch.h>
#include <tuple>
#include <numeric>
#include <queue>
#include "iou_box3d/iou_utils.h"
#include "nms/nms.h"


// Get the (boxes_centers, boxes_tris, boxes_planes, boxes_volume)
std::tuple<std::vector<vec3<float>>, std::vector<face_verts>, std::vector<face_verts>, std::vector<float>> CentersTrisPlanesVolumes(const at::Tensor& boxes) {
    const int N = boxes.size(0);

    // Create an accessor for the boxes
    auto boxes_a = boxes.accessor<float, 3>();

    std::vector<vec3<float>> centers;
    std::vector<face_verts> tris;
    std::vector<face_verts> planes;
    std::vector<float> volumes;

    centers.reserve(N);
    tris.reserve(N);
    planes.reserve(N);
    volumes.reserve(N);

    for (int i=0; i < N; i++) {
        const auto& box = boxes_a[i];
        centers.emplace_back(BoxCenter(boxes[i]));
        tris.emplace_back(GetBoxTris(box));
        planes.emplace_back(GetBoxPlanes(box));
        volumes.emplace_back(BoxVolume(tris[i], centers[i]));
    }
    return std::make_tuple(centers, tris, planes, volumes);
}


float IoU(
        const vec3<float> &b1_center,
        const face_verts &b1_tris,
        const face_verts &b1_planes,
        const float &b1_vol,
        const vec3<float> &b2_center,
        const face_verts &b2_tris,
        const face_verts &b2_planes,
        const float &b2_vol) {

    face_verts box1_intersect = BoxIntersections(b1_tris, b2_planes, b2_center);
    face_verts box2_intersect = BoxIntersections(b2_tris, b1_planes, b1_center);
    box1_intersect.reserve(box1_intersect.size() + box2_intersect.size());

    if (box2_intersect.size() > 0) {
        // Identify if any triangles in Box2 are coplanar with Box1
        std::vector<int> tri2_keep(box2_intersect.size());
        std::fill(tri2_keep.begin(), tri2_keep.end(), 1);
        for (size_t b1 = 0; b1 < box1_intersect.size(); ++b1) {
            for (size_t b2 = 0; b2 < box2_intersect.size(); ++b2) {
                const bool is_coplanar = IsCoplanarTriTri(box1_intersect[b1], box2_intersect[b2]);
                const float area = FaceArea(box1_intersect[b1]);
                if ((is_coplanar) && (area > aEpsilon)) {
                    tri2_keep[b2] = 0;
                }
            }
        }

        // Keep only the non-coplanar triangles in Box2 - add them to the Box1 triangles.
        for (size_t b2 = 0; b2 < box2_intersect.size(); ++b2) {
            if (tri2_keep[b2] == 1) {
                box1_intersect.emplace_back((box2_intersect[b2]));
            }
        }
    }

    float iou = 0.0;
    float vol;
    // If there are triangles in the intersecting shape
    if (box1_intersect.size() > 0) {
        // The intersecting shape is a polyhedron made up of the
        // triangular faces that are all now in box1_intersect.
        // Calculate the polyhedron center
        const vec3<float> polyhedron_center = PolyhedronCenter(box1_intersect);
        // Compute intersecting polyhedron volume
        vol = BoxVolume(box1_intersect, polyhedron_center);
        // Compute IoU
        iou = vol / (b1_vol + b2_vol - vol);
    }
    return iou;
}


at::Tensor NMSCpu(
        const at::Tensor &labels_t,
        const at::Tensor &scores,
        const at::Tensor &boxes,
        const double iou_threshold,
        const double distance_threshold) {

    if (labels_t.numel() == 0) {
        return at::empty({0}, boxes.options().dtype(at::kLong));
    }

    const int N = labels_t.size(0);

    std::vector<vec3<float>> centers;
    std::vector<face_verts> tris;
    std::vector<face_verts> planes;
    std::vector<float> volumes;
    std::tie(centers, tris, planes, volumes) = CentersTrisPlanesVolumes(boxes);

    auto labels = labels_t.data_ptr<int64_t>();

    at::Tensor scores_sorted_idx_t = std::get<1>(scores.sort(0));
    auto scores_sorted_idx = scores_sorted_idx_t.data_ptr<int64_t>();

    // Initialize a boolean vector to all-true (mask which boxes are kept)
    at::Tensor remains_t = torch::ones({N}, labels_t.options().dtype(at::kByte));
    auto remains = remains_t.data_ptr<uint8_t>();

    // Algorithm: Iterate through a nested loop where the outer loop defines the starting idx
    //  and the inner loop checks if any of the consecutive boxes remove the current one. If
    //  a match is found, set the remains entry to 0 and break the inner loop.
    for (int64_t _i = 0; _i < N; _i++) {
        // Retrieve the index that accesses the sorted scores
        auto i = scores_sorted_idx[_i];
        auto label_i = labels[i];
        for (int64_t _j = _i + 1; _j < N; _j++) {
            // Retrieve the index that accesses the sorted scores
            auto j = scores_sorted_idx[_j];
            // If boxes i and j are assigned different labels, continue
            if (label_i != labels[j])
                continue;

            // If the center distance is large enough, there is no need to evaluate the IoU
            if ((norm(centers[i] - centers[j]) / volumes[i]) > distance_threshold)
                continue;

            // Due to the sorted access, box j is guaranteed to have a higher score than box i
            // Evaluate IoU between boxes i and j, if the threshold is surpassed, keep only box j
            float iou = IoU(centers[i], tris[i], planes[i], volumes[i], centers[j], tris[j], planes[j], volumes[j]);
            if (iou >= iou_threshold) {
                remains[i] = 0;
                break;
            }
        }
    }

    return remains_t;
}