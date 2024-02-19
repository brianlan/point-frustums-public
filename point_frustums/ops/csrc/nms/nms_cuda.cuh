//
// Created by Enrico Stauss on 19.01.24.
//
#include <float.h>
#include <math.h>
#include "utils/float_math.cuh"
#include "iou_box3d/iou_utils.cuh"


template<typename BoxTris, typename FaceVertsBoxPlanes>
__device__ inline std::tuple<float, float>
GetIntersectionAndUnion(const float3 &box1_center, const float3 &box2_center, const BoxTris &box1_tris,
                        const BoxTris &box2_tris, const FaceVertsBoxPlanes &box1_planes,
                        const FaceVertsBoxPlanes box2_planes, const float box1_volume, const float box2_volume) {
    // This is an extraction of the CUDA kernel from the PyTorch3d code for the calculation of the 3D IoU.

    // Initialize the faces for both boxes (MAX_TRIS is the max faces possible in the intersecting shape)
    FaceVerts box1_intersect[MAX_TRIS];
    FaceVerts box2_intersect[MAX_TRIS];
    // Tris in Box1 intersection with Planes in Box2
    for (int j = 0; j < NUM_TRIS; ++j) {
        box1_intersect[j] = box1_tris[j];
    }
    // Tris in Box2 intersection with Planes in Box1
    for (int j = 0; j < NUM_TRIS; ++j) {
        box2_intersect[j] = box2_tris[j];
    }
    // Get the count of the actual number of faces in the intersecting shape
    int box1_count = BoxIntersections(box2_planes, box2_center, box1_intersect);
    const int box2_count = BoxIntersections(box1_planes, box1_center, box2_intersect);

    // If there are overlapping regions in Box2, remove any coplanar faces
    if (box2_count > 0) {
        // Identify if any triangles in Box2 are coplanar with Box1
        Keep tri2_keep[MAX_TRIS];
        for (int j = 0; j < MAX_TRIS; ++j) {
            // Initialize the valid faces to be true
            tri2_keep[j].keep = j < box2_count ? true : false;
        }
        for (int b = 0; b < box1_count; ++b) {
            const bool is_coplanar = IsCoplanarTriTri(box1_intersect[b], box2_intersect[b]);
            const float area = FaceArea(box1_intersect[b]);
            if ((is_coplanar) && (area > aEpsilon)) {
                tri2_keep[b].keep = false;
            }
        }

        // Keep only the non-coplanar triangles in Box2 - add them to the Box1 triangles.
        for (int b = 0; b < box2_count; ++b) {
            if (tri2_keep[b].keep) {
                box1_intersect[box1_count] = box2_intersect[b];
                // box1_count will determine the total faces in the intersecting shape
                box1_count++;
            }
        }
    }

    // Initialize the vol and iou to 0.0 in case there are no triangles in the intersecting shape.
    float boxes_intersection = 0.0;
    float boxes_union = 0.0;

    // If there are triangles in the intersecting shape
    if (box1_count > 0) {
        // The intersecting shape is a polyhedron made up of the triangular faces that are all now in box1_intersect.
        // Calculate the polyhedron center
        const float3 poly_center = PolyhedronCenter(box1_intersect, box1_count);
        // Compute intersecting polyhedron volume and intersection
        boxes_intersection = BoxVolume(box1_intersect, poly_center, box1_count);
        boxes_union = box1_vol + box2_vol - boxes_intersection;
    }
    return std::make_tuple(boxes_intersection, boxes_union);
}