//
// Created by Enrico Stauss on 15.03.24.
//

#include "iou_box3d/iou_utils.cuh"
#include "utils/float_math.cuh"

int N_FLOATS_PER_TRIANGLE = NUM_TRIS * 4 * 3;
int N_FLOATS_PER_PLANE = NUM_PLANES * 4 * 3;

// Here I can assume that the content of the struct will be placed contiguously in memory.
// If I then call reinterp_cast on a row of a tensor it should work out.
struct BoxParameters {
  float3 center;
  FaceVerts triangles[NUM_TRIS];
  FaceVerts planes[NUM_PLANES];
  float volume;
};
