//
// Created by Enrico Stauss on 15.03.24.
//

#include "utils/vec3.h"
#include "iou_box3d/iou_utils.h"


int N_FLOATS_PER_TRIANGLE = NUM_TRIS * 3 * 3
int N_FLOATS_PER_PLANE = NUM_PLANES* 4 * 3


// This is not a great idea as the face_verts vectors will not be placed contiguously 
// in memory but only the pointer. I'd need to change the face_verts type to an array
// but this requires more substantial changes to the initial implememtation.
struct BoxParameters {
  vec3<float> center;
  face_verts box_tris(N_FLOATS_PER_TRIANGLE);
  face_verts box_planes(N_FLOATS_PER_PLANE);
  float volume;
};
