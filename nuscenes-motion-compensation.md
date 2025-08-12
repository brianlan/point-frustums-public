# Solution from @enrico-stauss

## Source Link
https://github.com/nutonomy/nuscenes-devkit/issues/902#issuecomment-1704380456

## Sentence by sentence explanation
"Yeah sure @nightrome, as you noted it comes down to the EGO velocity and the comparably low speed of rotation."
- Acknowledges that the issue is caused by the vehicle's movement (ego velocity) relative to the LiDAR sensor's rotation speed.

"Imagine for example a tree to the left with a clockwise spinning sensor as indicated in the sketch below."
- Sets up a concrete example using a tree as the object being measured by a rotating LiDAR sensor.

"The point cloud is logged at timestamp t⁰, see now the point recorded at t⁻¹ which hit the tree."
- Explains that the final point cloud represents a snapshot at time t⁰, but contains points measured at earlier times (t⁻¹) when the sensor hit the tree.

"Given a stationary EGO, then the next point measured at t⁰ and an azimuthal difference of Δφ would measure the tree, too."
- If the vehicle weren't moving, the next measurement at the expected angular position would also hit the tree.

"But as the EGO has now moved, the point passes the tree and measures the background."
- Because the vehicle moved during the measurement time, the expected next measurement misses the tree and hits whatever is behind it.

"When assembling the range image by transforming XYZ coordinates to spherical coordinates the delta angle will not be δφ as expected but rather Δφ̃."
- When converting 3D points to range image format, the angular spacing between points is not uniform as expected, but distorted.

"That will be visible as the bespoken shadowing on object's corners."
- This angular distortion manifests as shadows (missing points) at the edges of objects in range images.

"A similar thing happens on the other side of the measurement."
- The same effect occurs on both sides of objects, creating symmetric artifacts.

"The effect will be negligible for forward/backward measurements as the EGO mostly moves forward."
- Since vehicles primarily move forward, this effect is minimal for objects directly ahead or behind.

"The shadows are NOT an issue with the dataset but rather something intrinsic to the measurement."
- This is not a calibration error but a fundamental characteristic of how rotating LiDAR works on moving vehicles.

"It might be worth consideration when constructing range images though."
- Suggests this should be accounted for in range image algorithms.

## The Core Algorithm Proposed

Motion Distortion Correction Formula:
Δy = (φ/2π) × (s/20) × v_y

Where:
- φ = azimuth angle of the point
- s = number of sweeps
- 20 = LiDAR rotation frequency (20Hz)
- v_y = ego vehicle's lateral velocity

How it works:
1. For each LiDAR point, calculate how much the vehicle moved during the time it took to measure that point
2. Shift the point's y-coordinate by the calculated offset to "undo" the motion distortion
3. This creates more uniform angular spacing in the resulting range image

Why this algorithm:
- Problem: Vehicle motion during LiDAR sweep causes non-uniform angular sampling, creating shadows and peaks in range images
- Root cause: Points measured at different times during the sweep experience different amounts of vehicle motion
- Solution: Compensate by shifting points back to where they would have been if measured from a stationary vehicle
- Result: More uniform point density in range images, reducing artifacts that can hurt CNN-based range imaging models

The algorithm essentially "unrolls" the motion distortion to create cleaner range images suitable for deep learning applications, though the author notes this trades off geometric accuracy for data uniformity.

# Feasible Enhancements

## 1. Full 3D Motion Correction (High Impact, Low Complexity)

  Current: Only corrects forward motion (Y-axis)
  Enhanced:
  ```python
  # Full 3D correction
  delta_x = (phi / (2 * pi)) * (s / f_lidar) * v_x  # Lateral motion
  delta_y = (phi / (2 * pi)) * (s / f_lidar) * v_y  # Forward motion (original)
  delta_z = (phi / (2 * pi)) * (s / f_lidar) * v_z  # Vertical motion

  corrected_points = original_points - np.array([delta_x, delta_y, delta_z])
  ```

## 2. Angular Velocity Correction (High Impact, Medium Complexity)

  Add rotational motion effects:
  ```python
  def motion_correction_with_rotation(points, phi, ego_velocity, ego_angular_velocity, s, f_lidar):
      # Time elapsed for this azimuth angle
      dt = (phi / (2 * pi)) * (s / f_lidar)

      # Translational correction
      translation_correction = dt * ego_velocity

      # Rotational correction: delta = omega × r × dt
      rotation_correction = dt * np.cross(ego_angular_velocity, points, axis=-1)

      # Total correction
      total_correction = translation_correction + rotation_correction
      return points - total_correction
  ```