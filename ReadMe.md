# Point Frustums
This repository contains the refactored PyTorch [9] code to my master's thesis on the topic '**Leveraging Inherent Properties of Point Clouds by Spherical Coordinate Voxelization**'.

The core idea is to discretize the field of view into frustums centered around the origin, and encode each frustum. This yields a featuremap similar to the one in camera or range imaging models.
One of the main contributions of this thesis is the **output encoding in spherical coordinates**.

Models used as reference are, among others `PointPillars` [3], `FCOS` [8] and `FCOS-LiDAR` [6]. It should be noted that [12] takes a similar discretization approach but then maps features back to BEV Cartesian coordinates.

## Installation
The project is developed with Python 3.10 and can be installed via pip within the locally existing repository via:
```bash
  pip install -e .
```
or directly from GitHub via:
```bash
  pip install git+https://github.com/enrico-stauss/point-frustums-public.git
```
For development or contribution, it is recommended to install the project in editable mode:
```bash
  pip install -e ".[dev]"
```

### Requirements


## Basic Structure
The migrated code fully integrates with PyTorch Lightning. Core components are briefly described below.

### NuScenes Dataloader
A PyTorch `Dataset` that internally uses the API provided by the `nuscenes-devkit` [2]. 
The NuScenes dataloader requires the NuScenes dataset as well as the CAN bus data to be downloaded and unpacked as documented by the `nuscenes-devkit` [2]. You can either specify `--data.data_root=<NUSCENES/ROOT/DIR>` or wait to be prompted the first time you invoke a training. The data_root will be cached locally for the specific dataset and version for subsequent use.
The implemented dataset can be configured to `load_velocity` (optionally with `load_can`) to provide EGO velocities for training and inference. 

<details>
<summary>Format of loaded samples</summary>
```
> lidar:
>   SENSOR_ID: torch.Tensor | np.ndarray
> camera:
>   SENSOR_ID: torch.Tensor | np.ndarray
> metadata:
>   rotation: list  # quaternion
>   translation: list  # xyz
>   location: str
>   sample_token: str
>   scene_description: str
>   scene_name: str
>   timestamp: int  # unix timestamp in microseconds
>   vehicle: str
>   velocity: np.ndarray  # in m/s
>   augmentations:  # dict containing a log of augmentations applied to the sample
>     Normalize: {}
>     ...
>   SENSOR_ID:
>     modality: str
>     rotation: list  # quaternion
>     translation: list  # xyz
>   ...
> targets: point_frustums.utils.targets.Targets
>   ...
```
</details>
A `LightningDataModule` configures and conveniently provides `DataLoader`s for the `LightningCLI`.

#### Point Cloud Data
PCs are loaded with channels {`x`, `y`, `z`, `intensity`, `timestamp`, `r`, `theta`, `phi`} where the timestamp is interpolated based on the azimuth and the rotation frequency. 
The batch of PCs is stored in a list of tensors to respect the irregular format.

#### Camera Data
Camera images are loaded and stacked on dimension zero.

#### Augmentations
An augmentation framework is provided that enables easily implementing physically consistent augmentations for the multimodal use-case.
Augmentations are implemented as objects with methods `lidar`, `camera`, `radar` and `targets` that are then applied to the respective modality.
To implement RandomFlip for example, it convenient to flip the data and targets jointly.

### The Point Frustums Model
The detection module consists of several components:
- The `torch.model` that provides raw, encoded output,
- The training target assignment and loss functions,
- The output postprocessing for the inference case and
- The evaluator

which are decoupled but conveniently grouped in a `LightningModule`.

### Miscellaneous Noteworthy Details
- Training target assignment by solving an Optimal Transport Problem [4, 6]
- 3D IoU (not differentiable, CPU and CUDA implementation, copied from PyTorch3d [7] and modified to pairwise computation)
- 3D NMS (not differentiable, CPU implementation, based on 3D IoU)
- Feature Pyramid Network [10] with circular padding on the horizontal axis 
- 6D continuous orientation encoding [7, 11]
- Integration with the `nuscenes-devkit` [2] evaluation tool
- The Vario-Focal Loss [5] applied to the regression branch
- The transformer based frustum feature encoder which resembles the one described in HVNet [13, 14]

## TODO
- [ ] Add output encoding to the documentation
- [ ] Finalize `torchmetrics` implementation of the NDS
- [x] Implement CUDA kernel for 3D NMS
- [x] Migrate remaining augmentations
- [ ] Provide results
- [x] Support loading data and targets in EGO COOS
- [ ] Implement Camera to LiDAR fusion 
- [ ] Implement PointPillars and FCOS-LiDAR in the framework


## References
- [1] Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., & Beijbom, O. (2019). nuScenes: A multimodal dataset for autonomous driving
- [2] Motional. (2021). nuscenes-devkit. https://github.com/nutonomy/nuscenes-devkit
- [3] Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2018). PointPillars: Fast Encoders for Object Detection from Point Clouds
- [4] Ge, Z., Liu, S., Li, Z., Yoshie, O., & Sun, J. (2021). OTA: Optimal Transport Assignment for Object Detection.
- [5] Zhang, H., Wang, Y., Dayoub, F., & Sünderhauf, N. (2020). VarifocalNet: An IoU-aware Dense Object Detector.
- [6] Tian, Z., Chu, X., Wang, X., Wei, X., & Shen, C. (2022). Fully Convolutional One-Stage 3D Object Detection on LiDAR Range Images.
- [7] Ravi, N., Reizenstein, J., Novotny, D., Gordon, T., Lo, W.-Y., Johnson, J., & Gkioxari, G. (2020). Accelerating 3D Deep Learning with PyTorch3D.
- [8] Tian, Z., Shen, C., Chen, H., & He, T. (2019). FCOS: Fully Convolutional One-Stage Object Detection.
- [9] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
- [10] Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2016). Feature Pyramid Networks for Object Detection.
- [11] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2018). On the Continuity of Rotation Representations in Neural Networks.
- [12] Zhou, Y., Sun, P., Zhang, Y., Anguelov, D., Gao, J., Ouyang, T., Guo, J., Ngiam, J., & Vasudevan, V. (2019). End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds.
- [13] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need.
- [14] Ye, M., Xu, S., & Cao, T. (2020). HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection.