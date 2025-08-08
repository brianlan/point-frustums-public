# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This repository implements Point Frustums, a 3D object detection approach for point clouds using spherical coordinate voxelization. The method discretizes the field of view into frustums centered around the origin and encodes each frustum to create featuremaps similar to camera/range imaging models. The core contribution is output encoding in spherical coordinates.

## Development Commands

### Installation
```bash
# Standard installation
pip install -e .

# Development installation with dev dependencies
pip install -e ".[dev]"
```

### Training & Inference
```bash
# Run training with PyTorch Lightning CLI
python -m point_frustums fit --config configs/point_frustums.yaml

# Alternative entry point
point-frustums fit --config configs/point_frustums.yaml
```

### Code Quality
```bash
# Format code (line length: 120 characters)
black .

# Lint code
pylint point_frustums/

# Run tests
pytest test/
```

### Building C++/CUDA Extensions
The project includes custom CUDA operations for 3D IoU computation and NMS. These are built automatically during installation using `setup.py` with PyTorch's C++ extension system.

## Architecture Overview

### Core Components

**Model Architecture (point_frustums/models/)**
- `PointFrustumsModel`: Main detection model combining backbone, neck, and head
- `PointFrustumsBackbone`: Frustum encoder using transformer-based architecture (similar to HVNet)
- `PointFrustumsNeck`: Feature Pyramid Network with circular padding on horizontal axis
- `PointFrustumsHead`: Detection head with separate classification and regression branches

**Runtime System (point_frustums/models/base_runtime.py)**
- `Detection3DRuntime`: Abstract base class for PyTorch Lightning modules
- `PointFrustums`: Main Lightning module implementing training/validation loops, loss computation, and target assignment

**Data Pipeline (point_frustums/dataloaders/)**
- `NuScenes`: PyTorch Dataset for NuScenes with multimodal support (LiDAR, camera, radar)
- Supports both regular and streaming datasets via LitData
- Includes physically consistent augmentation framework

### Key Technical Details

**Coordinate Systems**
- Input: Point clouds in Cartesian coordinates with channels {x, y, z, intensity, timestamp, r, theta, phi}
- Processing: Spherical coordinate discretization into frustums
- Output: Detection boxes in original coordinate system

**Target Assignment**
- Uses Optimal Transport Problem (OTA) for assigning predictions to ground truth
- Combines classification cost (focal loss), IoU cost, and center distance cost
- Implemented using Sinkhorn algorithm with unbalanced transport

**Loss Functions**
- Classification: Sigmoid focal loss
- Regression: Smooth L1 loss for center, size, orientation components
- Quality estimation: Varifocal Loss (VFL) applied to regression branch
- 6D continuous orientation encoding (Zhou et al., 2018)

**Custom Operations (point_frustums/ops/)**
- 3D IoU computation (CPU and CUDA implementations, modified from PyTorch3D)
- 3D NMS (CPU implementation based on 3D IoU)
- Spherical coordinate convolutions

## Configuration System

The project uses YAML configuration files with PyTorch Lightning CLI:

**Main Config**: `configs/point_frustums.yaml`
- References dataset config: `configs/datasets/nuscenes.yaml`  
- References model config: `configs/models/point_frustums.yaml`

**Key Configuration Classes** (point_frustums/config_dataclasses/):
- `DatasetConfig`: Dataset parameters, augmentations, sensor configurations
- `PointFrustumsConfig`: Model architecture, target assignment, losses, predictions
- `TargetAssignment`: OTA parameters (alpha, beta, gamma coefficients, epsilon regularization)
- `Losses`: Loss function weights and activation schedules

## Dataset Setup

**NuScenes Dataset**:
- Requires NuScenes dataset and CAN bus data
- Set `data_root` parameter or will be prompted on first run
- Supports loading EGO velocities with CAN bus data
- Data root is cached locally for subsequent use

**Data Format**:
```python
{
    'lidar': {'LIDAR_TOP': tensor},      # Point clouds
    'camera': {'CAM_*': tensor},         # Camera images  
    'metadata': {                        # Sample metadata
        'rotation': list,                # quaternion
        'translation': list,             # xyz
        'velocity': ndarray,             # m/s (if load_velocity=True)
        'sample_token': str,
        # ... sensor-specific metadata
    },
    'targets': Targets                   # Ground truth annotations
}
```

## Testing

**Test Structure** (test/):
- Unit tests for geometry operations, augmentations, model components
- NDS metric validation with test cases
- Test data includes ground truth results for validation

**Key Test Areas**:
- Quaternion and rotation matrix operations
- Target assignment algorithm
- Frustum encoder functionality
- NuScenes Detection Score (NDS) computation

## Logging and Monitoring

**Supported Loggers**:
- TensorBoard: Custom scalar layouts for losses and NDS metrics
- Weights & Biases: Model watching, code logging, 3D visualizations

**Metrics**:
- NuScenes Detection Score (NDS) - primary evaluation metric
- Per-class Average Precision (AP)
- Target assignment statistics (missed targets, outflows)
- Featuremap-level assignment counts

## Development Notes

**Code Style**:
- Line length: 120 characters (Black formatter)
- Pylint configuration disables certain warnings for ML code patterns
- Type hints used throughout

**Dependencies**:
- PyTorch 2.2.1 (pinned version)
- PyTorch Lightning for training framework
- NuScenes DevKit for dataset API
- Custom C++/CUDA extensions require CUDA toolkit

**Memory Considerations**:
- Point clouds stored as list of tensors (irregular shapes)
- Frustum discretization creates large featuremaps
- Custom CUDA kernels for memory-efficient operations

## Common Debugging Areas

**Target Assignment Issues**:
- Check OTA parameters (alpha, beta, gamma) in config
- Monitor "TargetAssignments/Missed Targets" and "Number of Outflows" metrics
- Verify receptive field calculations for different FPN levels

**Coordinate System Problems**:
- Ensure consistent coordinate systems between input data and targets
- Check spherical<->Cartesian coordinate transformations
- Verify frustum boundary calculations

**Performance Issues**:
- CUDA operations may fall back to CPU if CUDA unavailable
- Large batch sizes may cause OOM due to irregular point cloud sizes
- NMS threshold and top-k parameters affect inference speed