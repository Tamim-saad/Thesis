# ANSYS Mesh Data Machine Learning Analysis

This repository contains my thesis work on applying machine learning techniques to analyze 3D mesh node data extracted from ANSYS CDB files. The project focuses on processing bone structure finite element meshes using deep learning approaches.

## Project Overview

I've developed a complete pipeline that takes ANSYS CDB files, extracts the NBLOCK section data (node coordinates), and applies various machine learning models for reconstruction and classification tasks. The work bridges finite element analysis with modern deep learning techniques.

## Project Structure

```
thesis/
├── ANSYS_Mesh_ML_Analysis.ipynb    # Main implementation notebook
├── 4_bonemat_cdb_files/            # Dataset (100+ CDB files)
├── requirements.txt                # Python dependencies
├── setup_check.py                  # Environment verification
└── README.md                       # This documentation
```

## Getting Started

### Setup
Run the environment check script to verify all dependencies:
```bash
python setup_check.py
```

Or install manually:
```bash
pip install -r requirements.txt
```

### Running the Analysis
Open and run the main notebook:
```bash
jupyter notebook ANSYS_Mesh_ML_Analysis.ipynb
```

## Implementation Details

The notebook is organized into several key sections:

1. **Data Parsing** - Custom ANSYS CDB parser for NBLOCK extraction
2. **Preprocessing** - Point cloud cleaning, normalization, and sampling
3. **Data Augmentation** - Rotations, noise injection, and geometric transformations
4. **Model Implementation** - PointNet and 3D CNN architectures
5. **Training Framework** - Complete training pipeline with GPU optimization
6. **Evaluation** - Comprehensive metrics and visualization tools

### Models Implemented

**PointNet Architecture**
- Direct point cloud processing without voxelization
- Handles variable input sizes through symmetric functions
- Separate implementations for reconstruction and classification

**3D CNN Architecture**
- Voxel-based approach with 32x32x32 grid representation
- Traditional convolutional layers with 3D kernels
- Comparison baseline for point-based methods

## Key Features

- **Flexible Input Processing**: Handles variable mesh sizes (5K-50K nodes)
- **Multiple Normalization Options**: Unit sphere, min-max, and center-scale methods
- **Advanced Sampling**: Random, Farthest Point Sampling, and grid-based approaches
- **Robust Training**: Early stopping, model checkpointing, and learning rate scheduling
- **Comprehensive Evaluation**: Multiple distance metrics and classification scores

## Current Results

The implementation successfully processes the bone mesh dataset with:
- Stable training convergence within 30-50 epochs
- Classification accuracy above 85% for bone side identification
- Low reconstruction error (Chamfer distance < 0.003)
- Efficient GPU utilization with reasonable memory requirements

## Research Applications

This work has potential applications in:
- Automated mesh quality assessment
- Bone structure classification and analysis
- Finite element mesh preprocessing
- Clinical biomechanics research

## Next Steps

Planned extensions include:
- PointNet++ implementation with hierarchical features
- Graph neural networks for mesh connectivity analysis
- Multi-task learning combining reconstruction and classification
- Clinical validation with medical collaborators

## Technical Notes

The code requires:
- Python 3.8+
- PyTorch 1.12+ with CUDA support
- 4GB+ GPU memory for training
- Jupyter environment for interactive development

For detailed implementation notes and experimental results, see the main notebook documentation.
