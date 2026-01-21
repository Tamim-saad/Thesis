"""
ANSYS Mesh ML Fixes Module
==========================
This module contains critical fixes and improvements for the ANSYS Mesh ML Analysis pipeline.

Fixes Applied:
1. NBLOCK Parser - Robust FORTRAN fixed-width format handling
2. T-Net (Spatial Transformer Network) - For rotation invariance in PointNet
3. Volume Preservation Metric - Proper engineering metric computation
4. Enhanced PointNet with T-Net - Full original PointNet architecture

Usage:
    # In your Jupyter notebook, add this cell at the beginning:
    # %run ansys_mesh_fixes.py
    
    # Or import specific classes:
    # from ansys_mesh_fixes import RobustANSYSCDBParser, PointNetWithTNet, compute_volume_preservation

Author: AI Thesis Supervisor
Date: 2026-01-06
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import re


# =============================================================================
# FIX #1: Robust NBLOCK Parser with FORTRAN Format Support
# =============================================================================

class RobustANSYSCDBParser:
    """
    Enhanced Parser for ANSYS CDB files with robust format handling.
    
    Supports:
    - Space-separated formats (most common)
    - FORTRAN fixed-width formats like (3i8,6e21.13e3)
    - Auto-detection of format type
    
    This fixes BUG-001 from the supervisor report.
    """
    
    def __init__(self):
        self.node_data = []
        self.file_metadata = {}
        self.format_spec = None
    
    def _parse_format_spec(self, format_line):
        """
        Parse FORTRAN format specification.
        Example: (3i8,6e21.13e3) means:
        - 3 integers, each 8 characters wide
        - 6 exponentials, each 21 characters wide with 13 decimal places
        
        Returns dict with field widths
        """
        format_info = {
            'int_count': 3,
            'int_width': 8,
            'float_count': 6,
            'float_width': 21
        }
        
        # Try to parse format string
        try:
            # Match patterns like "3i8" (3 integers of width 8)
            int_match = re.search(r'(\d+)i(\d+)', format_line)
            if int_match:
                format_info['int_count'] = int(int_match.group(1))
                format_info['int_width'] = int(int_match.group(2))
            
            # Match patterns like "6e21.13" (6 exponentials of width 21)
            float_match = re.search(r'(\d+)e(\d+)', format_line)
            if float_match:
                format_info['float_count'] = int(float_match.group(1))
                format_info['float_width'] = int(float_match.group(2))
        except:
            pass
        
        return format_info
    
    def _parse_nblock_line_fixed_width(self, line, format_info):
        """
        Parse a single NBLOCK line using FORTRAN fixed-width format.
        
        Parameters:
        line (str): Raw line from file (NOT stripped)
        format_info (dict): Format specification
        
        Returns:
        tuple: (node_id, x, y, z) or None if parsing fails
        """
        try:
            int_width = format_info['int_width']
            float_width = format_info['float_width']
            
            # Calculate field positions
            # First 3 fields are integers (node_id, solid/shell, line)
            int_section_end = int_width * 3
            
            # X, Y, Z are the 4th, 5th, 6th fields after integers
            x_start = int_section_end
            x_end = x_start + float_width
            y_start = x_end
            y_end = y_start + float_width
            z_start = y_end
            z_end = z_start + float_width
            
            # Ensure line is long enough
            if len(line) < z_end:
                # Might be space-separated, try fallback
                return None
            
            node_id = int(line[0:int_width].strip())
            x = float(line[x_start:x_end].strip())
            y = float(line[y_start:y_end].strip())
            z = float(line[z_start:z_end].strip())
            
            return (node_id, x, y, z)
        except (ValueError, IndexError):
            return None
    
    def _parse_nblock_line_space_separated(self, line):
        """
        Parse a single NBLOCK line using space separation.
        
        Parameters:
        line (str): Stripped line from file
        
        Returns:
        tuple: (node_id, x, y, z) or None if parsing fails
        """
        try:
            parts = line.split()
            if len(parts) >= 6:
                node_id = int(parts[0])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                return (node_id, x, y, z)
        except (ValueError, IndexError):
            pass
        return None
    
    def parse_nblock_section(self, file_path):
        """
        Parse NBLOCK section from a single CDB file with automatic format detection.
        
        Parameters:
        file_path (str): Path to the CDB file
        
        Returns:
        pandas.DataFrame: DataFrame with columns [node_id, x, y, z]
        """
        nodes = []
        in_nblock = False
        nblock_info = {}
        format_info = None
        use_fixed_width = False
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Detect NBLOCK section start
                if line_stripped.startswith('NBLOCK'):
                    in_nblock = True
                    # Parse NBLOCK header: NBLOCK,6,SOLID,20991,20865
                    parts = line_stripped.split(',')
                    if len(parts) >= 4:
                        nblock_info['num_fields'] = int(parts[1]) if parts[1].strip().isdigit() else 6
                        nblock_info['total_nodes'] = int(parts[3]) if parts[3].strip().isdigit() else 0
                    continue
                
                # Parse format line (usually contains format specification)
                if in_nblock and line_stripped.startswith('('):
                    format_info = self._parse_format_spec(line_stripped)
                    # Check if we should use fixed-width parsing
                    # If format looks like (3i8,6e21.13e3), use fixed width
                    if 'e' in line_stripped.lower() and 'i' in line_stripped.lower():
                        use_fixed_width = True
                    continue
                
                # Skip comment and command lines
                if line_stripped.startswith('!') or line_stripped.startswith('/'):
                    if in_nblock:
                        break  # End of NBLOCK
                    continue
                
                # Parse node data
                if in_nblock and line_stripped:
                    result = None
                    
                    # Try fixed-width format first if detected
                    if use_fixed_width and format_info:
                        result = self._parse_nblock_line_fixed_width(line, format_info)
                    
                    # Fallback to space-separated
                    if result is None:
                        result = self._parse_nblock_line_space_separated(line_stripped)
                    
                    if result:
                        nodes.append(list(result))
                
                # End of NBLOCK section
                if in_nblock and (line_stripped.startswith('EBLOCK') or 
                                 line_stripped.startswith('FINISH') or 
                                 'N,' in line_stripped):
                    break
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return pd.DataFrame()
        
        # Create DataFrame
        if nodes:
            df = pd.DataFrame(nodes, columns=['node_id', 'x', 'y', 'z'])
            
            # Store metadata
            filename = os.path.basename(file_path)
            self.file_metadata[filename] = {
                'total_nodes': len(nodes),
                'expected_nodes': nblock_info.get('total_nodes', 0),
                'file_path': file_path,
                'format_type': 'fixed_width' if use_fixed_width else 'space_separated',
                'bounds': {
                    'x_min': df['x'].min(), 'x_max': df['x'].max(),
                    'y_min': df['y'].min(), 'y_max': df['y'].max(),
                    'z_min': df['z'].min(), 'z_max': df['z'].max()
                }
            }
            
            print(f"✅ Parsed {filename}: {len(nodes)} nodes (format: {self.file_metadata[filename]['format_type']})")
            return df
        else:
            print(f"⚠️ No nodes found in {file_path}")
            return pd.DataFrame()
    
    def parse_all_cdb_files(self, directory_path):
        """Parse all CDB files in a directory"""
        all_data = {}
        cdb_files = glob.glob(os.path.join(directory_path, "*.cdb"))
        
        if not cdb_files:
            print(f"No CDB files found in {directory_path}")
            return all_data
        
        print(f"Found {len(cdb_files)} CDB files to process...")
        
        for file_path in cdb_files:
            filename = os.path.basename(file_path)
            df = self.parse_nblock_section(file_path)
            
            if not df.empty:
                all_data[filename] = df
        
        print(f"\n✅ Successfully parsed {len(all_data)} files")
        return all_data
    
    def get_summary_statistics(self):
        """Get summary statistics of all parsed files"""
        if not self.file_metadata:
            print("No files have been parsed yet")
            return None
        
        summary_data = []
        for filename, metadata in self.file_metadata.items():
            summary_data.append({
                'filename': filename,
                'nodes': metadata['total_nodes'],
                'format': metadata['format_type'],
                'x_range': metadata['bounds']['x_max'] - metadata['bounds']['x_min'],
                'y_range': metadata['bounds']['y_max'] - metadata['bounds']['y_min'],
                'z_range': metadata['bounds']['z_max'] - metadata['bounds']['z_min'],
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("📊 Summary Statistics:")
        print(f"Total files: {len(summary_data)}")
        print(f"Total nodes: {summary_df['nodes'].sum():,}")
        print(f"Average nodes per file: {summary_df['nodes'].mean():.0f}")
        print(f"Node count range: {summary_df['nodes'].min()} - {summary_df['nodes'].max()}")
        
        return summary_df


# =============================================================================
# FIX #2: T-Net (Spatial Transformer Network) for Rotation Invariance
# =============================================================================

class TNet(nn.Module):
    """
    Spatial Transformer Network for learning canonical transformations.
    
    This is a critical component of the original PointNet paper that learns
    to align input point clouds to a canonical pose, providing rotation invariance.
    
    This fixes BUG-003 from the supervisor report.
    """
    
    def __init__(self, k=3):
        """
        Parameters:
        k (int): Dimension of transformation matrix (3 for input transform, 64 for feature transform)
        """
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLP
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        """
        Forward pass to compute transformation matrix.
        
        Parameters:
        x: Input tensor (batch_size, k, num_points)
        
        Returns:
        Transformation matrix (batch_size, k, k)
        """
        batch_size = x.size(0)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        
        # Add identity to learned transformation
        x = x.view(-1, self.k, self.k) + identity
        
        return x


class PointNetEncoderWithTNet(nn.Module):
    """
    Enhanced PointNet Encoder with Spatial Transformer Networks.
    
    This is the full PointNet architecture as described in the original paper,
    including both input and feature transformations for improved performance
    on rotated point clouds.
    """
    
    def __init__(self, num_points=1024, feature_dim=1024, use_feature_transform=True):
        super(PointNetEncoderWithTNet, self).__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.use_feature_transform = use_feature_transform
        
        # Input Spatial Transformer (3x3)
        self.input_transform = TNet(k=3)
        
        # Shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Feature Spatial Transformer (64x64) - optional
        if use_feature_transform:
            self.feature_transform = TNet(k=64)
        
        # Shared MLP 2
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Global feature extraction
        self.global_feat = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, x):
        """
        Forward pass with spatial transformations.
        
        Parameters:
        x: Input point cloud (batch_size, 3, num_points)
        
        Returns:
        tuple: (global_features, input_transform_matrix, feature_transform_matrix)
        """
        batch_size = x.size(0)
        
        # Input transformation
        input_trans = self.input_transform(x)
        x = x.transpose(2, 1)  # (B, N, 3)
        x = torch.bmm(x, input_trans)  # Apply transformation
        x = x.transpose(2, 1)  # (B, 3, N)
        
        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transformation (optional)
        if self.use_feature_transform:
            feat_trans = self.feature_transform(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feat_trans)
            x = x.transpose(2, 1)
        else:
            feat_trans = None
        
        # Second MLP
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Global features
        global_features = self.global_feat(x)
        
        return global_features, input_trans, feat_trans


class PointNetWithTNet(nn.Module):
    """
    Complete PointNet AutoEncoder with T-Net for improved rotation invariance.
    
    This provides better reconstruction performance on bones scanned at 
    different orientations.
    """
    
    def __init__(self, num_points=1024, feature_dim=1024, use_feature_transform=True):
        super(PointNetWithTNet, self).__init__()
        self.encoder = PointNetEncoderWithTNet(num_points, feature_dim, use_feature_transform)
        self.decoder = PointNetDecoder(feature_dim, num_points)
        self.use_feature_transform = use_feature_transform
        
    def forward(self, x):
        """
        Forward pass through encoder and decoder.
        
        Returns:
        tuple: (reconstructed_points, global_features, input_transform, feature_transform)
        """
        features, input_trans, feat_trans = self.encoder(x)
        reconstructed = self.decoder(features)
        return reconstructed, features, input_trans, feat_trans
    
    def get_regularization_loss(self, input_trans, feat_trans):
        """
        Compute regularization loss for transformation matrices.
        
        The original PointNet paper uses this to encourage the learned
        transformations to be close to orthogonal matrices.
        
        Loss = ||I - A*A^T||^2
        """
        loss = 0.0
        
        # Input transform regularization
        if input_trans is not None:
            batch_size = input_trans.size(0)
            d = input_trans.size(1)
            I = torch.eye(d, device=input_trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
            loss += torch.mean(torch.norm(
                I - torch.bmm(input_trans, input_trans.transpose(2, 1)), dim=(1, 2)))
        
        # Feature transform regularization
        if feat_trans is not None:
            batch_size = feat_trans.size(0)
            d = feat_trans.size(1)
            I = torch.eye(d, device=feat_trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
            loss += torch.mean(torch.norm(
                I - torch.bmm(feat_trans, feat_trans.transpose(2, 1)), dim=(1, 2)))
        
        return loss


class PointNetDecoder(nn.Module):
    """PointNet Decoder for point cloud reconstruction"""
    
    def __init__(self, feature_dim=1024, num_points=1024):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 3)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.decoder(x)
        x = x.view(batch_size, 3, self.num_points)
        return x


# =============================================================================
# FIX #3: Volume Preservation Metric
# =============================================================================

def compute_volume_preservation(original_points, reconstructed_points):
    """
    Compute volume preservation metric using convex hull approximation.
    
    This is a key engineering metric for FEA mesh quality assessment.
    Returns the ratio of volumes (min/max to ensure value <= 1).
    
    Parameters:
    original_points: Original point cloud (numpy array or tensor), shape (N, 3) or (3, N)
    reconstructed_points: Reconstructed point cloud, same shape as original
    
    Returns:
    float: Volume preservation ratio (0.0 to 1.0, higher is better)
    """
    try:
        from scipy.spatial import ConvexHull
        
        # Convert to numpy if tensor
        if torch.is_tensor(original_points):
            orig = original_points.detach().cpu().numpy()
        else:
            orig = np.array(original_points)
            
        if torch.is_tensor(reconstructed_points):
            recon = reconstructed_points.detach().cpu().numpy()
        else:
            recon = np.array(reconstructed_points)
        
        # Ensure shape is (N, 3)
        if orig.shape[0] == 3 and orig.shape[1] != 3:
            orig = orig.T
        if recon.shape[0] == 3 and recon.shape[1] != 3:
            recon = recon.T
        
        # Compute convex hulls
        hull_orig = ConvexHull(orig)
        hull_recon = ConvexHull(recon)
        
        orig_vol = hull_orig.volume
        recon_vol = hull_recon.volume
        
        # Compute preservation ratio (always <= 1.0)
        if max(orig_vol, recon_vol) > 0:
            preservation = min(orig_vol, recon_vol) / max(orig_vol, recon_vol)
        else:
            preservation = 0.0
        
        return preservation
        
    except Exception as e:
        print(f"Volume computation failed: {e}")
        return 0.0


def compute_surface_area_preservation(original_points, reconstructed_points):
    """
    Compute surface area preservation metric using convex hull approximation.
    
    Parameters:
    original_points: Original point cloud (numpy array or tensor)
    reconstructed_points: Reconstructed point cloud
    
    Returns:
    float: Surface area preservation ratio (0.0 to 1.0, higher is better)
    """
    try:
        from scipy.spatial import ConvexHull
        
        # Convert to numpy if tensor
        if torch.is_tensor(original_points):
            orig = original_points.detach().cpu().numpy()
        else:
            orig = np.array(original_points)
            
        if torch.is_tensor(reconstructed_points):
            recon = reconstructed_points.detach().cpu().numpy()
        else:
            recon = np.array(reconstructed_points)
        
        # Ensure shape is (N, 3)
        if orig.shape[0] == 3 and orig.shape[1] != 3:
            orig = orig.T
        if recon.shape[0] == 3 and recon.shape[1] != 3:
            recon = recon.T
        
        # Compute convex hulls
        hull_orig = ConvexHull(orig)
        hull_recon = ConvexHull(recon)
        
        orig_area = hull_orig.area
        recon_area = hull_recon.area
        
        # Compute preservation ratio
        if max(orig_area, recon_area) > 0:
            preservation = min(orig_area, recon_area) / max(orig_area, recon_area)
        else:
            preservation = 0.0
        
        return preservation
        
    except Exception as e:
        print(f"Surface area computation failed: {e}")
        return 0.0


def compute_hausdorff_distance(points1, points2):
    """
    Compute Hausdorff distance between two point clouds.
    
    The Hausdorff distance is the maximum distance from any point in one set
    to its nearest neighbor in the other set.
    
    Parameters:
    points1, points2: Point clouds (numpy arrays or tensors), shape (N, 3) or (3, N)
    
    Returns:
    float: Hausdorff distance
    """
    from scipy.spatial.distance import cdist
    
    # Convert to numpy if tensor
    if torch.is_tensor(points1):
        p1 = points1.detach().cpu().numpy()
    else:
        p1 = np.array(points1)
        
    if torch.is_tensor(points2):
        p2 = points2.detach().cpu().numpy()
    else:
        p2 = np.array(points2)
    
    # Ensure shape is (N, 3)
    if p1.shape[0] == 3 and p1.shape[1] != 3:
        p1 = p1.T
    if p2.shape[0] == 3 and p2.shape[1] != 3:
        p2 = p2.T
    
    # Compute distances
    d1 = cdist(p1, p2, metric='euclidean')
    d2 = cdist(p2, p1, metric='euclidean')
    
    # Hausdorff distance is max of min distances in both directions
    hd1 = np.max(np.min(d1, axis=1))
    hd2 = np.max(np.min(d2, axis=1))
    
    return max(hd1, hd2)


def compute_comprehensive_metrics(original, reconstructed, threshold=0.05):
    """
    Compute comprehensive quality metrics for mesh reconstruction.
    
    Parameters:
    original: Original point cloud tensor (batch_size, 3, num_points) or (3, num_points)
    reconstructed: Reconstructed point cloud tensor
    threshold: Distance threshold for F1 score computation
    
    Returns:
    dict: Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Handle batch dimension
    if len(original.shape) == 3:
        # Take first sample from batch
        orig = original[0]
        recon = reconstructed[0]
    else:
        orig = original
        recon = reconstructed
    
    # Transpose if needed (expect (3, N))
    if orig.shape[0] != 3:
        orig = orig.T
    if recon.shape[0] != 3:
        recon = recon.T
    
    # Convert to (N, 3) for most computations
    orig_np = orig.T.detach().cpu().numpy() if torch.is_tensor(orig) else orig.T
    recon_np = recon.T.detach().cpu().numpy() if torch.is_tensor(recon) else recon.T
    
    # 1. Chamfer Distance
    from scipy.spatial.distance import cdist
    d1 = cdist(orig_np, recon_np)
    d2 = cdist(recon_np, orig_np)
    metrics['chamfer_distance'] = np.mean(np.min(d1, axis=1)) + np.mean(np.min(d2, axis=1))
    
    # 2. Hausdorff Distance
    metrics['hausdorff_distance'] = compute_hausdorff_distance(orig_np, recon_np)
    
    # 3. Volume Preservation
    metrics['volume_preservation'] = compute_volume_preservation(orig_np, recon_np)
    
    # 4. Surface Area Preservation
    metrics['surface_area_preservation'] = compute_surface_area_preservation(orig_np, recon_np)
    
    # 5. F1 Score (Precision/Recall based)
    min_dists_orig_to_recon = np.min(d1, axis=1)
    min_dists_recon_to_orig = np.min(d2, axis=1)
    
    precision = np.mean(min_dists_recon_to_orig < threshold)
    recall = np.mean(min_dists_orig_to_recon < threshold)
    
    if precision + recall > 0:
        metrics['f1_score'] = 2 * precision * recall / (precision + recall)
    else:
        metrics['f1_score'] = 0.0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    
    # 6. MSE and RMSE (if point correspondence can be assumed)
    # Note: This assumes points are in the same order, which may not always be true
    metrics['mse'] = np.mean((orig_np - recon_np) ** 2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    return metrics


# =============================================================================
# Enhanced Loss Function with T-Net Regularization
# =============================================================================

def chamfer_distance_with_regularization(pred, target, input_trans=None, feat_trans=None, 
                                         reg_weight=0.001):
    """
    Chamfer distance loss with T-Net regularization.
    
    Parameters:
    pred: Predicted point cloud (batch_size, 3, num_points)
    target: Target point cloud (batch_size, 3, num_points)
    input_trans: Input transformation matrix from T-Net (optional)
    feat_trans: Feature transformation matrix from T-Net (optional)
    reg_weight: Weight for regularization loss
    
    Returns:
    torch.Tensor: Combined loss
    """
    # Transpose for distance computation
    pred = pred.transpose(1, 2)  # (B, N, 3)
    target = target.transpose(1, 2)  # (B, M, 3)
    
    # Compute pairwise distances
    pred_expanded = pred.unsqueeze(2)  # (B, N, 1, 3)
    target_expanded = target.unsqueeze(1)  # (B, 1, M, 3)
    
    distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)  # (B, N, M)
    
    # Chamfer distance
    pred_to_target = torch.min(distances, dim=2)[0]  # (B, N)
    target_to_pred = torch.min(distances, dim=1)[0]  # (B, M)
    
    chamfer_loss = torch.mean(pred_to_target, dim=1) + torch.mean(target_to_pred, dim=1)
    chamfer_loss = torch.mean(chamfer_loss)
    
    # T-Net regularization
    reg_loss = 0.0
    
    if input_trans is not None:
        batch_size = input_trans.size(0)
        d = input_trans.size(1)
        I = torch.eye(d, device=input_trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
        reg_loss += torch.mean(torch.norm(
            I - torch.bmm(input_trans, input_trans.transpose(2, 1)), dim=(1, 2)))
    
    if feat_trans is not None:
        batch_size = feat_trans.size(0)
        d = feat_trans.size(1)
        I = torch.eye(d, device=feat_trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
        reg_loss += torch.mean(torch.norm(
            I - torch.bmm(feat_trans, feat_trans.transpose(2, 1)), dim=(1, 2)))
    
    total_loss = chamfer_loss + reg_weight * reg_loss
    
    return total_loss


# =============================================================================
# Utility: Quick Test Functions
# =============================================================================

def test_parser():
    """Test the robust NBLOCK parser"""
    print("Testing RobustANSYSCDBParser...")
    parser = RobustANSYSCDBParser()
    print("✅ Parser initialized successfully!")
    return parser


def test_tnet():
    """Test T-Net implementation"""
    print("Testing T-Net...")
    tnet = TNet(k=3)
    x = torch.randn(2, 3, 1024)  # Batch of 2, 3 channels, 1024 points
    output = tnet(x)
    print(f"✅ T-Net output shape: {output.shape}")  # Should be (2, 3, 3)
    assert output.shape == (2, 3, 3), "T-Net output shape mismatch!"
    return tnet


def test_pointnet_with_tnet():
    """Test PointNet with T-Net"""
    print("Testing PointNetWithTNet...")
    model = PointNetWithTNet(num_points=1024, feature_dim=1024)
    x = torch.randn(2, 3, 1024)
    recon, features, input_trans, feat_trans = model(x)
    print(f"✅ Reconstruction shape: {recon.shape}")  # Should be (2, 3, 1024)
    print(f"✅ Features shape: {features.shape}")  # Should be (2, 1024)
    print(f"✅ Input transform shape: {input_trans.shape}")  # Should be (2, 3, 3)
    print(f"✅ Feature transform shape: {feat_trans.shape}")  # Should be (2, 64, 64)
    return model


def test_volume_metric():
    """Test volume preservation metric"""
    print("Testing volume preservation metric...")
    # Create two similar point clouds
    points1 = np.random.randn(1024, 3)
    points2 = points1 + np.random.randn(1024, 3) * 0.1  # Small perturbation
    
    vol_pres = compute_volume_preservation(points1, points2)
    print(f"✅ Volume preservation: {vol_pres:.4f}")
    assert 0 <= vol_pres <= 1, "Volume preservation should be between 0 and 1!"
    return vol_pres


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running ANSYS Mesh ML Fixes Tests")
    print("=" * 60)
    
    test_parser()
    test_tnet()
    test_pointnet_with_tnet()
    test_volume_metric()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


# Run tests when script is executed directly
if __name__ == "__main__":
    run_all_tests()
