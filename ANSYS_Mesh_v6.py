"""
ANSYS Mesh v6: ML-based FEA Mesh Quality Assessment for Bone Structures
========================================================================
Research Question: Can GNN learn bone-specific mesh quality patterns
beyond standard threshold-based metrics? Compared with PointNet baseline.

Pipeline:
  1. Parse ANSYS CDB files (NBLOCK + EBLOCK)
  2. Compute FEA mesh quality metrics per element
  3. Label elements/meshes by quality
  4. Build graph representation from mesh connectivity
  5. Train PointNet (baseline) + GNN/GAT (main) models
  6. Evaluate with K-Fold Cross Validation
  7. Visualize and analyze results

Author: Tamim Saad
"""

# ============================================================
# SECTION 0: SETUP (Run this cell first in Colab)
# ============================================================
"""
# Uncomment and run in Google Colab:
!pip install ansys-mapdl-reader pyvista vtk meshio -q
!apt-get install -y xvfb -q 2>/dev/null
!pip install pyvista[jupyter] -q
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, re, warnings, time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try importing pyansys and pyvista
USE_PYANSYS = False
try:
    from ansys.mapdl.reader import Archive
    USE_PYANSYS = True
    print("✅ ansys-mapdl-reader available")
except ImportError:
    print("⚠️ ansys-mapdl-reader not available, using custom parser")

USE_PYVISTA = False
try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    USE_PYVISTA = True
    print("✅ PyVista available")
except ImportError:
    print("⚠️ PyVista not available, using manual quality metrics")

# Seeds
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
CONFIG = {
    'data_dir': '/content/drive/MyDrive/Thesis/Data',  # Colab path
    'num_points': 1024,        # Points per mesh for PointNet
    'num_graph_nodes': 2048,   # Nodes per graph for GNN
    'batch_size': 4,
    'epochs': 100,
    'lr': 0.001,
    'k_folds': 5,
    'patience': 15,
    'gat_heads': 4,
    'gat_hidden': 64,
    'pointnet_hidden': 256,
    # Quality thresholds (standard FEA)
    'ar_good': 3.0,       # Aspect ratio < 3 = good
    'ar_poor': 10.0,      # Aspect ratio > 10 = poor
    'sj_good': 0.5,       # Scaled Jacobian > 0.5 = good
    'sj_poor': 0.2,       # Scaled Jacobian < 0.2 = poor
    'skew_good': 0.25,    # Skewness < 0.25 = good
    'skew_poor': 0.75,    # Skewness > 0.75 = poor
}

# ============================================================
# SECTION 2.5: DATA PREPROCESSING & CLEANING
# ============================================================
class MeshDataPreprocessor:
    """Clean, normalize, and preprocess mesh point clouds."""

    @staticmethod
    def clean_data(nodes_df):
        """Remove duplicates, NaN, and outliers from node data."""
        if nodes_df.empty:
            return nodes_df
        # Remove NaN
        cleaned = nodes_df.dropna(subset=['x', 'y', 'z']).copy()
        # Remove duplicates based on coordinates
        cleaned = cleaned.drop_duplicates(subset=['x', 'y', 'z'])
        # Remove statistical outliers (beyond 3 sigma)
        for col in ['x', 'y', 'z']:
            mean, std = cleaned[col].mean(), cleaned[col].std()
            if std > 0:
                cleaned = cleaned[abs(cleaned[col] - mean) <= 3 * std]
        return cleaned.reset_index(drop=True)

    @staticmethod
    def normalize_points(points):
        """Center and scale points to unit sphere."""
        centroid = points.mean(axis=0)
        centered = points - centroid
        max_dist = np.max(np.linalg.norm(centered, axis=1)) + 1e-8
        return centered / max_dist

    @staticmethod
    def sample_points(points, n_points, method='fps'):
        """Sample fixed number of points.

        Methods:
          - 'random': random uniform sampling
          - 'fps': farthest point sampling (better coverage)
        """
        n = len(points)
        if n == 0:
            return np.zeros((n_points, 3))
        if n <= n_points:
            idx = np.random.choice(n, n_points, replace=True)
            return points[idx]
        if method == 'fps':
            return MeshDataPreprocessor._farthest_point_sample(points, n_points)
        else:
            idx = np.random.choice(n, n_points, replace=False)
            return points[idx]

    @staticmethod
    def _farthest_point_sample(points, n_samples):
        """Farthest Point Sampling for better spatial coverage."""
        n = len(points)
        selected = np.zeros(n_samples, dtype=int)
        distances = np.full(n, np.inf)
        # Start from random point
        selected[0] = np.random.randint(n)
        for i in range(1, n_samples):
            last = points[selected[i-1]]
            dist = np.linalg.norm(points - last, axis=1)
            distances = np.minimum(distances, dist)
            selected[i] = np.argmax(distances)
        return points[selected]

    @staticmethod
    def compute_point_cloud_stats(all_data):
        """Compute detailed statistics for all meshes."""
        stats = []
        for fname, data in all_data.items():
            nodes = data['nodes']
            pts = nodes[['x', 'y', 'z']].values
            n_nodes = len(nodes)
            n_elements = len(data['elements'])
            x_range = pts[:, 0].max() - pts[:, 0].min() if n_nodes > 0 else 0
            y_range = pts[:, 1].max() - pts[:, 1].min() if n_nodes > 0 else 0
            z_range = pts[:, 2].max() - pts[:, 2].min() if n_nodes > 0 else 0

            # Estimate mesh density
            if n_nodes > 1:
                from scipy.spatial import cKDTree
                tree = cKDTree(pts[:min(n_nodes, 5000)])
                dists, _ = tree.query(pts[:min(n_nodes, 5000)], k=2)
                avg_nn_dist = np.mean(dists[:, 1])
            else:
                avg_nn_dist = 0

            # Extract specimen info from filename
            parts = fname.replace('_bonemat.cdb', '').replace('_re', '')
            side = 'left' if '_left' in fname else 'right' if '_right' in fname else 'unknown'
            specimen_id = parts.split('_')[0]

            stats.append({
                'filename': fname,
                'specimen_id': specimen_id,
                'side': side,
                'n_nodes': n_nodes,
                'n_elements': n_elements,
                'x_range_mm': x_range,
                'y_range_mm': y_range,
                'z_range_mm': z_range,
                'volume_bbox': x_range * y_range * z_range,
                'avg_nn_distance': avg_nn_dist,
                'centroid_x': pts[:, 0].mean() if n_nodes > 0 else 0,
                'centroid_y': pts[:, 1].mean() if n_nodes > 0 else 0,
                'centroid_z': pts[:, 2].mean() if n_nodes > 0 else 0,
            })
        return pd.DataFrame(stats)


# ============================================================
# SECTION 2.6: POINT CLOUD AUGMENTATION
# ============================================================
class PointCloudAugmentation:
    """Data augmentation techniques for 3D point clouds.

    Bone mesh-specific augmentations to increase training data diversity.
    """

    @staticmethod
    def random_rotation(points, max_angle=180):
        """Random rotation around all 3 axes."""
        angles = np.random.uniform(-max_angle, max_angle, 3) * np.pi / 180
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]),  np.cos(angles[0])]])
        Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]),  np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return (points @ R.T).astype(np.float32)

    @staticmethod
    def random_scale(points, scale_range=(0.85, 1.15)):
        """Random uniform scaling."""
        scale = np.random.uniform(*scale_range)
        return (points * scale).astype(np.float32)

    @staticmethod
    def random_jitter(points, sigma=0.005, clip=0.02):
        """Add Gaussian noise to point positions."""
        noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
        return (points + noise).astype(np.float32)

    @staticmethod
    def random_translate(points, max_shift=0.1):
        """Random translation."""
        shift = np.random.uniform(-max_shift, max_shift, 3)
        return (points + shift).astype(np.float32)

    @staticmethod
    def random_dropout(points, max_drop_ratio=0.1):
        """Randomly remove some points."""
        n = len(points)
        keep_ratio = 1.0 - np.random.uniform(0, max_drop_ratio)
        n_keep = max(int(n * keep_ratio), 3)
        idx = np.random.choice(n, n_keep, replace=False)
        return points[idx]

    @staticmethod
    def random_anisotropic_scale(points, scale_range=(0.9, 1.1)):
        """Scale differently along each axis (bone-specific)."""
        scales = np.random.uniform(scale_range[0], scale_range[1], 3)
        return (points * scales).astype(np.float32)

    @staticmethod
    def augment(points, intensity='medium'):
        """Apply a chain of random augmentations.

        intensity: 'light', 'medium', 'heavy'
        """
        aug = PointCloudAugmentation
        if intensity == 'light':
            points = aug.random_rotation(points, max_angle=15)
            points = aug.random_jitter(points, sigma=0.002)
        elif intensity == 'medium':
            points = aug.random_rotation(points, max_angle=45)
            points = aug.random_scale(points, (0.9, 1.1))
            points = aug.random_jitter(points, sigma=0.005)
            if np.random.random() > 0.5:
                points = aug.random_translate(points, 0.05)
        elif intensity == 'heavy':
            points = aug.random_rotation(points)
            points = aug.random_anisotropic_scale(points, (0.85, 1.15))
            points = aug.random_jitter(points, sigma=0.008)
            points = aug.random_translate(points, 0.1)
            if np.random.random() > 0.5:
                points = aug.random_dropout(points, 0.15)
        return points


# ============================================================
# SECTION 2.7: DATA EXPLORATION & STATISTICS
# ============================================================
def explore_dataset(all_data, stats_df=None):
    """Comprehensive data exploration with visualizations."""
    print("\n" + "="*60)
    print("  📊 DATASET EXPLORATION")
    print("="*60)

    total_nodes = sum(len(d['nodes']) for d in all_data.values())
    total_elements = sum(len(d['elements']) for d in all_data.values())
    node_counts = [len(d['nodes']) for d in all_data.values()]
    elem_counts = [len(d['elements']) for d in all_data.values()]

    print(f"\n  Total files: {len(all_data)}")
    print(f"  Total nodes: {total_nodes:,}")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Nodes/file: {np.mean(node_counts):,.0f} ± {np.std(node_counts):,.0f}")
    print(f"  Min nodes: {min(node_counts):,} | Max nodes: {max(node_counts):,}")
    if total_elements > 0:
        print(f"  Elements/file: {np.mean(elem_counts):,.0f} ± {np.std(elem_counts):,.0f}")

    # Count left/right
    left = sum(1 for f in all_data if '_left' in f)
    right = sum(1 for f in all_data if '_right' in f)
    print(f"  Left bones: {left} | Right bones: {right}")

    # Unique specimens
    specimens = set()
    for f in all_data:
        sid = f.split('_')[0]
        specimens.add(sid)
    print(f"  Unique specimens: {len(specimens)}")

    # Visualization: node count distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Overview', fontsize=14, fontweight='bold')

    # 1. Node count histogram
    ax = axes[0][0]
    ax.hist(node_counts, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(node_counts), color='red', linestyle='--',
               label=f'Mean={np.mean(node_counts):,.0f}')
    ax.set_title('Nodes per Mesh')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Count')
    ax.legend()

    # 2. Element count histogram
    ax = axes[0][1]
    if total_elements > 0:
        ax.hist(elem_counts, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(np.mean(elem_counts), color='red', linestyle='--',
                   label=f'Mean={np.mean(elem_counts):,.0f}')
        ax.set_title('Elements per Mesh')
        ax.set_xlabel('Number of Elements')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No element data', transform=ax.transAxes,
                ha='center', fontsize=12)
        ax.set_title('Elements per Mesh')

    # 3. Left vs Right distribution
    ax = axes[1][0]
    ax.bar(['Left', 'Right', 'Other'], [left, right, len(all_data)-left-right],
           color=['#3498db', '#e74c3c', '#95a5a6'])
    ax.set_title('Left vs Right Bone Distribution')
    ax.set_ylabel('Count')

    # 4. Sorted node counts (to see variation)
    ax = axes[1][1]
    sorted_counts = sorted(node_counts, reverse=True)
    ax.plot(sorted_counts, 'o-', markersize=2, color='steelblue')
    ax.set_title('Sorted Node Counts (descending)')
    ax.set_xlabel('Mesh Index')
    ax.set_ylabel('Node Count')
    ax.axhline(np.mean(node_counts), color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

    # If stats_df provided, show spatial distribution
    if stats_df is not None and len(stats_df) > 0:
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        fig2.suptitle('Mesh Spatial Dimensions (mm)', fontsize=13, fontweight='bold')
        for i, (col, title) in enumerate([
            ('x_range_mm', 'X Range'), ('y_range_mm', 'Y Range'), ('z_range_mm', 'Z Range')
        ]):
            if col in stats_df.columns:
                axes2[i].hist(stats_df[col], bins=25, alpha=0.7, color='teal', edgecolor='black')
                axes2[i].set_title(title)
                axes2[i].set_xlabel('mm')
        plt.tight_layout()
        plt.savefig('spatial_dimensions.png', dpi=150, bbox_inches='tight')
        plt.show()

    return {'total_files': len(all_data), 'total_nodes': total_nodes,
            'total_elements': total_elements, 'specimens': len(specimens)}


# ============================================================
# SECTION 3: CDB PARSER
# ============================================================
class CDBParser:
    """Parse ANSYS CDB files to extract nodes and element connectivity."""

    def __init__(self):
        self.all_data = {}  # {filename: {'nodes': df, 'elements': list}}

    def parse_file(self, file_path):
        """Parse a single CDB file. Try pyansys first, fallback to custom."""
        fname = os.path.basename(file_path)
        if USE_PYANSYS:
            try:
                return self._parse_pyansys(file_path, fname)
            except Exception as e:
                print(f"  pyansys failed for {fname}: {e}, using custom parser")
        return self._parse_custom(file_path, fname)

    def _parse_pyansys(self, file_path, fname):
        """Parse using ansys-mapdl-reader (official, more reliable)."""
        archive = Archive(file_path)
        # Extract nodes
        node_ids = archive.nnum
        nodes = archive.nodes
        nodes_df = pd.DataFrame({
            'node_id': node_ids,
            'x': nodes[:, 0], 'y': nodes[:, 1], 'z': nodes[:, 2]
        })
        # Extract elements (cell connectivity)
        elements = []
        if hasattr(archive, 'elem') and len(archive.elem) > 0:
            for elem in archive.elem:
                node_list = [n for n in elem[-1] if n > 0]  # Filter padded zeros
                if len(node_list) >= 4:
                    elements.append(tuple(node_list[:4]))  # Take first 4 for tet
        print(f"  ✅ {fname}: {len(nodes_df)} nodes, {len(elements)} elements (pyansys)")
        return nodes_df, elements

    def _parse_custom(self, file_path, fname):
        """Custom parser for CDB files (fallback)."""
        nodes, elements = [], []
        in_nblock, in_eblock = False, False
        format_info = None

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            ls = line.strip()

            # --- NBLOCK parsing ---
            if ls.startswith('NBLOCK'):
                in_nblock, in_eblock = True, False
                continue
            if in_nblock and ls.startswith('('):
                # Parse FORTRAN format spec
                fmt = {}
                im = re.search(r'(\d+)i(\d+)', ls)
                if im: fmt['iw'] = int(im.group(2))
                fm = re.search(r'(\d+)e(\d+)', ls)
                if fm: fmt['fw'] = int(fm.group(2))
                format_info = fmt if fmt else None
                continue
            if in_nblock and ls and not ls.startswith('!') and not ls.startswith('/'):
                if ls.startswith('EBLOCK') or ls.startswith('-1') or ls.startswith('N,'):
                    in_nblock = False
                    if ls.startswith('EBLOCK'):
                        in_eblock = True
                    continue
                result = self._parse_node_line(line, ls, format_info)
                if result:
                    nodes.append(result)
                continue

            # --- EBLOCK parsing ---
            if ls.startswith('EBLOCK'):
                in_eblock, in_nblock = True, False
                continue
            if in_eblock and ls.startswith('('):
                continue  # Skip format line
            if in_eblock:
                if ls.startswith('-1') or ls.startswith('NBLOCK') or ls.startswith('FINISH') or not ls:
                    in_eblock = False
                    continue
                elem = self._parse_element_line(ls)
                if elem:
                    elements.append(elem)

        nodes_df = pd.DataFrame(nodes, columns=['node_id', 'x', 'y', 'z']) if nodes else pd.DataFrame()
        print(f"  ✅ {fname}: {len(nodes_df)} nodes, {len(elements)} elements (custom)")
        return nodes_df, elements

    def _parse_node_line(self, raw_line, stripped, fmt):
        """Parse a single NBLOCK line."""
        # Try fixed-width first
        if fmt and 'iw' in fmt and 'fw' in fmt:
            try:
                iw, fw = fmt['iw'], fmt['fw']
                offset = iw * 3
                nid = int(raw_line[:iw].strip())
                x = float(raw_line[offset:offset+fw].strip())
                y = float(raw_line[offset+fw:offset+2*fw].strip())
                z = float(raw_line[offset+2*fw:offset+3*fw].strip())
                return [nid, x, y, z]
            except (ValueError, IndexError):
                pass
        # Fallback: space-separated
        try:
            parts = stripped.split()
            if len(parts) >= 6:
                return [int(parts[0]), float(parts[3]), float(parts[4]), float(parts[5])]
        except (ValueError, IndexError):
            pass
        return None

    def _parse_element_line(self, line):
        """Parse a single EBLOCK line to extract element node IDs."""
        try:
            parts = line.split()
            if len(parts) >= 11:
                # Standard EBLOCK: first 11 fields are metadata, rest are node IDs
                num_nodes = int(parts[8]) if parts[8].isdigit() else 4
                node_start = 11
                if len(parts) >= node_start + 4:
                    node_ids = [int(parts[node_start + j]) for j in range(min(num_nodes, len(parts) - node_start))
                                if parts[node_start + j].lstrip('-').isdigit() and int(parts[node_start + j]) > 0]
                    if len(node_ids) >= 4:
                        return tuple(node_ids[:4])  # Tetrahedral: 4 nodes
            elif len(parts) >= 8:
                # Simpler format
                node_ids = [int(p) for p in parts if p.lstrip('-').isdigit() and int(p) > 0]
                if len(node_ids) >= 4:
                    return tuple(node_ids[-4:])
        except (ValueError, IndexError):
            pass
        return None

    def parse_directory(self, directory):
        """Parse all CDB files in a directory."""
        cdb_files = sorted(glob.glob(os.path.join(directory, "*.cdb")))
        if not cdb_files:
            print(f"❌ No CDB files found in {directory}")
            return {}
        print(f"🔍 Found {len(cdb_files)} CDB files")
        for fp in cdb_files:
            fname = os.path.basename(fp)
            nodes_df, elements = self.parse_file(fp)
            if not nodes_df.empty:
                self.all_data[fname] = {'nodes': nodes_df, 'elements': elements}
        n_with_elem = sum(1 for d in self.all_data.values() if len(d['elements']) > 0)
        print(f"\n✅ Parsed {len(self.all_data)} files ({n_with_elem} with elements)")
        return self.all_data

# ============================================================
# SECTION 4: FEA MESH QUALITY METRICS
# ============================================================
class MeshQualityAnalyzer:
    """Compute standard FEA mesh quality metrics per element."""

    @staticmethod
    def compute_tet_metrics(nodes_df, elements):
        """Compute quality metrics for tetrahedral elements.

        Returns dict with per-element metrics:
        - volume, aspect_ratio, scaled_jacobian, skewness, edge_ratio
        """
        if not elements:
            return None

        # Build node lookup: node_id -> [x, y, z]
        nid_to_xyz = {}
        for _, row in nodes_df.iterrows():
            nid_to_xyz[int(row['node_id'])] = np.array([row['x'], row['y'], row['z']])

        metrics = {'volume': [], 'aspect_ratio': [], 'scaled_jacobian': [],
                   'skewness': [], 'edge_ratio': [], 'min_angle': []}
        valid_count = 0

        for elem in elements:
            try:
                pts = [nid_to_xyz[nid] for nid in elem[:4] if nid in nid_to_xyz]
                if len(pts) < 4:
                    continue
                p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
                m = MeshQualityAnalyzer._tet_metrics(p0, p1, p2, p3)
                if m:
                    for k, v in m.items():
                        metrics[k].append(v)
                    valid_count += 1
            except (KeyError, ValueError):
                continue

        if valid_count == 0:
            return None

        return {k: np.array(v) for k, v in metrics.items()}

    @staticmethod
    def _tet_metrics(p0, p1, p2, p3):
        """Compute quality metrics for a single tetrahedron."""
        # Edge vectors
        edges = [p1-p0, p2-p0, p3-p0, p2-p1, p3-p1, p3-p2]
        edge_lengths = np.array([np.linalg.norm(e) for e in edges])

        if np.any(edge_lengths < 1e-12):
            return None  # Degenerate element

        # Volume via scalar triple product
        v0, v1, v2 = p1-p0, p2-p0, p3-p0
        det_J = np.dot(v0, np.cross(v1, v2))
        volume = abs(det_J) / 6.0

        if volume < 1e-15:
            return None  # Zero volume

        # Edge ratio (max/min edge length)
        edge_ratio = edge_lengths.max() / edge_lengths.min()

        # Aspect ratio via circumradius/inradius
        # Face areas
        faces = [(p0,p1,p2), (p0,p1,p3), (p0,p2,p3), (p1,p2,p3)]
        face_areas = []
        for fa, fb, fc in faces:
            face_areas.append(0.5 * np.linalg.norm(np.cross(fb-fa, fc-fa)))
        total_area = sum(face_areas)

        if total_area < 1e-15:
            return None

        # Inradius: IR = 3V / total_surface_area
        inradius = 3.0 * volume / total_area

        # Circumradius (simplified): CR = product of edge lengths / (8 * V * area_factor)
        # Using the formula: CR = |a||b||c| / (6V) for appropriate edges
        # Simplified aspect ratio = max_edge / (2 * sqrt(6) * inradius)
        ar_ideal = 1.0  # Perfect tet
        aspect_ratio = edge_lengths.max() / (2.0 * np.sqrt(6) * inradius)

        # Scaled Jacobian
        # For linear tet: normalize det by product of edge lengths
        l0, l1, l2 = np.linalg.norm(v0), np.linalg.norm(v1), np.linalg.norm(v2)
        if l0 * l1 * l2 < 1e-15:
            return None
        # Ideal tet det = sqrt(2) * (edge_length)^3
        # Scaled Jacobian ∈ [-1, 1], 1 = perfect
        scaled_jac = det_J / (l0 * l1 * l2 * np.sqrt(2))
        scaled_jac = np.clip(scaled_jac, -1.0, 1.0)

        # Skewness: 1 - shape_factor
        # Shape factor = V / V_ideal (equilateral with same circumradius)
        # V_ideal for equilateral tet with edge a: V = a^3 / (6*sqrt(2))
        mean_edge = edge_lengths.mean()
        v_ideal = mean_edge**3 / (6.0 * np.sqrt(2))
        shape_factor = min(volume / v_ideal, 1.0) if v_ideal > 0 else 0
        skewness = 1.0 - shape_factor

        # Min dihedral angle (simplified via face normals)
        normals = [np.cross(fb-fa, fc-fa) for fa, fb, fc in faces]
        normals = [n/np.linalg.norm(n) if np.linalg.norm(n) > 1e-12 else n for n in normals]
        angles = []
        for i in range(len(normals)):
            for j in range(i+1, len(normals)):
                cos_a = np.clip(np.dot(normals[i], normals[j]), -1, 1)
                angles.append(np.degrees(np.arccos(abs(cos_a))))
        min_angle = min(angles) if angles else 0

        return {
            'volume': volume,
            'aspect_ratio': aspect_ratio,
            'scaled_jacobian': scaled_jac,
            'skewness': skewness,
            'edge_ratio': edge_ratio,
            'min_angle': min_angle,
        }

    @staticmethod
    def compute_with_pyvista(file_path):
        """Compute quality using PyVista (industry-standard Verdict library)."""
        if not USE_PYVISTA or not USE_PYANSYS:
            return None
        try:
            archive = Archive(file_path)
            grid = archive.grid
            if grid.n_cells == 0:
                return None
            metrics = {}
            for measure in ['scaled_jacobian', 'aspect_ratio', 'skew', 'volume']:
                try:
                    qual = grid.compute_cell_quality(quality_measure=measure)
                    metrics[measure] = qual.cell_data['CellQuality']
                except Exception:
                    pass
            return metrics if metrics else None
        except Exception:
            return None

# ============================================================
# SECTION 5: QUALITY LABELING
# ============================================================
class QualityLabeler:
    """Label mesh quality based on computed metrics."""

    GOOD, ACCEPTABLE, POOR = 0, 1, 2
    LABELS = ['Good', 'Acceptable', 'Poor']

    @staticmethod
    def label_element(ar, sj, skew):
        """Label a single element: 0=Good, 1=Acceptable, 2=Poor."""
        votes = [0, 0, 0]  # good, acceptable, poor

        # Aspect ratio
        if ar < CONFIG['ar_good']:    votes[0] += 1
        elif ar > CONFIG['ar_poor']:  votes[2] += 1
        else:                         votes[1] += 1

        # Scaled Jacobian
        if sj > CONFIG['sj_good']:    votes[0] += 1
        elif sj < CONFIG['sj_poor']:  votes[2] += 1
        else:                         votes[1] += 1

        # Skewness
        if skew < CONFIG['skew_good']:  votes[0] += 1
        elif skew > CONFIG['skew_poor']: votes[2] += 1
        else:                           votes[1] += 1

        return int(np.argmax(votes))

    @staticmethod
    def label_mesh(metrics):
        """Label an entire mesh based on element quality distribution."""
        if metrics is None:
            return 1  # Default to Acceptable if no metrics

        n = len(metrics['aspect_ratio'])
        labels = np.array([
            QualityLabeler.label_element(
                metrics['aspect_ratio'][i],
                metrics['scaled_jacobian'][i],
                metrics['skewness'][i]
            ) for i in range(n)
        ])
        pct_good = np.mean(labels == 0)
        pct_poor = np.mean(labels == 2)

        if pct_good >= 0.7:
            return 0  # Good mesh
        elif pct_poor >= 0.3:
            return 2  # Poor mesh
        else:
            return 1  # Acceptable

    @staticmethod
    def get_mesh_features(metrics):
        """Extract statistical features from element metrics for ML."""
        if metrics is None:
            return np.zeros(18)
        feats = []
        for key in ['aspect_ratio', 'scaled_jacobian', 'skewness',
                     'edge_ratio', 'volume', 'min_angle']:
            arr = metrics.get(key, np.array([0]))
            feats.extend([np.mean(arr), np.std(arr), np.median(arr)])
        return np.array(feats)

# ============================================================
# SECTION 6: GRAPH CONSTRUCTION & DATASET
# ============================================================
def build_graph_from_mesh(nodes_df, elements, max_nodes=2048):
    """Build graph representation from mesh data.
    
    Nodes = mesh vertices, Edges = element connectivity.
    Returns: node_features (N×F), edge_index (2×E), positions (N×3)
    """
    # Subsample nodes if too large
    n = len(nodes_df)
    if n > max_nodes:
        idx = np.random.choice(n, max_nodes, replace=False)
        idx.sort()
        nodes_sub = nodes_df.iloc[idx].reset_index(drop=True)
    else:
        nodes_sub = nodes_df.copy()
    
    # Normalize positions to unit sphere
    pos = nodes_sub[['x', 'y', 'z']].values.astype(np.float32)
    centroid = pos.mean(axis=0)
    pos_centered = pos - centroid
    max_dist = np.max(np.linalg.norm(pos_centered, axis=1)) + 1e-8
    pos_norm = pos_centered / max_dist
    
    # Build adjacency from elements
    node_ids = set(nodes_sub['node_id'].values)
    nid_to_idx = {int(nid): i for i, nid in enumerate(nodes_sub['node_id'].values)}
    
    edges = set()
    for elem in elements:
        mapped = [nid_to_idx[nid] for nid in elem[:4] if nid in nid_to_idx]
        for a, b in combinations(mapped, 2):
            edges.add((a, b))
            edges.add((b, a))
    
    if len(edges) == 0:
        # Fallback: k-NN graph (k=8)
        from scipy.spatial import cKDTree
        tree = cKDTree(pos_norm)
        k = min(8, len(pos_norm) - 1)
        _, indices = tree.query(pos_norm, k=k+1)
        for i in range(len(pos_norm)):
            for j in indices[i, 1:]:
                edges.add((i, j))
                edges.add((j, i))
    
    edge_index = np.array(list(edges), dtype=np.int64).T  # 2 × E
    
    # Node features: position + local stats
    degree = np.zeros(len(nodes_sub), dtype=np.float32)
    for e in edges:
        degree[e[0]] += 1
    degree = degree / (degree.max() + 1e-8)
    
    node_features = np.concatenate([
        pos_norm,                          # x, y, z (3)
        degree.reshape(-1, 1),             # degree (1)
    ], axis=1).astype(np.float32)          # Total: 4 features
    
    return node_features, edge_index, pos_norm


class BoneMeshDataset(Dataset):
    """PyTorch Dataset for bone mesh quality classification."""
    
    def __init__(self, data_list, mode='pointnet'):
        """
        data_list: list of dicts with keys:
            'nodes_df', 'elements', 'label', 'metrics', 'filename'
        mode: 'pointnet' or 'gnn'
        """
        self.data = data_list
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        label = item['label']
        
        if self.mode == 'pointnet':
            return self._get_pointnet(item, label)
        else:
            return self._get_gnn(item, label)
    
    def _get_pointnet(self, item, label):
        """Return sampled point cloud for PointNet."""
        nodes = item['nodes_df']
        pts = nodes[['x', 'y', 'z']].values.astype(np.float32)
        
        # Normalize
        pts = pts - pts.mean(axis=0)
        max_d = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
        pts = pts / max_d
        
        # Sample to fixed size
        n = len(pts)
        target_n = CONFIG['num_points']
        if n >= target_n:
            idx = np.random.choice(n, target_n, replace=False)
        else:
            idx = np.random.choice(n, target_n, replace=True)
        pts = pts[idx]
        
        # Random augmentation
        if np.random.random() > 0.5:
            angle = np.random.uniform(0, 2*np.pi)
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0],
                          [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T
        pts += np.random.normal(0, 0.005, pts.shape).astype(np.float32)
        
        return torch.tensor(pts), torch.tensor(label, dtype=torch.long)
    
    def _get_gnn(self, item, label):
        """Return graph data for GNN."""
        node_feat, edge_idx, _ = build_graph_from_mesh(
            item['nodes_df'], item['elements'],
            max_nodes=CONFIG['num_graph_nodes']
        )
        return (torch.tensor(node_feat),
                torch.tensor(edge_idx, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))


def gnn_collate_fn(batch):
    """Custom collate for variable-size graphs."""
    node_feats, edge_idxs, labels = [], [], []
    node_offset = 0
    batch_idx = []
    
    for i, (nf, ei, lab) in enumerate(batch):
        node_feats.append(nf)
        edge_idxs.append(ei + node_offset)
        labels.append(lab)
        batch_idx.extend([i] * len(nf))
        node_offset += len(nf)
    
    return (torch.cat(node_feats, 0),
            torch.cat(edge_idxs, 1),
            torch.tensor(batch_idx, dtype=torch.long),
            torch.stack(labels))


# ============================================================
# SECTION 7: MODEL ARCHITECTURES
# ============================================================

# --- 7A: Graph Attention Network (GAT) ---
class GATLayer(nn.Module):
    """Single Graph Attention layer (Velickovic et al., 2018)."""
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.randn(num_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, x, edge_index):
        """x: (N, in_dim), edge_index: (2, E)"""
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.out_dim)  # (N, H, D)
        
        src, dst = edge_index[0], edge_index[1]
        
        # Attention scores
        e_src = (h[src] * self.a_src).sum(-1)  # (E, H)
        e_dst = (h[dst] * self.a_dst).sum(-1)  # (E, H)
        e = self.leaky_relu(e_src + e_dst)     # (E, H)
        
        # Softmax per destination node
        e_max = torch.zeros(N, self.num_heads, device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=True)
        e = torch.exp(e - e_max[dst])
        
        e_sum = torch.zeros(N, self.num_heads, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        alpha = e / (e_sum[dst] + 1e-8)  # (E, H)
        alpha = self.dropout(alpha)
        
        # Aggregate
        msg = h[src] * alpha.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
        
        if self.concat:
            return out.view(N, -1)  # (N, H*D)
        else:
            return out.mean(dim=1)  # (N, D)


class BoneMeshGAT(nn.Module):
    """GAT model for mesh quality classification."""
    
    def __init__(self, in_dim=4, hidden=64, num_classes=3, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden, heads, dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.gat2 = GATLayer(hidden * heads, hidden, heads, dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.gat3 = GATLayer(hidden * heads, hidden, 1, dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128),  # *2 for mean+max pool
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, edge_index, batch):
        """x: (total_N, F), edge_index: (2, total_E), batch: (total_N,)"""
        h = F.elu(self.bn1(self.gat1(x, edge_index)))
        h = F.elu(self.bn2(self.gat2(h, edge_index)))
        h = F.elu(self.bn3(self.gat3(h, edge_index)))
        
        # Global pooling: mean + max
        num_graphs = batch.max().item() + 1
        h_mean = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_max = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_mean.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce='mean', include_self=True)
        h_max.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce='amax', include_self=True)
        
        graph_feat = torch.cat([h_mean, h_max], dim=1)
        return self.classifier(graph_feat)


# --- 7B: PointNet (Baseline) ---
class PointNetEncoder(nn.Module):
    """PointNet encoder: shared MLPs + max pooling."""
    
    def __init__(self, hidden=256):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Linear(128, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
    
    def forward(self, x):
        """x: (B, N, 3)"""
        B, N, _ = x.shape
        x = x.view(B*N, 3)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = x.view(B, N, -1)
        x = x.max(dim=1)[0]  # Global max pool: (B, hidden)
        return x


class PointNetClassifier(nn.Module):
    """PointNet classifier for mesh quality."""
    
    def __init__(self, num_classes=3, hidden=256, dropout=0.3):
        super().__init__()
        self.encoder = PointNetEncoder(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# --- 7C: Enhanced PointNet with T-Net (Spatial Transformer Network) ---
class TNet(nn.Module):
    """T-Net: learns an affine transformation matrix for input alignment."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(k, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, k * k),
        )
        # Initialize as identity
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.fc[-1].bias.data = torch.eye(k).flatten()

    def forward(self, x):
        """x: (B, N, k)"""
        B, N, _ = x.shape
        h = x.view(B * N, self.k)
        h = self.mlp(h).view(B, N, -1)
        h = h.max(dim=1)[0]  # (B, 256)
        transform = self.fc(h).view(B, self.k, self.k)
        return transform


class EnhancedPointNet(nn.Module):
    """PointNet with T-Net alignment for mesh quality classification."""

    def __init__(self, num_classes=3, hidden=256, dropout=0.3):
        super().__init__()
        self.tnet3 = TNet(k=3)  # Input transform
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.tnet64 = TNet(k=64)  # Feature transform
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.feat_transform_reg = 0  # Regularization loss for feature transform

    def forward(self, x):
        """x: (B, N, 3)"""
        B, N, _ = x.shape
        # Input alignment
        T3 = self.tnet3(x)  # (B, 3, 3)
        x = torch.bmm(x, T3)

        x = x.view(B * N, 3)
        x = self.mlp1(x).view(B, N, 64)

        # Feature alignment
        T64 = self.tnet64(x)  # (B, 64, 64)
        x = torch.bmm(x, T64)
        # Regularization: T64 should be close to orthogonal
        I = torch.eye(64, device=x.device).unsqueeze(0).expand(B, -1, -1)
        self.feat_transform_reg = torch.mean(
            torch.norm(torch.bmm(T64, T64.transpose(1, 2)) - I, dim=(1, 2)))

        x = x.view(B * N, 64)
        x = self.mlp2(x).view(B, N, -1)
        x = x.max(dim=1)[0]  # (B, hidden)
        return self.head(x)


# --- 7D: Graph Transformer ---
class GraphTransformerLayer(nn.Module):
    """Transformer-style self-attention on graph nodes.

    Unlike GAT (which uses additive attention on edges), this uses
    scaled dot-product attention with optional edge masking.
    Better at capturing long-range dependencies in the mesh.
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, use_edge_mask=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.use_edge_mask = use_edge_mask

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim * 2, out_dim),
        )
        self.dropout = nn.Dropout(dropout)
        # Project input to out_dim if needed
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(self, x, edge_index, batch):
        """x: (N, in_dim), edge_index: (2, E), batch: (N,)"""
        N = x.size(0)
        residual = self.residual_proj(x)

        Q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim)

        # For efficiency on large graphs: use sparse attention via edge_index
        # Compute attention only between connected nodes
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)

        q_dst = Q[dst]  # (E, H, D)
        k_src = K[src]  # (E, H, D)
        v_src = V[src]  # (E, H, D)

        # Scaled dot-product attention
        attn_scores = (q_dst * k_src).sum(-1) * self.scale  # (E, H)

        # Softmax per destination node
        attn_max = torch.zeros(N, self.num_heads, device=x.device)
        attn_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(attn_scores),
                                  attn_scores, reduce='amax', include_self=True)
        attn_weights = torch.exp(attn_scores - attn_max[dst])
        attn_sum = torch.zeros(N, self.num_heads, device=x.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(attn_weights), attn_weights)
        attn_weights = attn_weights / (attn_sum[dst] + 1e-8)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        msg = v_src * attn_weights.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
        out = out.view(N, self.out_dim)
        out = self.W_o(out)

        # Residual + LayerNorm
        out = self.layer_norm1(residual + self.dropout(out))
        # FFN
        out = self.layer_norm2(out + self.dropout(self.ffn(out)))
        return out


class BoneMeshGraphTransformer(nn.Module):
    """Graph Transformer for mesh quality classification.

    Uses transformer-style attention instead of GAT's additive attention.
    Better at capturing global mesh patterns — important for bone quality.
    """

    def __init__(self, in_dim=4, hidden=64, num_classes=3, num_heads=4,
                 num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden, hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Classification head with mean + max + std pooling
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 128),  # *3 for mean+max+std
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, batch)

        # Graph-level pooling: mean + max + std
        num_graphs = batch.max().item() + 1
        h_mean = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_max = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_mean.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h,
                                reduce='mean', include_self=True)
        h_max.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h,
                               reduce='amax', include_self=True)
        # Std pooling
        h_sq_mean = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_sq_mean.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h**2,
                                   reduce='mean', include_self=True)
        h_std = torch.sqrt(torch.clamp(h_sq_mean - h_mean**2, min=1e-8))

        graph_feat = torch.cat([h_mean, h_max, h_std], dim=1)
        return self.classifier(graph_feat)


# --- 7E: Quality Regression Head (predict continuous quality scores) ---
class QualityRegressionHead(nn.Module):
    """Predicts continuous mesh quality metrics from learned features.

    Instead of just classifying Good/Acceptable/Poor, this predicts
    the actual mean Aspect Ratio, Scaled Jacobian, and Skewness values.
    This is more informative for FEA practitioners.
    """

    def __init__(self, feature_dim, num_targets=3):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_targets),  # AR, SJ, Skewness
        )

    def forward(self, features):
        return self.regressor(features)


class DualTaskGAT(nn.Module):
    """GAT with both classification and regression heads.

    Simultaneously predicts:
    1. Quality class: Good/Acceptable/Poor (classification)
    2. Quality scores: mean AR, SJ, Skewness (regression)

    Multi-task learning helps the model learn better representations.
    """

    def __init__(self, in_dim=4, hidden=64, num_classes=3, heads=4, dropout=0.2):
        super().__init__()
        # Shared GNN backbone
        self.gat1 = GATLayer(in_dim, hidden, heads, dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.gat2 = GATLayer(hidden * heads, hidden, heads, dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.gat3 = GATLayer(hidden * heads, hidden, 1, dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden)

        pool_dim = hidden * 2  # mean + max
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(pool_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        # Regression head
        self.reg_head = QualityRegressionHead(pool_dim, num_targets=3)

    def forward(self, x, edge_index, batch):
        h = F.elu(self.bn1(self.gat1(x, edge_index)))
        h = F.elu(self.bn2(self.gat2(h, edge_index)))
        h = F.elu(self.bn3(self.gat3(h, edge_index)))

        num_graphs = batch.max().item() + 1
        h_mean = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_max = torch.zeros(num_graphs, h.size(1), device=h.device)
        h_mean.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h,
                                reduce='mean', include_self=True)
        h_max.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h,
                               reduce='amax', include_self=True)
        graph_feat = torch.cat([h_mean, h_max], dim=1)

        cls_logits = self.cls_head(graph_feat)
        reg_preds = self.reg_head(graph_feat)
        return cls_logits, reg_preds


# ============================================================
# SECTION 7.5: MODEL FACTORY & TRAINING UTILITIES
# ============================================================
MODEL_REGISTRY = {
    'pointnet': lambda nc: PointNetClassifier(nc, CONFIG['pointnet_hidden']),
    'pointnet_tnet': lambda nc: EnhancedPointNet(nc, CONFIG['pointnet_hidden']),
    'gat': lambda nc: BoneMeshGAT(in_dim=4, hidden=CONFIG['gat_hidden'],
                                   num_classes=nc, heads=CONFIG['gat_heads']),
    'graph_transformer': lambda nc: BoneMeshGraphTransformer(
        in_dim=4, hidden=CONFIG['gat_hidden'], num_classes=nc,
        num_heads=CONFIG['gat_heads'], num_layers=3),
    'dual_gat': lambda nc: DualTaskGAT(in_dim=4, hidden=CONFIG['gat_hidden'],
                                         num_classes=nc, heads=CONFIG['gat_heads']),
}


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def print_model_summary(model_name, model):
    """Print model architecture summary."""
    n_params = count_parameters(model)
    print(f"\n  📐 {model_name}")
    print(f"     Parameters: {n_params:,}")
    print(f"     Layers: {sum(1 for _ in model.modules()) - 1}")


class TrainingHistory:
    """Track and plot training metrics across epochs."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []

    def record(self, train_loss, val_loss, train_acc, val_acc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.lrs.append(lr)

    def plot(self, title="Training History", save_path=None):
        """Plot loss and accuracy curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        epochs = range(1, len(self.train_losses) + 1)

        # Loss
        ax = axes[0]
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=1.5)
        ax.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1]
        ax.plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=1.5)
        ax.plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Learning rate
        ax = axes[2]
        ax.plot(epochs, self.lrs, 'g-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================
# SECTION 8: TRAINING PIPELINE
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, model_type='pointnet'):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        optimizer.zero_grad()
        if model_type == 'pointnet':
            pts, labels = batch
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            logits = model(pts)
        else:  # gnn
            node_feat, edge_idx, batch_idx, labels = batch
            node_feat = node_feat.to(DEVICE)
            edge_idx = edge_idx.to(DEVICE)
            batch_idx = batch_idx.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(node_feat, edge_idx, batch_idx)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, model_type='pointnet'):
    """Evaluate model. Returns loss, accuracy, predictions, true labels."""
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []

    for batch in loader:
        if model_type == 'pointnet':
            pts, labels = batch
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            logits = model(pts)
        else:
            node_feat, edge_idx, batch_idx, labels = batch
            node_feat = node_feat.to(DEVICE)
            edge_idx = edge_idx.to(DEVICE)
            batch_idx = batch_idx.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(node_feat, edge_idx, batch_idx)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        probs = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    n = max(len(all_labels), 1)
    return (total_loss / n, np.array(all_preds),
            np.array(all_labels), np.array(all_probs))


def run_kfold_experiment(data_list, model_type='pointnet'):
    """Run K-Fold Cross Validation experiment.

    Supports model_type: 'pointnet', 'pointnet_tnet', 'gat', 'graph_transformer', 'dual_gat'
    Returns dict with per-fold and aggregate results.
    """
    labels = np.array([d['label'] for d in data_list])
    num_classes = len(np.unique(labels))
    is_pointnet = 'pointnet' in model_type

    print(f"\n{'='*60}")
    print(f"  K-Fold CV: {model_type.upper()} | {CONFIG['k_folds']} folds")
    print(f"  Classes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    fold_results = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), labels)):
        print(f"\n--- Fold {fold+1}/{CONFIG['k_folds']} ---")
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        # Compute class weights for imbalanced data
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        # Create datasets and loaders based on model type
        if is_pointnet:
            train_ds = BoneMeshDataset(train_data, mode='pointnet')
            val_ds = BoneMeshDataset(val_data, mode='pointnet')
            train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, CONFIG['batch_size'])
        else:
            train_ds = BoneMeshDataset(train_data, mode='gnn')
            val_ds = BoneMeshDataset(val_data, mode='gnn')
            train_loader = DataLoader(train_ds, CONFIG['batch_size'],
                                      shuffle=True, collate_fn=gnn_collate_fn)
            val_loader = DataLoader(val_ds, CONFIG['batch_size'],
                                    collate_fn=gnn_collate_fn)

        # Create model using registry
        if model_type in MODEL_REGISTRY:
            model = MODEL_REGISTRY[model_type](num_classes).to(DEVICE)
        elif is_pointnet:
            model = PointNetClassifier(num_classes, CONFIG['pointnet_hidden']).to(DEVICE)
        else:
            model = BoneMeshGAT(in_dim=4, hidden=CONFIG['gat_hidden'],
                                num_classes=num_classes,
                                heads=CONFIG['gat_heads']).to(DEVICE)

        if fold == 0:
            n_params = count_parameters(model)
            print(f"  Model parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = TrainingHistory()

        # Determine effective model_type for train/eval functions
        effective_type = 'pointnet' if is_pointnet else 'gnn'

        for epoch in range(CONFIG['epochs']):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, effective_type)
            val_loss, val_preds, val_true, val_probs = evaluate(
                model, val_loader, criterion, effective_type)
            val_acc = accuracy_score(val_true, val_preds)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            history.record(train_loss, val_loss, train_acc, val_acc, current_lr)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} "
                      f"Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and final eval
        if best_state:
            model.load_state_dict(best_state)
            model.to(DEVICE)
        _, val_preds, val_true, val_probs = evaluate(
            model, val_loader, criterion, effective_type)

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
        print(f"  Fold {fold+1} Best — Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': val_acc,
            'f1_weighted': val_f1,
            'predictions': val_preds,
            'true_labels': val_true,
            'probabilities': val_probs,
            'best_val_loss': best_val_loss,
        })
        all_histories.append(history)

    # Plot training history for last fold
    if all_histories:
        all_histories[-1].plot(
            title=f'{model_type.upper()} Training History (last fold)',
            save_path=f'training_history_{model_type}.png')

    # Aggregate results
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1_weighted'] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} RESULTS ({CONFIG['k_folds']}-Fold CV)")
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"{'='*60}")

    all_preds = np.concatenate([r['predictions'] for r in fold_results])
    all_true = np.concatenate([r['true_labels'] for r in fold_results])
    print("\nOverall Classification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=QualityLabeler.LABELS[:num_classes],
                                zero_division=0))

    return {'model_type': model_type, 'folds': fold_results,
            'mean_acc': np.mean(accs), 'std_acc': np.std(accs),
            'mean_f1': np.mean(f1s), 'std_f1': np.std(f1s),
            'all_preds': all_preds, 'all_true': all_true,
            'histories': all_histories}


# ============================================================
# SECTION 9: EVALUATION & VISUALIZATION
# ============================================================
def plot_quality_distribution(all_metrics, labels):
    """Plot distribution of mesh quality metrics and labels."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('FEA Mesh Quality Metrics Distribution', fontsize=14, fontweight='bold')

    metric_names = ['aspect_ratio', 'scaled_jacobian', 'skewness',
                    'edge_ratio', 'volume', 'min_angle']
    titles = ['Aspect Ratio', 'Scaled Jacobian', 'Skewness',
              'Edge Length Ratio', 'Element Volume', 'Min Dihedral Angle (°)']

    for idx, (key, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx // 3][idx % 3]
        all_vals = []
        for m in all_metrics:
            if m and key in m:
                all_vals.extend(m[key].tolist())
        if all_vals:
            ax.hist(all_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(np.mean(all_vals), color='red', linestyle='--', label=f'Mean={np.mean(all_vals):.3f}')
            ax.set_title(title)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig('mesh_quality_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Label distribution
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    unique, counts = np.unique(labels, return_counts=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax2.bar([QualityLabeler.LABELS[u] for u in unique],
                   counts, color=[colors[u] for u in unique])
    ax2.set_title('Mesh Quality Label Distribution', fontweight='bold')
    ax2.set_ylabel('Count')
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison(results_pn, results_gnn):
    """Compare PointNet vs GNN results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PointNet vs GNN: Mesh Quality Classification', fontsize=14, fontweight='bold')

    # 1. Accuracy comparison per fold
    ax = axes[0]
    folds = range(1, CONFIG['k_folds'] + 1)
    pn_accs = [r['accuracy'] for r in results_pn['folds']]
    gnn_accs = [r['accuracy'] for r in results_gnn['folds']]
    x = np.arange(len(folds))
    w = 0.35
    ax.bar(x - w/2, pn_accs, w, label='PointNet', color='#3498db', alpha=0.8)
    ax.bar(x + w/2, gnn_accs, w, label='GNN (GAT)', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Fold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # 2. Confusion matrices
    for idx, (results, name, color) in enumerate([
        (results_pn, 'PointNet', 'Blues'),
        (results_gnn, 'GNN (GAT)', 'Reds')
    ]):
        ax = axes[idx + 1]
        cm = confusion_matrix(results['all_true'], results['all_preds'])
        num_c = cm.shape[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                    xticklabels=QualityLabeler.LABELS[:num_c],
                    yticklabels=QualityLabeler.LABELS[:num_c])
        ax.set_title(f'{name}\nAcc={results["mean_acc"]:.3f}±{results["std_acc"]:.3f}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Summary table
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'PointNet':<20} {'GNN (GAT)':<20}")
    print("-"*60)
    print(f"{'Accuracy':<20} {results_pn['mean_acc']:.4f}±{results_pn['std_acc']:.4f}"
          f"{'':>5}{results_gnn['mean_acc']:.4f}±{results_gnn['std_acc']:.4f}")
    print(f"{'F1 (weighted)':<20} {results_pn['mean_f1']:.4f}±{results_pn['std_f1']:.4f}"
          f"{'':>5}{results_gnn['mean_f1']:.4f}±{results_gnn['std_f1']:.4f}")
    print("="*60)


def plot_mesh_quality_3d(nodes_df, elements, metrics, title="Mesh Quality"):
    """3D visualization of mesh colored by element quality."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not available for 3D visualization")
        return

    x, y, z = nodes_df['x'].values, nodes_df['y'].values, nodes_df['z'].values

    fig = go.Figure()
    # Point cloud colored by local quality
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=1.5, color=z, colorscale='Viridis', opacity=0.6),
        name='Nodes'
    ))

    # Alphahull mesh surface
    n_max = 5000
    if len(x) > n_max:
        idx = np.random.choice(len(x), n_max, replace=False)
        x_s, y_s, z_s = x[idx], y[idx], z[idx]
    else:
        x_s, y_s, z_s = x, y, z

    fig.add_trace(go.Mesh3d(
        x=x_s, y=y_s, z=z_s, alphahull=7,
        color='cyan', opacity=0.3, name='Surface'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                   aspectmode='data'),
        width=900, height=700
    )
    fig.show()
    return fig


# ============================================================
# SECTION 9.5: ADVANCED RESEARCH EVALUATION
# ============================================================
def plot_roc_curves(results_dict):
    """Plot ROC curves for each model (multi-class: one-vs-rest)."""
    from sklearn.preprocessing import label_binarize

    fig, axes = plt.subplots(1, len(results_dict), figsize=(7*len(results_dict), 6))
    if len(results_dict) == 1:
        axes = [axes]

    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for ax, (model_name, results) in zip(axes, results_dict.items()):
        all_true = results['all_true']
        all_probs = np.concatenate([r['probabilities'] for r in results['folds']])
        num_classes = all_probs.shape[1]
        y_bin = label_binarize(all_true, classes=list(range(num_classes)))

        for i in range(num_classes):
            if y_bin.shape[1] > i:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'{QualityLabeler.LABELS[i]} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves: {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metric_correlation(all_metrics):
    """Plot correlation heatmap of FEA quality metrics."""
    # Aggregate all element metrics
    metric_data = defaultdict(list)
    for m in all_metrics:
        if m is None:
            continue
        for key in m:
            metric_data[key].extend(m[key].tolist())

    if not metric_data:
        print("⚠️ No metrics available for correlation analysis")
        return

    df = pd.DataFrame(metric_data)
    # Take a sample if too large
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('FEA Mesh Quality Metrics Analysis', fontsize=14, fontweight='bold')

    # 1. Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Metric Correlation Matrix')

    # 2. Aspect Ratio vs Scaled Jacobian scatter
    if 'aspect_ratio' in df.columns and 'scaled_jacobian' in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=42)
        scatter = axes[1].scatter(sample['aspect_ratio'], sample['scaled_jacobian'],
                                   c=sample['skewness'] if 'skewness' in df.columns else 'blue',
                                   cmap='RdYlGn_r', alpha=0.3, s=5)
        axes[1].set_xlabel('Aspect Ratio')
        axes[1].set_ylabel('Scaled Jacobian')
        axes[1].set_title('AR vs SJ (colored by Skewness)')
        if 'skewness' in df.columns:
            plt.colorbar(scatter, ax=axes[1], label='Skewness')
        # Add quality region boundaries
        axes[1].axvline(CONFIG['ar_good'], color='g', linestyle='--', alpha=0.5, label='AR good')
        axes[1].axvline(CONFIG['ar_poor'], color='r', linestyle='--', alpha=0.5, label='AR poor')
        axes[1].axhline(CONFIG['sj_good'], color='g', linestyle=':', alpha=0.5)
        axes[1].axhline(CONFIG['sj_poor'], color='r', linestyle=':', alpha=0.5)
        axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('metric_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_tsne_latent_space(data_list, model, model_type='gnn'):
    """Visualize learned representations using t-SNE.

    Shows whether the model has learned to separate quality classes
    in its internal feature space.
    """
    from sklearn.manifold import TSNE

    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for item in data_list:
            label = item['label']
            if model_type in ['pointnet', 'pointnet_tnet']:
                pts = item['nodes_df'][['x', 'y', 'z']].values.astype(np.float32)
                pts = pts - pts.mean(axis=0)
                max_d = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
                pts = pts / max_d
                n = len(pts)
                target_n = CONFIG['num_points']
                idx = np.random.choice(n, target_n, replace=(n < target_n))
                pts = pts[idx]
                x = torch.tensor(pts).unsqueeze(0).to(DEVICE)
                feat = model.encoder(x).cpu().numpy().flatten()
            else:
                node_feat, edge_idx, _ = build_graph_from_mesh(
                    item['nodes_df'], item['elements'], CONFIG['num_graph_nodes'])
                x = torch.tensor(node_feat).to(DEVICE)
                ei = torch.tensor(edge_idx, dtype=torch.long).to(DEVICE)
                batch = torch.zeros(len(x), dtype=torch.long, device=DEVICE)
                # Get features before classifier
                if hasattr(model, 'gat1'):
                    h = F.elu(model.bn1(model.gat1(x, ei)))
                    h = F.elu(model.bn2(model.gat2(h, ei)))
                    h = F.elu(model.bn3(model.gat3(h, ei)))
                elif hasattr(model, 'input_proj'):
                    h = model.input_proj(x)
                    for layer in model.layers:
                        h = layer(h, ei, batch)
                else:
                    continue
                feat = h.mean(dim=0).cpu().numpy()

            features.append(feat)
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    if len(features) < 5:
        print("⚠️ Not enough data for t-SNE")
        return

    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for i, name in enumerate(QualityLabeler.LABELS):
        mask = labels == i
        if mask.any():
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=colors[i], label=f'{name} (n={mask.sum()})',
                      alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
    ax.set_title(f't-SNE: Learned Feature Space ({model_type.upper()})', fontweight='bold')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'tsne_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()


def statistical_comparison(results_dict):
    """Statistical significance test between models (McNemar's test).

    Tests whether two models make significantly different errors.
    Important for thesis: proves GNN is *statistically* better/worse than PointNet.
    """
    model_names = list(results_dict.keys())
    if len(model_names) < 2:
        return

    print("\n" + "="*60)
    print("  📊 STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            res_a, res_b = results_dict[name_a], results_dict[name_b]

            pred_a = res_a['all_preds']
            pred_b = res_b['all_preds']
            true = res_a['all_true']

            # Build contingency table
            correct_a = (pred_a == true)
            correct_b = (pred_b == true)

            # n01: A wrong, B correct; n10: A correct, B wrong
            n01 = np.sum(~correct_a & correct_b)
            n10 = np.sum(correct_a & ~correct_b)

            # McNemar's test (chi-squared)
            if n01 + n10 > 0:
                chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
                from scipy import stats as scipy_stats
                p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)
            else:
                chi2, p_value = 0, 1.0

            sig = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
            print(f"\n  {name_a} vs {name_b}:")
            print(f"    {name_a} correct, {name_b} wrong: {n10}")
            print(f"    {name_a} wrong, {name_b} correct: {n01}")
            print(f"    McNemar's χ² = {chi2:.4f}, p = {p_value:.4f}")
            print(f"    {sig} (α=0.05)")

            diff_acc = res_b['mean_acc'] - res_a['mean_acc']
            better = name_b if diff_acc > 0 else name_a
            print(f"    Better model: {better} (Δacc = {abs(diff_acc):.4f})")


def quality_regression_scatter(data_list, model, model_type='gat'):
    """Scatter plot: predicted vs actual quality scores (regression task)."""
    if not hasattr(model, 'reg_head'):
        print("  ℹ️ Model doesn't have regression head, skipping")
        return

    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for item in data_list:
            metrics = item['metrics']
            if metrics is None:
                continue
            # Actual quality scores
            actual = np.array([
                np.mean(metrics['aspect_ratio']),
                np.mean(metrics['scaled_jacobian']),
                np.mean(metrics['skewness']),
            ])
            # Predict
            node_feat, edge_idx, _ = build_graph_from_mesh(
                item['nodes_df'], item['elements'], CONFIG['num_graph_nodes'])
            x = torch.tensor(node_feat).to(DEVICE)
            ei = torch.tensor(edge_idx, dtype=torch.long).to(DEVICE)
            batch_idx = torch.zeros(len(x), dtype=torch.long, device=DEVICE)
            _, reg_pred = model(x, ei, batch_idx)
            pred = reg_pred.cpu().numpy().flatten()

            actuals.append(actual)
            predictions.append(pred)

    if not actuals:
        return

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    metric_names = ['Aspect Ratio', 'Scaled Jacobian', 'Skewness']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Quality Regression: Predicted vs Actual', fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes, metric_names)):
        ax.scatter(actuals[:, i], predictions[:, i], alpha=0.5, s=30, color='steelblue')
        # Perfect prediction line
        lims = [min(actuals[:, i].min(), predictions[:, i].min()),
                max(actuals[:, i].max(), predictions[:, i].max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        # R² score
        ss_res = np.sum((actuals[:, i] - predictions[:, i])**2)
        ss_tot = np.sum((actuals[:, i] - actuals[:, i].mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        ax.set_title(f'{name}\nR² = {r2:.4f}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_model_comparison(results_dict):
    """Comprehensive comparison of all models."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Model Comparison: Bone Mesh Quality Classification',
                 fontsize=14, fontweight='bold')

    names = list(results_dict.keys())
    accs = [r['mean_acc'] for r in results_dict.values()]
    stds = [r['std_acc'] for r in results_dict.values()]
    f1s = [r['mean_f1'] for r in results_dict.values()]
    f1_stds = [r['std_f1'] for r in results_dict.values()]

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # 1. Accuracy bars with error bars
    ax = axes[0][0]
    x = np.arange(n_models)
    ax.bar(x, accs, yerr=stds, color=colors, edgecolor='black',
           capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Mean Accuracy (± std)')
    ax.set_ylim(0, 1.05)
    for i, (a, s) in enumerate(zip(accs, stds)):
        ax.text(i, a + s + 0.02, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')

    # 2. F1 bars
    ax = axes[0][1]
    ax.bar(x, f1s, yerr=f1_stds, color=colors, edgecolor='black',
           capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('F1 Score (weighted)')
    ax.set_title('Mean F1 Score (± std)')
    ax.set_ylim(0, 1.05)

    # 3. Per-fold accuracy line chart
    ax = axes[1][0]
    for i, (name, results) in enumerate(results_dict.items()):
        fold_accs = [r['accuracy'] for r in results['folds']]
        ax.plot(range(1, len(fold_accs)+1), fold_accs, 'o-',
                label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Confusion matrix of best model
    best_name = names[int(np.argmax(f1s))]
    ax = axes[1][1]
    best_results = results_dict[best_name]
    cm = confusion_matrix(best_results['all_true'], best_results['all_preds'])
    num_c = cm.shape[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=QualityLabeler.LABELS[:num_c],
                yticklabels=QualityLabeler.LABELS[:num_c])
    ax.set_title(f'Best Model: {best_name}\nAcc={best_results["mean_acc"]:.3f}')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('multi_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary table
    print("\n" + "="*70)
    print("  COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<18} {'F1 (weighted)':<18} {'Params':>10}")
    print("-"*70)
    for name in names:
        r = results_dict[name]
        params = r.get('n_params', '?')
        print(f"{name:<25} {r['mean_acc']:.4f}±{r['std_acc']:.4f}"
              f"{'':>4}{r['mean_f1']:.4f}±{r['std_f1']:.4f}"
              f"{'':>4}{params:>10}")
    print("-"*70)
    print(f"🏆 Best model: {best_name} (F1={max(f1s):.4f})")
    print("="*70)


# ============================================================
# SECTION 10: MAIN PIPELINE
# ============================================================
def run_pipeline():
    """Execute the complete ML pipeline for bone mesh quality assessment."""
    print("="*60)
    print("  ANSYS Mesh v6: ML-based FEA Mesh Quality Assessment")
    print("  for Bone Structures")
    print("  Research: GNN vs PointNet vs Graph Transformer")
    print("="*60)

    # --- Step 1: Mount Drive (Colab) ---
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except ImportError:
        print("ℹ️ Not running on Colab, using local paths")

    # --- Step 2: Parse CDB files ---
    print("\n📂 STEP 1: Parsing CDB files...")
    parser = CDBParser()
    all_data = parser.parse_directory(CONFIG['data_dir'])
    if not all_data:
        print("❌ No data found. Check data_dir in CONFIG.")
        return

    # --- Step 3: Data Exploration ---
    print("\n📊 STEP 2: Data Exploration...")
    preprocessor = MeshDataPreprocessor()
    stats_df = preprocessor.compute_point_cloud_stats(all_data)
    explore_dataset(all_data, stats_df)
    print(f"\n  📋 Per-file statistics:")
    print(stats_df[['filename', 'n_nodes', 'n_elements', 'side',
                     'x_range_mm', 'y_range_mm', 'z_range_mm']].to_string(index=False))

    # --- Step 4: Compute quality metrics ---
    print("\n📊 STEP 3: Computing FEA mesh quality metrics...")
    all_metrics = {}
    analyzer = MeshQualityAnalyzer()
    for fname, data in all_data.items():
        metrics = analyzer.compute_tet_metrics(data['nodes'], data['elements'])
        all_metrics[fname] = metrics
        if metrics:
            n_elem = len(metrics['aspect_ratio'])
            ar_mean = np.mean(metrics['aspect_ratio'])
            sj_mean = np.mean(metrics['scaled_jacobian'])
            sk_mean = np.mean(metrics['skewness'])
            print(f"  {fname}: {n_elem} elements | "
                  f"AR={ar_mean:.2f} SJ={sj_mean:.3f} Sk={sk_mean:.3f}")
        else:
            print(f"  {fname}: No elements / metrics unavailable")

    # --- Step 5: Metric correlation analysis ---
    print("\n📈 STEP 4: Metric Correlation Analysis...")
    metrics_list = [all_metrics[d] for d in all_data if all_metrics.get(d)]
    plot_metric_correlation(metrics_list)

    # --- Step 6: Label meshes ---
    print("\n🏷️ STEP 5: Labeling mesh quality...")
    data_list = []
    for fname, data in all_data.items():
        metrics = all_metrics.get(fname)
        label = QualityLabeler.label_mesh(metrics)
        # Also store regression targets
        reg_targets = None
        if metrics:
            reg_targets = np.array([
                np.mean(metrics['aspect_ratio']),
                np.mean(metrics['scaled_jacobian']),
                np.mean(metrics['skewness']),
            ])
        data_list.append({
            'filename': fname,
            'nodes_df': data['nodes'],
            'elements': data['elements'],
            'metrics': metrics,
            'label': label,
            'reg_targets': reg_targets,
        })

    labels = np.array([d['label'] for d in data_list])
    print(f"\n  Label distribution:")
    for i, name in enumerate(QualityLabeler.LABELS):
        count = np.sum(labels == i)
        pct = 100 * count / len(labels)
        print(f"    {name}: {count} ({pct:.1f}%)")

    # Visualize quality metrics distribution
    plot_quality_distribution(metrics_list, labels)

    # 3D visualization of sample meshes (one per class)
    for label_val in range(len(QualityLabeler.LABELS)):
        samples = [d for d in data_list if d['label'] == label_val]
        if samples:
            s = samples[0]
            plot_mesh_quality_3d(s['nodes_df'], s['elements'], s['metrics'],
                                 f"{QualityLabeler.LABELS[label_val]}: {s['filename']}")

    # --- Step 7: Model Summary ---
    print("\n📐 STEP 6: Model architectures...")
    num_classes = len(np.unique(labels))
    for model_name, factory in MODEL_REGISTRY.items():
        model = factory(num_classes)
        print_model_summary(model_name, model)

    # --- Step 8: Train all models ---
    models_to_train = ['pointnet', 'gat', 'graph_transformer']
    results_dict = {}

    for model_name in models_to_train:
        model_type = 'pointnet' if 'pointnet' in model_name else 'gnn'
        print(f"\n🤖 Training {model_name.upper()}...")

        # Determine if we use the registry
        is_pointnet = 'pointnet' in model_name

        # Override model creation in run_kfold to use our registry
        results = run_kfold_experiment(data_list, model_type=model_name)

        # Store parameter count
        model_tmp = MODEL_REGISTRY[model_name](num_classes)
        results['n_params'] = f"{count_parameters(model_tmp):,}"
        results_dict[model_name] = results

    # --- Step 9: Compare all models ---
    print("\n📊 STEP 7: Model Comparison...")
    plot_all_model_comparison(results_dict)

    # ROC curves
    print("\n📈 STEP 8: ROC Curves...")
    plot_roc_curves(results_dict)

    # Statistical significance
    print("\n📊 STEP 9: Statistical Significance...")
    statistical_comparison(results_dict)

    # t-SNE of best model
    best_model_name = max(results_dict, key=lambda k: results_dict[k]['mean_f1'])
    print(f"\n🔍 STEP 10: t-SNE Visualization ({best_model_name})...")
    # Note: t-SNE requires training a model first — skip in pipeline for now
    # Uncomment below if you want to run after training:
    # best_model = MODEL_REGISTRY[best_model_name](num_classes).to(DEVICE)
    # plot_tsne_latent_space(data_list, best_model, best_model_name)

    # --- Final Summary ---
    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"  Dataset: {len(data_list)} bone meshes")
    print(f"  Specimens: {len(set(d['filename'].split('_')[0] for d in data_list))}")
    print(f"  Total nodes: {sum(len(d['nodes_df']) for d in data_list):,}")
    n_elem = sum(len(d['elements']) for d in data_list)
    print(f"  Total elements: {n_elem:,}")
    print(f"\n  Models trained: {', '.join(models_to_train)}")
    print(f"  Cross-validation: {CONFIG['k_folds']}-fold")

    for name, r in results_dict.items():
        print(f"\n  {name.upper():<25} Acc: {r['mean_acc']:.4f}±{r['std_acc']:.4f} "
              f"| F1: {r['mean_f1']:.4f}±{r['std_f1']:.4f} | Params: {r['n_params']}")

    best = max(results_dict, key=lambda k: results_dict[k]['mean_f1'])
    print(f"\n  🏆 Best model: {best.upper()}")
    print("="*60)

    # Outputs saved
    print("\n📁 Saved outputs:")
    outputs = ['dataset_overview.png', 'spatial_dimensions.png',
               'mesh_quality_distribution.png', 'label_distribution.png',
               'metric_correlation.png', 'model_comparison.png',
               'multi_model_comparison.png', 'roc_curves.png']
    for o in outputs:
        exists = "✅" if os.path.exists(o) else "⏳"
        print(f"  {exists} {o}")

    return results_dict, data_list


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    results = run_pipeline()

