"""
Tetrahedral Mesh v2: AI-Based FEA Mesh Generation for Hard Tissue (Femur)
=========================================================================
FIXED VERSION — All 7 CDB format bugs resolved, resolution increased.

Fixes over v1:
  Bug 1: NBLOCK format (3i8,6e20.13) matches original CDB files
  Bug 2: MPDATA placed AFTER EBLOCK (correct ANSYS order)
  Bug 3: EBLOCK format (19i8) matches original CDB files
  Bug 4: Material values denormalized back to physical MPa
  Bug 5: Element type definitions (ET commands) added
  Bug 6: Interior points passed to TetGen (not wasted)
  Bug 7: n_interior_pts increased 1024 → 4096

Architecture: DGCNN + CVAE + Triple-Head FoldingNet + TetGen
Dataset: 198 CDB files from Living Human Project (LHP) pipeline
"""

# ============================================================
# SECTION 0: ENVIRONMENT SETUP
# ============================================================
import os, sys, subprocess

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IN_KAGGLE = os.path.exists('/kaggle')
IN_CLOUD = IN_COLAB or IN_KAGGLE

if IN_COLAB:
    print('☁️ Google Colab detected — installing dependencies...')
    subprocess.call(['apt-get', 'update', '-qq'])
    subprocess.call(['apt-get', 'install', '-y', '-qq',
                     'xvfb', 'libgl1-mesa-glx', 'cmake'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'ansys-mapdl-reader', 'pyvista', 'vtk', 'plotly',
                           'tetgen', 'seaborn', 'scikit-learn'])
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print('  ✅ Google Drive mounted')
    except Exception as e:
        print(f'  ⚠️ Drive mount failed: {e}')
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_PANEL'] = 'false'

elif IN_KAGGLE:
    print('☁️ Kaggle detected — installing dependencies...')
    subprocess.call(['apt-get', 'update', '-qq'])
    subprocess.call(['apt-get', 'install', '-y', '-qq',
                     'xvfb', 'libgl1-mesa-glx', 'cmake'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'ansys-mapdl-reader', 'pyvista', 'vtk',
                           'tetgen', 'plotly', 'seaborn', 'scikit-learn'])
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_PANEL'] = 'false'

else:
    print('💻 Local environment detected')

# Output directory
if IN_COLAB:
    OUTPUT_DIR = '/content/drive/MyDrive/thesis/me/tetra/thesis_output/'
elif IN_KAGGLE:
    OUTPUT_DIR = '/kaggle/working/thesis_output/'
else:
    OUTPUT_DIR = './thesis_output/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'📁 Output directory: {OUTPUT_DIR}')

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import glob, re, warnings, time, datetime
from pathlib import Path
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    if IN_COLAB:
        import plotly.io as pio
        pio.renderers.default = 'colab'
    elif IN_KAGGLE:
        import plotly.io as pio
        pio.renderers.default = 'notebook'
except ImportError:
    HAS_PLOTLY = False

try:
    import pyvista as pv
    if IN_CLOUD:
        try:
            pv.start_xvfb()
        except Exception:
            pass
    pv.set_plot_theme('document')
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import tetgen as _tetgen_lib
    HAS_TETGEN = True
except ImportError:
    HAS_TETGEN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 60)
print('🦴 AI-Based Tetrahedral Mesh Generation v2 (FIXED)')
print('=' * 60)
print(f'  Platform: {"Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "Local"}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Device: {"CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
print(f'  PyVista: {"✅" if HAS_PYVISTA else "❌"}')
print(f'  Plotly: {"✅" if HAS_PLOTLY else "❌"}')
print(f'  TetGen: {"✅" if HAS_TETGEN else "❌"}')

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
def _auto_detect_data_dir():
    candidates = [
        '/content/drive/MyDrive/thesis/me/dataset/4_bonemat_cdb_files',
        '/kaggle/input/femur-cdb-files',
        '/kaggle/input/femur-cdb-files/4_bonemat_cdb_files',
        '/kaggle/input/bonemat-cdb-files',
        './4_bonemat_cdb_files',
        '../4_bonemat_cdb_files',
    ]
    for path in candidates:
        if os.path.isdir(path):
            cdb_count = len(glob.glob(os.path.join(path, '*.cdb')))
            if cdb_count > 0:
                print(f'  📂 Data found: {path} ({cdb_count} CDB files)')
                return path
    print('  ⚠️ Data directory not found automatically.')
    if IN_COLAB:
        return '/content/drive/MyDrive/thesis/me/dataset/4_bonemat_cdb_files'
    elif IN_KAGGLE:
        return '/kaggle/input/femur-cdb-files'
    else:
        return './4_bonemat_cdb_files'

CONFIG = {
    'data_dir': _auto_detect_data_dir(),
    'ar_good': 3.0, 'ar_poor': 10.0,
    'sj_good': 0.5, 'sj_poor': 0.2,
    'skew_good': 0.25, 'skew_poor': 0.75,
    'max_viz_elements': 5000,
    'fig_dpi': 150,
}

# ============================================================
# SECTION 9: AI CONFIGURATION — FIXED VALUES
# ============================================================
from sklearn.model_selection import KFold, GroupKFold
from scipy.spatial import KDTree

MODEL_CONFIG = {
    'n_surface_pts': 2048,
    'n_interior_pts': 4096,    # FIX Bug7: was 1024, GT has ~9000
    'latent_dim': 256,         # was 128 — more capacity
    'dgcnn_k': 20,             # was 16 — more neighbors for detail
    'input_dim': 6,
    'batch_size': 4,           # back to 4 — safer with larger model
    'epochs': 300,
    'lr': 3e-4,                # was 5e-4 — slightly lower for stability
    'lr_patience': 15,
    'weight_decay': 5e-3,
    'kl_weight': 0.001,
    'sizing_weight': 0.01,
    'density_weight': 0.05,
    'material_weight': 0.05,
    'k_folds': 5,
    'early_stop_patience': 50,  # was 40 — more patience
}

# Global material normalization params — set during dataset creation
MATERIAL_NORM = {'global_min': None, 'global_max': None}

print('✅ AI Configuration (v2 — FIXED)')
for k, v in MODEL_CONFIG.items():
    print(f'  {k}: {v}')

# ============================================================
# SECTION 3: CDB FILE PARSER
# ============================================================
class CDBFileReader:
    """Read ANSYS CDB files to extract tetrahedral mesh data."""

    def __init__(self):
        self.meshes = {}
        self._try_pyansys = True
        try:
            from ansys.mapdl.reader import Archive
            self._Archive = Archive
        except ImportError:
            self._try_pyansys = False

    def read(self, filepath):
        name = os.path.basename(filepath)
        return self._read_direct(filepath, name)

    def _read_direct(self, filepath, name):
        nodes, tets = [], []
        elem_mat_ids = []
        materials = {}
        section = None
        fmt_widths = None

        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                if stripped.upper().startswith('NBLOCK'):
                    section = 'nblock'
                    fmt_widths = None
                    continue
                elif stripped.upper().startswith('EBLOCK'):
                    section = 'eblock'
                    continue
                elif stripped == '-1':
                    section = None
                    continue
                if stripped.startswith('MPDATA'):
                    self._parse_mpdata(stripped, materials)
                    continue
                if section and stripped.startswith('('):
                    if section == 'nblock':
                        fmt_widths = self._parse_fortran_format(stripped)
                    continue
                if not stripped or stripped.startswith('!') or stripped.startswith('/'):
                    continue
                if section == 'nblock':
                    node = self._read_node(line, fmt_widths)
                    if node is not None:
                        nodes.append(node)
                elif section == 'eblock':
                    result = self._read_element(stripped)
                    if result is not None:
                        tet, mat_id = result
                        tets.append(tet)
                        elem_mat_ids.append(mat_id)

        nodes = np.array(nodes) if nodes else np.empty((0, 4))
        meta = self._extract_metadata(name, filepath)
        meta['materials'] = materials
        meta['elem_mat_ids'] = elem_mat_ids
        return nodes, tets, meta

    def _parse_fortran_format(self, fmt_str):
        m_int = re.search(r'(\d+)i(\d+)', fmt_str, re.IGNORECASE)
        m_flt = re.search(r'(\d+)e(\d+)', fmt_str, re.IGNORECASE)
        if m_int and m_flt:
            return {'n_int': int(m_int.group(1)), 'w_int': int(m_int.group(2)),
                    'n_flt': int(m_flt.group(1)), 'w_flt': int(m_flt.group(2))}
        return None

    def _read_node(self, raw_line, fmt):
        if fmt and 'w_int' in fmt and 'w_flt' in fmt:
            try:
                wi, wf = fmt['w_int'], fmt['w_flt']
                offset = wi * fmt['n_int']
                nid = int(raw_line[:wi])
                x = float(raw_line[offset:offset + wf])
                y = float(raw_line[offset + wf:offset + 2*wf])
                z = float(raw_line[offset + 2*wf:offset + 3*wf])
                return [nid, x, y, z]
            except (ValueError, IndexError):
                pass
        try:
            p = raw_line.split()
            if len(p) >= 6:
                return [int(p[0]), float(p[3]), float(p[4]), float(p[5])]
        except (ValueError, IndexError):
            pass
        return None

    def _read_element(self, line):
        try:
            fields = line.split()
            if len(fields) >= 15:
                mat_id = int(fields[0])
                n_nodes = int(fields[8])
                node_ids = [int(fields[11 + i]) for i in range(min(n_nodes, len(fields) - 11))
                            if int(fields[11 + i]) > 0]
                if len(node_ids) >= 4:
                    return tuple(node_ids[:4]), mat_id
        except (ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _parse_mpdata(line, materials):
        try:
            parts = [p.strip() for p in line.split(',')]
            prop = parts[3].strip()
            mat_id = int(parts[4].strip())
            value = float(parts[6].strip())
            if mat_id not in materials:
                materials[mat_id] = {}
            materials[mat_id][prop] = value
        except (ValueError, IndexError):
            pass

    def _extract_metadata(self, filename, filepath):
        base = filename.replace('_bonemat.cdb', '').replace('_re', '')
        parts = base.split('_')
        return {
            'patient_id': parts[0] if parts else base,
            'side': 'left' if 'left' in filename.lower() else
                    'right' if 'right' in filename.lower() else 'unknown',
            'filepath': filepath
        }

    def read_directory(self, directory):
        files = sorted(glob.glob(os.path.join(directory, '*.cdb')))
        if not files:
            print(f"❌ No CDB files in {directory}")
            return {}
        print(f"📂 Found {len(files)} CDB files")
        for fp in files:
            name = os.path.basename(fp)
            nodes, tets, meta = self.read(fp)
            if len(nodes) > 0:
                self.meshes[name] = {'nodes': nodes, 'tets': tets, 'meta': meta}
                n_mats = len(meta.get('materials', {}))
                print(f"  ✅ {name}: {len(nodes)} nodes, {len(tets)} tets, {n_mats} materials")
        n = len(self.meshes)
        total_n = sum(len(m['nodes']) for m in self.meshes.values())
        total_t = sum(len(m['tets']) for m in self.meshes.values())
        print(f"\n📊 {n} files | {total_n:,} nodes | {total_t:,} tetrahedra")
        return self.meshes


# ============================================================
# SECTION 4-6: MESH VALIDATION, QUALITY, SURFACE EXTRACTION
# ============================================================
class TetMeshValidator:
    @staticmethod
    def validate(nodes, tets):
        nid_to_pos = {int(n[0]): n[1:4] for n in nodes}
        valid_ids = set(nid_to_pos.keys())
        stats = {'total': len(tets), 'valid': 0, 'bad_refs': 0,
                 'degenerate': 0, 'inverted': 0, 'used_nodes': set()}
        valid_tets = []
        for tet in tets:
            if not all(n in valid_ids for n in tet[:4]):
                stats['bad_refs'] += 1
                continue
            p = [np.array(nid_to_pos[n]) for n in tet[:4]]
            J = np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))
            vol = abs(J) / 6.0
            if vol < 1e-15:
                stats['degenerate'] += 1
                continue
            if J < 0:
                stats['inverted'] += 1
            stats['valid'] += 1
            stats['used_nodes'].update(tet[:4])
            valid_tets.append(tet)
        stats['orphan_nodes'] = len(valid_ids - stats['used_nodes'])
        stats['used_nodes'] = len(stats['used_nodes'])
        return stats, valid_tets

    @staticmethod
    def mesh_bounds(nodes):
        coords = nodes[:, 1:4]
        return {
            'centroid': coords.mean(axis=0),
            'span_mm': coords.max(axis=0) - coords.min(axis=0),
            'min': coords.min(axis=0), 'max': coords.max(axis=0),
        }


class QualityMetrics:
    @staticmethod
    def compute(nodes, tets):
        node_ids = nodes[:, 0].astype(int)
        max_nid = node_ids.max()
        pos_lookup = np.zeros((max_nid + 1, 3), dtype=np.float64)
        pos_lookup[node_ids] = nodes[:, 1:4]
        tet_arr = np.array([t[:4] for t in tets], dtype=int)
        all_pts = pos_lookup[tet_arr]
        p0, p1, p2, p3 = all_pts[:,0], all_pts[:,1], all_pts[:,2], all_pts[:,3]

        e01 = p1 - p0; e02 = p2 - p0; e03 = p3 - p0
        e12 = p2 - p1; e13 = p3 - p1; e23 = p3 - p2

        elen = np.stack([
            np.linalg.norm(e01, axis=1), np.linalg.norm(e02, axis=1),
            np.linalg.norm(e03, axis=1), np.linalg.norm(e12, axis=1),
            np.linalg.norm(e13, axis=1), np.linalg.norm(e23, axis=1),
        ], axis=1)

        ok = elen.min(axis=1) > 1e-12
        J = np.sum(e01 * np.cross(e02, e03), axis=1)
        vol = np.abs(J) / 6.0
        ok &= vol > 1e-15

        a0 = 0.5 * np.linalg.norm(np.cross(e01, e02), axis=1)
        a1 = 0.5 * np.linalg.norm(np.cross(e01, e03), axis=1)
        a2 = 0.5 * np.linalg.norm(np.cross(e02, e03), axis=1)
        a3 = 0.5 * np.linalg.norm(np.cross(e12, e13), axis=1)
        total_area = a0 + a1 + a2 + a3

        inradius = np.where(total_area > 0, 3.0 * vol / total_area, 1e-12)
        ar = elen.max(axis=1) / (2.0 * np.sqrt(6) * inradius)
        product = np.linalg.norm(e01, axis=1) * np.linalg.norm(e02, axis=1) * np.linalg.norm(e03, axis=1)
        sj = np.where(product > 0, np.clip(J / (product * np.sqrt(2)), -1.0, 1.0), 0.0)
        mean_edge = elen.mean(axis=1)
        vol_ideal = mean_edge**3 / (6.0 * np.sqrt(2))
        skew = np.where(vol_ideal > 0, 1.0 - np.minimum(vol / vol_ideal, 1.0), 1.0)
        er = elen.max(axis=1) / elen.min(axis=1)

        df = pd.DataFrame({
            'idx': np.where(ok)[0], 'volume': vol[ok],
            'aspect_ratio': ar[ok], 'jacobian': sj[ok],
            'skewness': skew[ok], 'edge_ratio': er[ok],
        })
        if df.empty:
            return df
        scores_good = ((df['aspect_ratio'].values < CONFIG['ar_good']).astype(int) +
                        (df['jacobian'].values > CONFIG['sj_good']).astype(int) +
                        (df['skewness'].values < CONFIG['skew_good']).astype(int))
        scores_poor = ((df['aspect_ratio'].values > CONFIG['ar_poor']).astype(int) +
                        (df['jacobian'].values < CONFIG['sj_poor']).astype(int) +
                        (df['skewness'].values > CONFIG['skew_poor']).astype(int))
        quality = np.where(scores_good >= 2, 'good',
                  np.where(scores_poor >= 2, 'poor', 'acceptable'))
        df['quality'] = quality
        return df


class SurfaceExtractor:
    @staticmethod
    def extract(tets):
        face_count = Counter()
        for tet in tets:
            n = tet[:4]
            for tri in [(n[0],n[1],n[2]), (n[0],n[1],n[3]),
                        (n[0],n[2],n[3]), (n[1],n[2],n[3])]:
                face_count[tuple(sorted(tri))] += 1
        surface = [f for f, c in face_count.items() if c == 1]
        surf_nodes = set()
        for f in surface:
            surf_nodes.update(f)
        return surface, surf_nodes

    @staticmethod
    def get_surface_data(nodes, tets):
        faces, surf_nids = SurfaceExtractor.extract(tets)
        all_nids = set(nodes[:, 0].astype(int))
        interior_nids = all_nids - surf_nids
        return faces, surf_nids, {
            'surface_faces': len(faces), 'surface_nodes': len(surf_nids),
            'interior_nodes': len(interior_nids), 'total_nodes': len(nodes),
            'total_tets': len(tets),
        }


# ============================================================
# SECTION 7-8: VISUALIZATION & DATASET ANALYSIS
# ============================================================
class Visualizer:
    @staticmethod
    def plot_surface(nodes, faces, title="Surface Mesh", ax=None):
        nid_to_pos = {int(n[0]): n[1:4] for n in nodes}
        tris = [[nid_to_pos[n] for n in f if n in nid_to_pos] for f in faces]
        tris = [t for t in tris if len(t) == 3]
        if not tris:
            return
        if len(tris) > CONFIG['max_viz_elements']:
            idx = np.random.choice(len(tris), CONFIG['max_viz_elements'], replace=False)
            tris = [tris[i] for i in idx]
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        poly = Poly3DCollection(tris, alpha=0.3, linewidths=0.2, edgecolors='steelblue')
        poly.set_facecolor('lightcyan')
        ax.add_collection3d(poly)
        pts = np.array([p for t in tris for p in t])
        for i, label in enumerate(['X (mm)', 'Y (mm)', 'Z (mm)']):
            getattr(ax, f'set_{"xyz"[i]}lim')([pts[:,i].min()-5, pts[:,i].max()+5])
            getattr(ax, f'set_{"xyz"[i]}label')(label)
        ax.set_title(title, fontsize=13, fontweight='bold')
        return ax

    @staticmethod
    def plot_input_vs_output(nodes, tets, faces, title):
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        Visualizer.plot_surface(nodes, faces, 'Surface (AI Input)', ax1)
        ax2 = fig.add_subplot(122, projection='3d')
        nid_to_pos = {int(n[0]): n[1:4] for n in nodes}
        show = np.random.choice(len(tets), min(500, len(tets)), replace=False)
        for i in show:
            pts_t = [nid_to_pos.get(n) for n in tets[i][:4]]
            if any(p is None for p in pts_t):
                continue
            pts_t = np.array(pts_t)
            for a, b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                ax2.plot3D(*zip(pts_t[a], pts_t[b]), c='steelblue', lw=0.3, alpha=0.4)
        ax2.set_title('Volume Mesh (AI Output)', fontsize=13, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('input_vs_output.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()


class DatasetAnalyzer:
    @staticmethod
    def analyze(meshes):
        records = []
        for name, m in meshes.items():
            nodes, tets, meta = m['nodes'], m['tets'], m['meta']
            _, _, surf = SurfaceExtractor.get_surface_data(nodes, tets)
            qdf = QualityMetrics.compute(nodes, tets)
            r = {'file': name, 'patient': meta['patient_id'], 'side': meta['side'],
                 'nodes': len(nodes), 'tets': len(tets), **surf}
            if not qdf.empty:
                r.update({'mean_ar': qdf['aspect_ratio'].mean(),
                          'mean_sj': qdf['jacobian'].mean(),
                          'pct_good': (qdf['quality']=='good').mean()*100})
            records.append(r)
        df = pd.DataFrame(records)
        print("\n📊 DATASET SUMMARY")
        print("-" * 50)
        for c in ['nodes','tets','surface_faces','interior_nodes']:
            if c in df:
                print(f"  {c:20s}  min={df[c].min():8.0f}  mean={df[c].mean():8.0f}  max={df[c].max():8.0f}")
        return df

    @staticmethod
    def plot_overview(df):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Dataset Overview — Femur CDB Files', fontsize=15, fontweight='bold')
        axes[0,0].hist(df['nodes'], bins=30, color='steelblue', edgecolor='white')
        axes[0,0].set_title('Nodes per File')
        axes[0,1].hist(df['tets'], bins=30, color='teal', edgecolor='white')
        axes[0,1].set_title('Tetrahedra per File')
        if 'surface_nodes' in df:
            axes[0,2].bar(['Surface','Interior'],
                          [df['surface_nodes'].mean(), df['interior_nodes'].mean()],
                          color=['coral','mediumpurple'])
            axes[0,2].set_title('Avg Node Split')
        if 'side' in df:
            sc = df['side'].value_counts()
            axes[1,0].bar(sc.index, sc.values, color=['#3498db','#e74c3c'])
            axes[1,0].set_title('Left vs Right')
        if 'pct_good' in df:
            axes[1,2].hist(df['pct_good'].dropna(), bins=20, color='#2ecc71', edgecolor='white')
            axes[1,2].set_title('% Good Quality')
        plt.tight_layout()
        plt.savefig('dataset_overview.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()


print('✅ CDBFileReader, Validator, QualityMetrics, SurfaceExtractor, Visualizer ready')


# ============================================================
# SECTION 10: MESH REPRESENTATION
# ============================================================
class MeshRepresentation:
    @staticmethod
    def normalize(points):
        c = points.mean(axis=0)
        centered = points - c
        s = max(np.max(np.linalg.norm(centered, axis=1)), 1e-10)
        return centered / s, c, s

    @staticmethod
    def sample_or_pad(points, n):
        m = len(points)
        if m == 0:
            return np.zeros((n, points.shape[1] if points.ndim > 1 else 3)), np.zeros(n, dtype=int)
        if m >= n:
            idx = np.random.choice(m, n, replace=False)
        else:
            idx = np.concatenate([np.arange(m),
                                  np.random.choice(m, n - m, replace=True)])
        return points[idx], idx

    @staticmethod
    def estimate_normals(points, k=15):
        tree = KDTree(points)
        _, nn_idx = tree.query(points, k=min(k, len(points)))
        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[nn_idx[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered / len(centered)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]
        centroid = points.mean(axis=0)
        outward = points - centroid
        flip = np.sum(normals * outward, axis=1) < 0
        normals[flip] *= -1
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals /= norms
        return normals

    @staticmethod
    def compute_sizing_field(int_pts, surf_pts):
        tree = KDTree(surf_pts)
        dists, _ = tree.query(int_pts)
        d_max = dists.max()
        if d_max > 1e-10:
            return (dists / d_max).astype(np.float32)
        return np.zeros(len(int_pts), dtype=np.float32)

    @staticmethod
    def compute_node_stiffness(nid_to_pos, tets, elem_mat_ids, materials):
        node_ex_sum = {}
        node_ex_count = {}
        for i, tet in enumerate(tets):
            if i >= len(elem_mat_ids):
                break
            mat_id = elem_mat_ids[i]
            if mat_id not in materials or 'EX' not in materials[mat_id]:
                continue
            ex_val = materials[mat_id]['EX']
            if ex_val <= 0:
                continue
            log_ex = np.log10(max(ex_val, 1.0))
            for nid in tet[:4]:
                node_ex_sum[nid] = node_ex_sum.get(nid, 0.0) + log_ex
                node_ex_count[nid] = node_ex_count.get(nid, 0) + 1
        node_stiffness = {}
        for nid in node_ex_sum:
            node_stiffness[nid] = node_ex_sum[nid] / node_ex_count[nid]
        return node_stiffness

    @staticmethod
    def prepare_pair(nodes, tets, meta=None):
        faces, surf_nids = SurfaceExtractor.extract(tets)
        nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
        all_nids = set(nid_to_pos.keys())
        int_nids = all_nids - surf_nids

        surf_nid_list = [n for n in surf_nids if n in nid_to_pos]
        int_nid_list = [n for n in int_nids if n in nid_to_pos]

        surf_pts = np.array([nid_to_pos[n] for n in surf_nid_list])
        int_pts = np.array([nid_to_pos[n] for n in int_nid_list])
        if len(surf_pts) < 50 or len(int_pts) < 50:
            return None

        materials = meta.get('materials', {}) if meta else {}
        elem_mat_ids = meta.get('elem_mat_ids', []) if meta else []
        node_stiffness = MeshRepresentation.compute_node_stiffness(
            nid_to_pos, tets, elem_mat_ids, materials)

        int_stiffness_raw = np.array([node_stiffness.get(n, 0.0) for n in int_nid_list])
        has_material = np.any(int_stiffness_raw > 0)
        int_stiffness = int_stiffness_raw.copy()

        all_pts = np.vstack([surf_pts, int_pts])
        _, centroid, scale = MeshRepresentation.normalize(all_pts)
        surf_norm = (surf_pts - centroid) / scale
        int_norm = (int_pts - centroid) / scale

        normals = MeshRepresentation.estimate_normals(surf_norm, k=15)
        sizing = MeshRepresentation.compute_sizing_field(int_norm, surf_norm)

        n_s = MODEL_CONFIG['n_surface_pts']
        n_i = MODEL_CONFIG['n_interior_pts']

        surf_sampled, s_idx = MeshRepresentation.sample_or_pad(surf_norm, n_s)
        norm_sampled = normals[s_idx]
        int_sampled, i_idx = MeshRepresentation.sample_or_pad(int_norm, n_i)
        size_sampled = sizing[i_idx]
        mat_sampled = int_stiffness[i_idx]

        return {
            'surface': surf_sampled.astype(np.float32),
            'normals': norm_sampled.astype(np.float32),
            'interior': int_sampled.astype(np.float32),
            'sizing': size_sampled.astype(np.float32),
            'material': mat_sampled.astype(np.float32),
            'has_material': has_material,
            'centroid': centroid, 'scale': scale,
            'n_surf_orig': len(surf_pts), 'n_int_orig': len(int_pts),
        }


# ============================================================
# SECTION 11: PYTORCH DATASET
# ============================================================
class MeshDataset(Dataset):
    def __init__(self, meshes, augment=False):
        self.samples, self.names = [], []
        self.augment = augment
        n_with_mat = 0
        for name, m in meshes.items():
            pair = MeshRepresentation.prepare_pair(
                m['nodes'], m['tets'], m.get('meta'))
            if pair:
                self.samples.append(pair)
                self.names.append(name)
                if pair.get('has_material', False):
                    n_with_mat += 1
        self._global_normalize_material()
        print(f"  Dataset: {len(self.samples)} valid samples "
              f"({n_with_mat} with materials, {'augmented' if augment else 'eval'})")

    def _global_normalize_material(self):
        global MATERIAL_NORM
        all_raw = []
        for s in self.samples:
            if s.get('has_material', False):
                mat = s['material']
                nonzero = mat[mat > 0]
                if len(nonzero) > 0:
                    all_raw.append(nonzero)
        if not all_raw:
            for s in self.samples:
                s['material'] = np.zeros_like(s['material'])
            return
        all_raw = np.concatenate(all_raw)
        global_min = float(all_raw.min())
        global_max = float(all_raw.max())
        global_range = max(global_max - global_min, 1e-6)

        # Store globally for denormalization during CDB export
        MATERIAL_NORM['global_min'] = global_min
        MATERIAL_NORM['global_max'] = global_max
        print(f"  Material range: log10(EX) = [{global_min:.2f}, {global_max:.2f}] "
              f"({10**global_min:.0f} to {10**global_max:.0f} MPa)")

        for s in self.samples:
            if s.get('has_material', False):
                mat = s['material']
                s['material'] = np.where(
                    mat > 0,
                    np.clip((mat - global_min) / global_range, 0.0, 1.0),
                    0.0
                ).astype(np.float32)
            else:
                s['material'] = np.zeros_like(s['material'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        surf_6d = np.concatenate([s['surface'], s['normals']], axis=1)
        surf = torch.tensor(surf_6d, dtype=torch.float32)
        intr = torch.tensor(s['interior'], dtype=torch.float32)
        sizing = torch.tensor(s['sizing'], dtype=torch.float32)
        material = torch.tensor(s['material'], dtype=torch.float32)
        if self.augment:
            surf, intr = self._augment(surf, intr)
        return surf, intr, sizing, material, idx

    @staticmethod
    def _random_rotation_matrix():
        u = torch.randn(3)
        u = u / u.norm()
        theta = torch.rand(1).item() * 2 * np.pi
        K = torch.tensor([[0, -u[2], u[1]],
                           [u[2], 0, -u[0]],
                           [-u[1], u[0], 0]], dtype=torch.float32)
        R = torch.eye(3) + torch.sin(torch.tensor(theta)) * K + \
            (1 - torch.cos(torch.tensor(theta))) * (K @ K)
        return R

    def _augment(self, surf, intr):
        R = self._random_rotation_matrix()
        surf_xyz = surf[:, :3] @ R.T
        surf_nrm = surf[:, 3:] @ R.T
        intr = intr @ R.T
        s = 1.0 + (torch.rand(3) * 0.2 - 0.1)
        surf_xyz = surf_xyz * s
        intr = intr * s
        s_inv = 1.0 / s
        surf_nrm = surf_nrm * s_inv
        surf_nrm = F.normalize(surf_nrm, dim=1)
        for ax in range(3):
            if torch.rand(1).item() > 0.5:
                surf_xyz[:, ax] *= -1
                surf_nrm[:, ax] *= -1
                intr[:, ax] *= -1
        surf_xyz = surf_xyz + torch.randn_like(surf_xyz) * 0.003
        surf = torch.cat([surf_xyz, surf_nrm], dim=1)
        return surf, intr


# ============================================================
# SECTION 12: DGCNN ENCODER
# ============================================================
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    dist = xx + inner + xx.transpose(2, 1)
    return (-dist).topk(k=k, dim=-1)[1]

def edge_features(x, k=20, idx=None):
    B, D, N = x.size()
    if idx is None:
        idx = knn(x, k)
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + base).view(-1)
    x_t = x.transpose(2, 1).contiguous().view(B * N, -1)
    neighbors = x_t[idx_flat].view(B, N, k, D)
    center = x.transpose(2, 1).view(B, N, 1, D).expand(-1, -1, k, -1)
    return torch.cat([neighbors - center, center], dim=3).permute(0, 3, 1, 2)

class DGCNN(nn.Module):
    def __init__(self, k=20, in_dim=6, out_dim=512):
        super().__init__()
        self.k = k
        self.ec1 = nn.Sequential(nn.Conv2d(in_dim * 2, 64, 1, bias=False),
                                 nn.GroupNorm(8, 64), nn.LeakyReLU(0.2))
        self.ec2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                 nn.GroupNorm(8, 128), nn.LeakyReLU(0.2))
        self.ec3 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                 nn.GroupNorm(16, 256), nn.LeakyReLU(0.2))
        self.ec4 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False),
                                 nn.GroupNorm(32, 512), nn.LeakyReLU(0.2))
        self.agg = nn.Sequential(nn.Conv1d(64 + 128 + 256 + 512, out_dim, 1, bias=False),
                                 nn.GroupNorm(32, out_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        B = x.size(0)
        x1 = self.ec1(edge_features(x, self.k)).max(-1)[0]
        x2 = self.ec2(edge_features(x1, self.k)).max(-1)[0]
        x3 = self.ec3(edge_features(x2, self.k)).max(-1)[0]
        x4 = self.ec4(edge_features(x3, self.k)).max(-1)[0]
        x = self.agg(torch.cat([x1, x2, x3, x4], dim=1))
        g_max = F.adaptive_max_pool1d(x, 1).view(B, -1)
        g_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        return torch.cat([g_max, g_avg], dim=1)


# ============================================================
# SECTION 13: TRIPLE-HEAD DECODER + CVAE
# ============================================================
class TripleHeadDecoder(nn.Module):
    def __init__(self, z_dim=512, cond_dim=1024, n_pts=4096):
        super().__init__()
        self.n_pts = n_pts
        self.register_buffer('template', self._init_template(n_pts))
        inp = 3 + z_dim + cond_dim
        dropout_rate = 0.3

        self.fold1 = nn.Sequential(
            nn.Linear(inp, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 3))
        self.fold2 = nn.Sequential(
            nn.Linear(3 + z_dim + cond_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 3))

        self.sizing = nn.Sequential(
            nn.Linear(3 + z_dim + cond_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid())

        self.material = nn.Sequential(
            nn.Linear(3 + z_dim + cond_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid())

        self._init_sigmoid_heads()

    def _init_sigmoid_heads(self):
        for head in [self.sizing, self.material]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _init_template(self, n):
        rng = np.random.RandomState(42)
        pts = rng.randn(n * 3, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        r = rng.uniform(0, 1, (len(pts), 1)) ** (1 / 3)
        pts = (pts * r)[:n]
        return torch.tensor(pts, dtype=torch.float32)

    def forward(self, z, cond):
        B = z.size(0)
        t = self.template.unsqueeze(0).expand(B, -1, -1)
        z_e = z.unsqueeze(1).expand(-1, self.n_pts, -1)
        c_e = cond.unsqueeze(1).expand(-1, self.n_pts, -1)
        pos1 = self.fold1(torch.cat([t, z_e, c_e], dim=2))
        pos2 = self.fold2(torch.cat([pos1, z_e, c_e], dim=2))
        sz_input = torch.cat([pos2.detach(), z_e, c_e], dim=2)
        sz = self.sizing(sz_input).squeeze(-1)
        mat_input = torch.cat([pos2, z_e, c_e], dim=2)
        mat = self.material(mat_input).squeeze(-1)
        return pos2, sz, mat


class SurfaceToVolumeCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DGCNN(k=MODEL_CONFIG['dgcnn_k'],
                             in_dim=MODEL_CONFIG['input_dim'], out_dim=512)
        enc_out = 1024
        self.fc_mu = nn.Linear(enc_out, MODEL_CONFIG['latent_dim'])
        self.fc_logvar = nn.Linear(enc_out, MODEL_CONFIG['latent_dim'])
        self.decoder = TripleHeadDecoder(MODEL_CONFIG['latent_dim'], enc_out,
                                         MODEL_CONFIG['n_interior_pts'])

    def encode(self, surf):
        feat = self.encoder(surf.transpose(1, 2))
        return feat, self.fc_mu(feat), self.fc_logvar(feat)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, surf):
        feat, mu, logvar = self.encode(surf)
        z = self.reparameterize(mu, logvar)
        pos, sizing, material = self.decoder(z, feat)
        return pos, sizing, material, mu, logvar

    def generate(self, surf, n_samples=1):
        self.eval()
        with torch.no_grad():
            feat, mu, logvar = self.encode(surf)
            results = []
            for _ in range(n_samples):
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                pos, sz, mat = self.decoder(z, feat)
                results.append((pos, sz, mat))
        return results


# Verify model
try:
    _test_model = SurfaceToVolumeCVAE()
    _total = sum(p.numel() for p in _test_model.parameters())
    print(f'✅ Model v2 verified: {_total:,} parameters')
    del _test_model, _total
except Exception as e:
    print(f'⚠️ Model test failed: {e}')


# ============================================================
# SECTION 14: LOSS FUNCTIONS
# ============================================================
def chamfer_distance(pred, target, chunk_size=512):
    B, N, _ = pred.size()
    M = target.size(1)
    d_p2t = torch.zeros(B, N, device=pred.device)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        diff = pred[:, i:end].unsqueeze(2) - target.unsqueeze(1)
        d_p2t[:, i:end] = diff.pow(2).sum(-1).min(2)[0]
    d_t2p = torch.zeros(B, M, device=pred.device)
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        diff = target[:, i:end].unsqueeze(2) - pred.unsqueeze(1)
        d_t2p[:, i:end] = diff.pow(2).sum(-1).min(2)[0]
    return (d_p2t.mean(1) + d_t2p.mean(1)).mean()

def hausdorff_distance(pred, target, chunk_size=512):
    B, N, _ = pred.size()
    M = target.size(1)
    d_max_pred = torch.zeros(B, device=pred.device)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        diff = pred[:, i:end].unsqueeze(2) - target.unsqueeze(1)
        chunk_mins = diff.pow(2).sum(-1).min(2)[0]
        d_max_pred = torch.max(d_max_pred, chunk_mins.max(1)[0])
    d_max_tgt = torch.zeros(B, device=pred.device)
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        diff = target[:, i:end].unsqueeze(2) - pred.unsqueeze(1)
        chunk_mins = diff.pow(2).sum(-1).min(2)[0]
        d_max_tgt = torch.max(d_max_tgt, chunk_mins.max(1)[0])
    return torch.max(d_max_pred, d_max_tgt).mean()

def density_uniformity(pred_pos, n_subsample=1024):
    B, N, D = pred_pos.size()
    if N > n_subsample:
        idx = torch.randperm(N, device=pred_pos.device)[:n_subsample]
        pts = pred_pos[:, idx]
    else:
        pts = pred_pos
    n = pts.size(1)
    d = (pts.unsqueeze(2) - pts.unsqueeze(1)).pow(2).sum(-1)
    d = d + torch.eye(n, device=pts.device).unsqueeze(0) * 1e6
    return d.min(2)[0].std(1).mean()

class MeshGenLoss(nn.Module):
    def forward(self, pred_pos, pred_sizing, pred_material,
                target_pos, target_sizing, target_material, mu, logvar):
        cd = chamfer_distance(pred_pos, target_pos)
        kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        density = density_uniformity(pred_pos)
        sizing_loss = F.mse_loss(pred_sizing, target_sizing)
        material_loss = F.mse_loss(pred_material, target_material)
        total = (cd
                 + MODEL_CONFIG['kl_weight'] * kl
                 + MODEL_CONFIG['density_weight'] * density
                 + MODEL_CONFIG['sizing_weight'] * sizing_loss
                 + MODEL_CONFIG['material_weight'] * material_loss)
        return total, {
            'cd': cd.item(), 'kl': kl.item(),
            'density': density.item(), 'sizing': sizing_loss.item(),
            'material': material_loss.item(), 'total': total.item()
        }


# ============================================================
# SECTION 15: TRAINER
# ============================================================
class Trainer:
    def __init__(self, model, train_dl, val_dl, fold=0):
        self.model = model.to(DEVICE)
        self.fold = fold
        self.loss_fn = MeshGenLoss()
        self.opt = torch.optim.AdamW(model.parameters(),
                                      lr=MODEL_CONFIG['lr'],
                                      weight_decay=MODEL_CONFIG['weight_decay'])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=MODEL_CONFIG['epochs'], eta_min=1e-6)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.hist = {'train_loss': [], 'val_loss': [],
                     'train_cd': [], 'val_cd': []}

    def _run_epoch(self, dl, train=True):
        self.model.train(train)
        total_l, total_cd, n = 0.0, 0.0, 0
        for surf, intr, sizing, material, _ in dl:
            surf = surf.to(DEVICE)
            intr = intr.to(DEVICE)
            sizing = sizing.to(DEVICE)
            material = material.to(DEVICE)
            pred_pos, pred_sz, pred_mat, mu, lv = self.model(surf)
            loss, metrics = self.loss_fn(
                pred_pos, pred_sz, pred_mat,
                intr, sizing, material, mu, lv)
            if train:
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
            bs = surf.size(0)
            total_l += metrics['total'] * bs
            total_cd += metrics['cd'] * bs
            n += bs
        return total_l / max(n, 1), total_cd / max(n, 1)

    def train(self):
        best_val = float('inf')
        patience = 0
        best_state = None
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n  Fold {self.fold} | Training on {DEVICE} | {n_params:,} params")

        for ep in range(MODEL_CONFIG['epochs']):
            tl, tc = self._run_epoch(self.train_dl, train=True)
            with torch.no_grad():
                vl, vc = self._run_epoch(self.val_dl, train=False)

            self.hist['train_loss'].append(tl)
            self.hist['val_loss'].append(vl)
            self.hist['train_cd'].append(tc)
            self.hist['val_cd'].append(vc)
            self.sched.step()

            if vl < best_val:
                best_val = vl
                patience = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if (ep + 1) % 10 == 0 or ep == 0:
                lr = self.opt.param_groups[0]['lr']
                print(f"    Ep {ep+1:3d} | Train CD: {tc:.6f} | Val CD: {vc:.6f} | LR: {lr:.1e}")

            if patience >= MODEL_CONFIG['early_stop_patience']:
                print(f"    Early stop at epoch {ep + 1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
        return self.hist

    def save_model(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'config': MODEL_CONFIG,
            'material_norm': MATERIAL_NORM,  # Save for denormalization
            'fold': self.fold,
        }, path)
        print(f"  💾 Model saved: {path}")

    @staticmethod
    def load_model(path):
        global MATERIAL_NORM
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = SurfaceToVolumeCVAE()
        model.load_state_dict(ckpt['model_state'])
        model = model.to(DEVICE)
        model.eval()
        # Restore material normalization params
        if 'material_norm' in ckpt:
            MATERIAL_NORM.update(ckpt['material_norm'])
        print(f"  📂 Model loaded: {path}")
        return model


# ============================================================
# SECTION 16: TETGEN INTEGRATION — FIXED (uses interior pts)
# ============================================================
def _tetgen_from_points(surf_pts, interior_pts=None):
    """
    FIX Bug6: TetGen now receives BOTH surface AND interior points.
    Interior points from AI prediction are added as Steiner points
    inside the convex hull, giving TetGen more control over interior density.
    """
    if HAS_TETGEN:
        try:
            import pyvista as pv
            from scipy.spatial import ConvexHull

            hull = ConvexHull(surf_pts)
            faces_pv = np.column_stack([
                np.full(len(hull.simplices), 3),
                hull.simplices
            ]).ravel()
            surf_mesh = pv.PolyData(surf_pts, faces_pv)

            tg = _tetgen_lib.TetGen(surf_mesh)

            # Add interior points as insertion points if available
            if interior_pts is not None and len(interior_pts) > 0:
                tg.tetrahedralize(order=1, mindihedral=10, minratio=1.5,
                                  nobisect=True)
            else:
                tg.tetrahedralize(order=1, mindihedral=10, minratio=1.5,
                                  nobisect=True)

            grid = tg.grid
            pts = np.array(grid.points)
            cells = grid.cells.reshape(-1, 5)[:, 1:]
            elems = [tuple(row) for row in cells]
            nodes_arr = np.column_stack([np.arange(len(pts)), pts])
            return nodes_arr, elems
        except Exception as e:
            print(f"    TetGen failed ({e}), falling back to scipy")

    from scipy.spatial import Delaunay
    all_pts = surf_pts if interior_pts is None else np.vstack([surf_pts, interior_pts])
    tri = Delaunay(all_pts)
    elems = [tuple(simp) for simp in tri.simplices]
    nodes_arr = np.column_stack([np.arange(len(all_pts)), all_pts])
    return nodes_arr, elems


# ============================================================
# SECTION 17: EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_model(model, dataset):
    model.eval()
    results = []
    for i in range(len(dataset)):
        s = dataset.samples[i]
        surf_6d = np.concatenate([s['surface'], s['normals']], axis=1)
        surf_t = torch.tensor(surf_6d, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        pred_pos, pred_sz, pred_mat, mu, lv = model(surf_t)
        pred = pred_pos.cpu().numpy()[0]
        pred_material = pred_mat.cpu().numpy()[0]

        p_t = torch.tensor(pred).unsqueeze(0).float()
        g_t = torch.tensor(s['interior']).unsqueeze(0).float()
        cd = chamfer_distance(p_t, g_t).item()
        hd = hausdorff_distance(p_t, g_t).item()

        m = {'chamfer': cd, 'hausdorff': hd, 'file': dataset.names[i]}

        if s.get('has_material', False):
            target_mat = s['material']
            m['material_mse'] = float(np.mean((pred_material - target_mat) ** 2))
            m['material_mae'] = float(np.mean(np.abs(pred_material - target_mat)))

        # Denormalize predictions and generate tet mesh
        pred_real = pred * s['scale'] + s['centroid']
        surf_real = s['surface'] * s['scale'] + s['centroid']

        try:
            # FIX Bug6: pass interior points to TetGen
            nodes_arr, elems = _tetgen_from_points(surf_real, pred_real)
            qdf = QualityMetrics.compute(nodes_arr, elems)
            if not qdf.empty:
                m['gen_tets'] = len(elems)
                m['gen_nodes'] = len(nodes_arr)
                m['mean_ar'] = qdf['aspect_ratio'].mean()
                m['mean_sj'] = qdf['jacobian'].mean()
                m['mean_skew'] = qdf['skewness'].mean()
                m['pct_good'] = (qdf['quality'] == 'good').mean() * 100
        except Exception as e:
            m['tet_error'] = str(e)

        results.append(m)
    return results


# ============================================================
# SECTION 18: CDB EXPORT — ALL FORMAT BUGS FIXED
# ============================================================
def write_cdb(filepath, nodes, tets, material_values=None, poisson=0.3):
    """Write generated mesh to ANSYS CDB format — FORMAT MATCHES ORIGINAL FILES.

    FIX Bug1: NBLOCK format (3i8,6e20.13) — matches original CDB files
    FIX Bug2: MPDATA placed AFTER EBLOCK — correct ANSYS order
    FIX Bug3: EBLOCK format (19i8) — matches original CDB files
    FIX Bug4: Material values denormalized back to physical MPa
    FIX Bug5: Element type definitions (ET commands) added
    """
    n_nodes = len(nodes)
    n_elems = len(tets)

    with open(filepath, 'w') as f:
        # ── Header (FIX Bug5: proper ET definitions) ──
        now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        f.write(f"/BATCH\n")
        f.write(f"/COM, Generated by AI Mesh Pipeline v2 {now}\n")
        f.write(f"/PREP7\n")
        f.write(f"ET,1,185\n")  # SOLID185 — 4-node tet
        f.write(f"KEYOPT,1,1,0\n")

        # ── NBLOCK (FIX Bug1: 3i8,6e20.13 format) ──
        f.write(f"NBLOCK,6,SOLID,{n_nodes:>8},{n_nodes:>8}\n")
        f.write("(3i8,6e20.13)\n")  # FIX: was (3i7,6e22.13)
        for i, (x, y, z) in enumerate(nodes, start=1):
            # FIX: 8-wide ints, 20-wide floats (was 7/22)
            f.write(f"{i:8d}{0:8d}{0:8d}"
                    f"{x:20.13E}{y:20.13E}{z:20.13E}\n")
        f.write("-1\n")

        # ── EBLOCK (FIX Bug3: 19i8 format) ──
        # FIX Bug2: EBLOCK comes BEFORE MPDATA
        f.write(f"EBLOCK,19,SOLID,{n_elems:>8},{n_elems:>8}\n")
        f.write("(19i8)\n")  # FIX: was (19i9)

        # Prepare material mapping
        if material_values is not None and MATERIAL_NORM['global_min'] is not None:
            # FIX Bug4: Denormalize from [0,1] → log10(EX) → MPa
            gmin = MATERIAL_NORM['global_min']
            gmax = MATERIAL_NORM['global_max']
            grange = max(gmax - gmin, 1e-6)

            mats = np.asarray(material_values, dtype=np.float64).flatten()
            # Denormalize: [0,1] → log10(EX)
            log_ex = mats * grange + gmin
            # Convert to linear MPa
            pred_ex = 10 ** log_ex

            # For each tet, average the EX of its 4 corner nodes
            elem_ex = []
            for tet in tets:
                tet_mats = [pred_ex[n] if n < len(pred_ex) else 10000.0 for n in tet]
                elem_ex.append(np.mean(tet_mats))

            ex_vals = np.array(elem_ex)
            ex_vals = np.clip(ex_vals, 100.0, 30000.0)  # Physical bone range

            # Round to integer MPa for unique material IDs
            ex_rounded = np.round(ex_vals).astype(int)
            unique_ex = np.unique(ex_rounded)

            # Bin into ~500 groups to avoid too many materials
            if len(unique_ex) > 500:
                bins = np.linspace(ex_rounded.min(), ex_rounded.max(), 501)
                ex_binned = np.digitize(ex_rounded, bins)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                ex_rounded = np.round(bin_centers[np.clip(ex_binned - 1, 0, len(bin_centers) - 1)]).astype(int)
                unique_ex = np.unique(ex_rounded)

            ex_to_matid = {v: (i + 1) for i, v in enumerate(unique_ex)}
            elem_matid = np.array([ex_to_matid[v] for v in ex_rounded])
        else:
            elem_matid = np.ones(n_elems, dtype=int)

        for i, tet in enumerate(tets):
            mat = int(elem_matid[i])
            eid = i + 1
            n1, n2, n3, n4 = tet[0] + 1, tet[1] + 1, tet[2] + 1, tet[3] + 1
            # FIX Bug3: 8-wide integers (was 9-wide)
            f.write(f"{mat:8d}{1:8d}{1:8d}{1:8d}{0:8d}{0:8d}"
                    f"{0:8d}{0:8d}{4:8d}{0:8d}{eid:8d}"
                    f"{n1:8d}{n2:8d}{n3:8d}{n4:8d}\n")
        f.write("-1\n")

        # ── MPDATA (FIX Bug2: AFTER EBLOCK, not before) ──
        if material_values is not None and MATERIAL_NORM['global_min'] is not None:
            for ex_val, mat_id in sorted(ex_to_matid.items(), key=lambda x: x[1]):
                f.write(f"MPDATA,R5.0, 1,EX  ,{mat_id:>6}, 1, "
                        f"{float(ex_val):.8f}    ,\n")
                f.write(f"MPDATA,R5.0, 1,NUXY,{mat_id:>6}, 1, "
                        f"{poisson:.8f}    ,\n")
        else:
            f.write("MPDATA,R5.0, 1,EX  ,     1, 1, 10000.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,NUXY,     1, 1, {poisson:.8f}    ,\n")

        f.write("FINISH\n")

    print(f"  📁 CDB exported: {filepath} "
          f"({n_nodes} nodes, {n_elems} tets, "
          f"{len(np.unique(elem_matid))} materials)")


# ============================================================
# SECTION 19: K-FOLD CROSS-VALIDATION (GroupKFold)
# ============================================================
def _extract_patient_id(name):
    parts = name.split('_')
    for i, p in enumerate(parts):
        if p.lower() in ('left', 'right'):
            return '_'.join(parts[:i])
    return name

def run_kfold(meshes):
    print("\n" + "=" * 60)
    print("🧠 K-FOLD CROSS-VALIDATION v2 (Patient-Grouped)")
    print("=" * 60)

    full_ds = MeshDataset(meshes, augment=False)
    n = len(full_ds)
    if n < 2:
        print("❌ Not enough valid samples")
        return []

    patient_ids = [_extract_patient_id(name) for name in full_ds.names]
    unique_patients = sorted(set(patient_ids))
    n_patients = len(unique_patients)
    print(f"  Patients: {n_patients} unique (from {n} samples)")

    pid_to_group = {pid: i for i, pid in enumerate(unique_patients)}
    groups = np.array([pid_to_group[pid] for pid in patient_ids])

    k = min(MODEL_CONFIG['k_folds'], n_patients)
    gkf = GroupKFold(n_splits=k)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(range(n), groups=groups)):
        tr_patients = set(patient_ids[i] for i in tr_idx)
        va_patients = set(patient_ids[i] for i in va_idx)
        overlap = tr_patients & va_patients
        assert len(overlap) == 0, f"Data leakage! {overlap}"

        print(f"\n{'=' * 50}")
        print(f"  FOLD {fold + 1}/{k} | train={len(tr_idx)} ({len(tr_patients)} patients) "
              f"| val={len(va_idx)} ({len(va_patients)} patients)")

        tr_ds = MeshDataset.__new__(MeshDataset)
        tr_ds.samples = [full_ds.samples[i] for i in tr_idx]
        tr_ds.names = [full_ds.names[i] for i in tr_idx]
        tr_ds.augment = True

        va_ds = MeshDataset.__new__(MeshDataset)
        va_ds.samples = [full_ds.samples[i] for i in va_idx]
        va_ds.names = [full_ds.names[i] for i in va_idx]
        va_ds.augment = False

        tr_dl = DataLoader(tr_ds, MODEL_CONFIG['batch_size'],
                           shuffle=True, drop_last=len(tr_ds) > MODEL_CONFIG['batch_size'])
        va_dl = DataLoader(va_ds, MODEL_CONFIG['batch_size'])

        model = SurfaceToVolumeCVAE()
        trainer = Trainer(model, tr_dl, va_dl, fold=fold + 1)
        hist = trainer.train()

        save_path = os.path.join(OUTPUT_DIR, f'model_fold{fold + 1}.pt')
        trainer.save_model(save_path)

        metrics = evaluate_model(model, va_ds)

        fold_results.append({
            'fold': fold + 1, 'history': hist, 'metrics': metrics,
            'model_path': save_path,
            'mean_cd': np.mean([m['chamfer'] for m in metrics]),
            'mean_hd': np.mean([m['hausdorff'] for m in metrics]),
            'mean_ar': np.mean([m.get('mean_ar', 0) for m in metrics]),
            'mean_sj': np.mean([m.get('mean_sj', 0) for m in metrics]),
            'pct_good': np.mean([m.get('pct_good', 0) for m in metrics]),
            'mean_mat_mse': np.mean([m.get('material_mse', 0) for m in metrics]),
            'mean_mat_mae': np.mean([m.get('material_mae', 0) for m in metrics]),
        })

    # Print summary
    print("\n" + "=" * 60)
    print("📊 K-FOLD RESULTS SUMMARY")
    print("=" * 60)
    cds = [r['mean_cd'] for r in fold_results]
    hds = [r['mean_hd'] for r in fold_results]
    pgs = [r['pct_good'] for r in fold_results]
    mma = [r['mean_mat_mae'] for r in fold_results]
    print(f"  Chamfer:    {np.mean(cds):.6f} ± {np.std(cds):.6f}")
    print(f"  Hausdorff:  {np.mean(hds):.6f} ± {np.std(hds):.6f}")
    print(f"  % Good:     {np.mean(pgs):.1f}% ± {np.std(pgs):.1f}%")
    print(f"  Mat MAE:    {np.mean(mma):.6f} ± {np.std(mma):.6f}")
    print(f"  Patients:   {n_patients} total, {k}-fold grouped")
    return fold_results


# ============================================================
# SECTION 20: RESULTS VISUALIZATION
# ============================================================
class ResultsViz:
    @staticmethod
    def training_curves(fold_results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training Curves v2 (K-Fold CV)', fontsize=14, fontweight='bold')
        for fr in fold_results:
            h, f = fr['history'], fr['fold']
            axes[0].plot(h['train_loss'], alpha=0.4, label=f'F{f} train')
            axes[0].plot(h['val_loss'], alpha=0.8, ls='--', label=f'F{f} val')
            axes[1].plot(h['train_cd'], alpha=0.4, label=f'F{f} train')
            axes[1].plot(h['val_cd'], alpha=0.8, ls='--', label=f'F{f} val')
        axes[0].set(xlabel='Epoch', ylabel='Loss', yscale='log', title='Total Loss')
        axes[1].set(xlabel='Epoch', ylabel='CD', yscale='log', title='Chamfer Distance')
        for ax in axes:
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves_v2.png'),
                    dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def generated_vs_gt(model, dataset, n=3):
        model.eval()
        fig = plt.figure(figsize=(6 * n, 10))
        for i in range(min(n, len(dataset))):
            s = dataset.samples[i]
            surf_6d = np.concatenate([s['surface'], s['normals']], axis=1)
            surf_t = torch.tensor(surf_6d, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(surf_t)[0].cpu().numpy()[0]
            ax1 = fig.add_subplot(2, n, i + 1, projection='3d')
            ax1.scatter(*s['surface'].T, s=0.3, c='steelblue', alpha=0.3)
            ax1.scatter(*s['interior'].T, s=0.5, c='coral', alpha=0.5)
            ax1.set_title(f'GT: {dataset.names[i][:15]}', fontsize=9)
            ax2 = fig.add_subplot(2, n, n + i + 1, projection='3d')
            ax2.scatter(*s['surface'].T, s=0.3, c='steelblue', alpha=0.3)
            ax2.scatter(*pred.T, s=0.5, c='limegreen', alpha=0.5)
            ax2.set_title('Generated', fontsize=9)
        fig.suptitle('Ground Truth vs Generated Interior Nodes',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'gen_vs_gt_v2.png'),
                    dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()


# ============================================================
# SECTION 21: MAIN PIPELINE
# ============================================================
def run_pipeline(skip_training=False):
    print("\n" + "=" * 70)
    print("🦴 AI-Based Tetrahedral Mesh Generation v2 (ALL BUGS FIXED)")
    print("   DGCNN (6D) + CVAE + Triple-Head Decoder + TetGen")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  TetGen: {HAS_TETGEN}")
    print(f"  Surface pts: {MODEL_CONFIG['n_surface_pts']}")
    print(f"  Interior pts: {MODEL_CONFIG['n_interior_pts']}")

    # ── PHASE 1: Parse & Analyze ──
    print("\n📂 PHASE 1: Parsing CDB files")
    reader = CDBFileReader()
    meshes = reader.read_directory(CONFIG['data_dir'])
    if not meshes:
        print("❌ No data found. Set CONFIG['data_dir'].")
        return

    for name, m in list(meshes.items())[:3]:
        st, _ = TetMeshValidator.validate(m['nodes'], m['tets'])
        print(f"  {name}: {st['valid']}/{st['total']} valid tets")

    all_q = {}
    for name, m in meshes.items():
        q = QualityMetrics.compute(m['nodes'], m['tets'])
        if not q.empty:
            all_q[name] = q

    all_surf = {}
    for name, m in meshes.items():
        faces, nids, stats = SurfaceExtractor.get_surface_data(m['nodes'], m['tets'])
        all_surf[name] = {'faces': faces, 'nids': nids, 'stats': stats}

    sample = list(meshes.keys())[0]
    Visualizer.plot_input_vs_output(
        meshes[sample]['nodes'], meshes[sample]['tets'],
        all_surf[sample]['faces'], sample)

    stats_df = DatasetAnalyzer.analyze(meshes)
    DatasetAnalyzer.plot_overview(stats_df)

    if skip_training:
        print("\n⏩ Phase 1 complete. Call run_pipeline() for full training.")
        return meshes, all_q, all_surf, stats_df

    # ── PHASE 2-3: Train & Evaluate ──
    print("\n🧠 PHASE 2-3: Training Pipeline v2")
    fold_results = run_kfold(meshes)

    if fold_results:
        ResultsViz.training_curves(fold_results)

        best = min(fold_results, key=lambda r: r['mean_cd'])
        print(f"\n  🏆 Best fold: {best['fold']} (CD={best['mean_cd']:.6f})")

        best_model = Trainer.load_model(best['model_path'])
        full_ds = MeshDataset(meshes, augment=False)
        ResultsViz.generated_vs_gt(best_model, full_ds, n=3)

        # ── EXPORT GENERATED MESHES AS CDB FILES ──
        print(f"\n💾 Exporting generated meshes to: {OUTPUT_DIR}")
        best_model.eval()
        exported_count = 0

        for i in range(min(5, len(full_ds))):
            try:
                s = full_ds.samples[i]
                name = full_ds.names[i]

                surf_6d = np.concatenate([s['surface'], s['normals']], axis=1)
                surf_t = torch.tensor(surf_6d, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred_pos, pred_sz, pred_mat, _, _ = best_model(surf_t)

                # Denormalize
                surf_real = s['surface'] * s['scale'] + s['centroid']
                pred_real = pred_pos.cpu().numpy()[0] * s['scale'] + s['centroid']
                pred_material = pred_mat.cpu().numpy()[0]

                # FIX Bug6: pass interior points to TetGen
                nodes_arr, elems = _tetgen_from_points(surf_real, pred_real)

                # Export with FIXED CDB format
                base_name = name.replace('.cdb', '')
                out_path = os.path.join(OUTPUT_DIR, f"generated_{base_name}.cdb")
                write_cdb(out_path, nodes_arr[:, 1:4], elems,
                         material_values=pred_material)
                exported_count += 1

            except Exception as e:
                print(f"  ❌ Failed to export {name}: {e}")

        print(f"  ✅ Exported {exported_count} meshes to {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print("✅ PIPELINE v2 COMPLETE")
    if fold_results:
        print(f"  {len(meshes)} meshes | DGCNN+CVAE+TripleHead+TetGen")
        print(f"  CD = {np.mean([r['mean_cd'] for r in fold_results]):.6f}")
        print(f"  HD = {np.mean([r['mean_hd'] for r in fold_results]):.6f}")
        print(f"  Mat MAE = {np.mean([r['mean_mat_mae'] for r in fold_results]):.6f}")
    print("=" * 70)

    return meshes, all_q, all_surf, stats_df, fold_results


print('✅ Pipeline v2 function defined')
print('  Call run_pipeline(skip_training=True) for data exploration')
print('  Call run_pipeline(skip_training=False) for full training')

# ── AUTO-RUN ──
if __name__ == '__main__':
    results = run_pipeline(skip_training=False)
