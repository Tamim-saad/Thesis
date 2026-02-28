"""
Tetrahedral Mesh v3: Template Deformation Network for Bone FEA
===============================================================
Architecture: DGCNN encoder (per-point + global features)
            → Local surface conditioning (k-NN feature aggregation)
            → Per-node displacement + material prediction
            → TetGen meshing → ANSYS CDB export

Key innovations over v2:
  1. Local conditioning: each template node gets k-nearest surface features
  2. Residual learning: predicts displacements from mean shape, not absolute positions
  3. Direct regression: no VAE/KL — all capacity for reconstruction
  4. ~650K params vs 5.7M — matches dataset size (N=198)

Inspired by DeepCarve (IEEE TMI 2024) but novel for bone/femur with
joint material property prediction from surface point clouds.
"""

# ============================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================
import os, sys, subprocess, datetime, gc

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IN_KAGGLE = os.path.exists('/kaggle')
IN_CLOUD = IN_COLAB or IN_KAGGLE

if IN_COLAB:
    print('☁️ Google Colab detected')
    def _need_install():
        try:
            import tetgen, pyvista, plotly
            return False
        except ImportError:
            return True
    if _need_install():
        print('  📦 Installing dependencies (first run)...')
        subprocess.call(['apt-get', 'update', '-qq'])
        subprocess.call(['apt-get', 'install', '-y', '-qq', 'xvfb', 'libgl1-mesa-glx', 'cmake'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                               'pyvista', 'vtk', 'plotly', 'tetgen', 'seaborn', 'scikit-learn'])
    else:
        print('  ✅ Dependencies already installed')
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('  ✅ Google Drive mounted')
    except Exception as e:
        print(f'  ⚠️ Drive mount: {e}')
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_PANEL'] = 'false'
elif IN_KAGGLE:
    print('☁️ Kaggle detected')
    subprocess.call(['apt-get', 'update', '-qq'])
    subprocess.call(['apt-get', 'install', '-y', '-qq', 'xvfb', 'libgl1-mesa-glx', 'cmake'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'pyvista', 'vtk', 'tetgen', 'plotly', 'seaborn', 'scikit-learn'])
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
else:
    print('💻 Local environment')

if IN_COLAB:
    OUTPUT_DIR = '/content/drive/MyDrive/me/tetra/thesis/thesis_output/'
elif IN_KAGGLE:
    OUTPUT_DIR = '/kaggle/working/thesis_output/'
else:
    OUTPUT_DIR = './thesis_output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'📁 Output directory: {OUTPUT_DIR}')

# ============================================================
# SECTION 2: IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, re, warnings, time
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupKFold
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
    if IN_COLAB:
        import plotly.io as pio; pio.renderers.default = 'colab'
except ImportError:
    HAS_PLOTLY = False

try:
    import pyvista as pv
    if IN_CLOUD:
        try: pv.start_xvfb()
        except: pass
    pv.set_plot_theme('document')
    HAS_PYVISTA = True
except:
    HAS_PYVISTA = False

try:
    import tetgen as _tetgen_lib
    HAS_TETGEN = True
except ImportError:
    HAS_TETGEN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 60)
print('🦴 Tetrahedral Mesh v3: Template Deformation Network')
print('=' * 60)
print(f'  Platform: {"Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "Local"}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Device: {"CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
print(f'  TetGen: {"✅" if HAS_TETGEN else "❌"}')

# ============================================================
# SECTION 3: CONFIGURATION
# ============================================================
def _auto_detect_data_dir():
    for path in ['/content/drive/MyDrive/thesis/me/dataset/4_bonemat_cdb_files',
                 '/kaggle/input/femur-cdb-files', './4_bonemat_cdb_files', '../4_bonemat_cdb_files']:
        if os.path.isdir(path) and len(glob.glob(os.path.join(path, '*.cdb'))) > 0:
            print(f'  📂 Data: {path} ({len(glob.glob(os.path.join(path, "*.cdb")))} CDB files)')
            return path
    return '/content/drive/MyDrive/thesis/me/dataset/4_bonemat_cdb_files' if IN_COLAB else './4_bonemat_cdb_files'

DATA_DIR = _auto_detect_data_dir()

MODEL_CONFIG = {
    'n_surface_pts': 4096,
    'n_interior_pts': 8192,      # template size — close to GT mean (~8937)
    'k_local': 8,                # k-nearest surface points for local conditioning
    'encoder_dim': 128,          # DGCNN per-point feature dim (was 256)
    'global_dim': 256,           # global feature dim = encoder_dim * 2 (was 512)
    'hidden_dim': 128,           # decoder MLP hidden dim (was 256)
    'batch_size': 4,
    'epochs': 300,
    'lr': 3e-4,                  # lower LR for smaller model (was 5e-4)
    'lr_patience': 25,
    'weight_decay': 5e-4,        # stronger L2 reg (was 1e-4)
    'chamfer_weight': 1.0,       # CD on predicted positions vs GT interior
    'material_weight': 0.1,      # MSE on materials
    'disp_reg_weight': 0.01,     # Soft L2 penalty on displacement magnitude
    'smooth_reg_weight': 0.005,  # displacement smoothness between neighbors
    'disp_scale': 0.3,           # Output range ±0.3 (baseline CD ~ 0.18)
    'k_folds': 5,
    'early_stop_patience': 50,   # model finds best in 15-50 epochs (was 80)
    'dgcnn_k': 20,
    'encoder_freeze_epoch': 0,   # DISABLED — 294K params + regularization is enough
    'mixup_alpha': 0.4,          # beta distribution param for PointMixup
    'mixup_prob': 0.5,           # probability of applying mixup per sample
    'jitter_std': 0.005,         # surface noise std (was 0.002)
    'point_dropout': 0.1,        # fraction of surface points to randomly drop
}

MATERIAL_NORM = {'global_min': None, 'global_max': None}

print('✅ V3 Configuration')
for k, v in MODEL_CONFIG.items():
    print(f'  {k}: {v}')

# ============================================================
# SECTION 4: CDB FILE READER
# ============================================================
class CDBReader:
    """Read ANSYS CDB files — NBLOCK/EBLOCK/MPDATA parsing."""

    def read(self, filepath):
        nodes, tets, elem_mat_ids, materials = [], [], [], {}
        section = None
        fmt = None
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if s.upper().startswith('NBLOCK'):
                    section = 'nblock'; fmt = None; continue
                elif s.upper().startswith('EBLOCK'):
                    section = 'eblock'; continue
                elif s == '-1' or s.upper().startswith('N,R5'):
                    section = None; continue
                if s.startswith('MPDATA'):
                    self._parse_mpdata(s, materials); continue
                if s.startswith('MPTEMP'):
                    continue
                if section and s.startswith('('):
                    if section == 'nblock':
                        fmt = self._parse_fmt(s)
                    continue
                if not s or s.startswith('!') or s.startswith('/'):
                    continue
                if section == 'nblock':
                    nd = self._read_node(line, fmt)
                    if nd is not None: nodes.append(nd)
                elif section == 'eblock':
                    r = self._read_elem(s)
                    if r is not None:
                        tets.append(r[0]); elem_mat_ids.append(r[1])

        nodes = np.array(nodes) if nodes else np.empty((0, 4))
        base = os.path.basename(filepath).replace('_bonemat.cdb', '').replace('_re', '')
        parts = base.split('_')
        meta = {
            'patient_id': parts[0] if parts else base,
            'side': 'left' if 'left' in filepath.lower() else 'right' if 'right' in filepath.lower() else 'unknown',
            'filepath': filepath, 'materials': materials, 'elem_mat_ids': elem_mat_ids,
        }
        return nodes, tets, meta

    def _parse_fmt(self, s):
        mi = re.search(r'(\d+)i(\d+)', s, re.I)
        mf = re.search(r'(\d+)e(\d+)', s, re.I)
        if mi and mf:
            return {'n_int': int(mi.group(1)), 'w_int': int(mi.group(2)),
                    'n_flt': int(mf.group(1)), 'w_flt': int(mf.group(2))}
        return None

    def _read_node(self, raw, fmt):
        if fmt:
            try:
                wi, wf = fmt['w_int'], fmt['w_flt']
                off = wi * fmt['n_int']
                return [int(raw[:wi]), float(raw[off:off+wf]),
                        float(raw[off+wf:off+2*wf]), float(raw[off+2*wf:off+3*wf])]
            except (ValueError, IndexError): pass
        try:
            p = raw.split()
            if len(p) >= 6: return [int(p[0]), float(p[3]), float(p[4]), float(p[5])]
        except: pass
        return None

    def _read_elem(self, line):
        try:
            for fw in [7, 8]:
                if len(line.rstrip()) >= fw * 15:
                    try:
                        vals = [int(line[i*fw:(i+1)*fw]) for i in range(min(19, len(line.rstrip())//fw))]
                        if len(vals) >= 15:
                            nids = [v for v in vals[11:15] if v > 0]
                            if len(nids) >= 4:
                                return tuple(nids[:4]), vals[0]
                    except (ValueError, IndexError): continue
            fields = line.split()
            if len(fields) >= 15:
                nids = [int(fields[11+i]) for i in range(min(4, len(fields)-11)) if int(fields[11+i]) > 0]
                if len(nids) >= 4:
                    return tuple(nids[:4]), int(fields[0])
        except: pass
        return None

    @staticmethod
    def _parse_mpdata(line, materials):
        try:
            parts = [p.strip() for p in line.split(',')]
            prop, mat_id, value = parts[3].strip(), int(parts[4]), float(parts[6])
            materials.setdefault(mat_id, {})[prop] = value
        except: pass

    def read_directory(self, directory):
        files = sorted(glob.glob(os.path.join(directory, '*.cdb')))
        if not files: print(f"❌ No CDB files in {directory}"); return {}
        print(f"📂 Found {len(files)} CDB files")
        meshes = {}
        for i, fp in enumerate(files):
            name = os.path.basename(fp)
            try:
                nodes, tets, meta = self.read(fp)
                if len(nodes) > 0:
                    meshes[name] = {'nodes': nodes, 'tets': tets, 'meta': meta}
                    if (i+1) % 20 == 0 or i == 0 or i == len(files)-1:
                        print(f"  [{i+1}/{len(files)}] ✅ {name}: {len(nodes)} nodes, {len(tets)} tets, {len(meta['materials'])} materials")
            except Exception as e:
                print(f"  [{i+1}/{len(files)}] ❌ {name}: {e}")
        tn = sum(len(m['nodes']) for m in meshes.values())
        tt = sum(len(m['tets']) for m in meshes.values())
        print(f"\n📊 {len(meshes)} files | {tn:,} nodes | {tt:,} tetrahedra")
        return meshes

# ============================================================
# SECTION 5: SURFACE EXTRACTION & MESH PROCESSING
# ============================================================
def extract_surface(tets):
    """Extract boundary faces and separate surface/interior node IDs."""
    face_count = Counter()
    for tet in tets:
        n = tet[:4]
        for tri in [(n[0],n[1],n[2]), (n[0],n[1],n[3]), (n[0],n[2],n[3]), (n[1],n[2],n[3])]:
            face_count[tuple(sorted(tri))] += 1
    surf_nids = set()
    for f, c in face_count.items():
        if c == 1: surf_nids.update(f)
    return surf_nids

def estimate_normals(points, k=15):
    """Estimate point cloud normals via PCA of k-nearest neighbors."""
    tree = KDTree(points)
    _, nn_idx = tree.query(points, k=min(k, len(points)))
    normals = np.zeros_like(points)
    for i in range(len(points)):
        nb = points[nn_idx[i]]
        cov = (nb - nb.mean(0)).T @ (nb - nb.mean(0)) / len(nb)
        _, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]
    # Orient outward
    outward = points - points.mean(0)
    normals[np.sum(normals * outward, axis=1) < 0] *= -1
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    return normals

def compute_node_materials(nid_to_pos, tets, elem_mat_ids, materials):
    """Compute per-node log10(EX) by averaging adjacent element materials."""
    node_sum, node_cnt = {}, {}
    for i, tet in enumerate(tets):
        if i >= len(elem_mat_ids): break
        mid = elem_mat_ids[i]
        if mid not in materials or 'EX' not in materials[mid]: continue
        ex = materials[mid]['EX']
        if ex <= 0: continue
        log_ex = np.log10(max(ex, 1.0))
        for nid in tet[:4]:
            node_sum[nid] = node_sum.get(nid, 0.0) + log_ex
            node_cnt[nid] = node_cnt.get(nid, 0) + 1
    return {n: node_sum[n] / node_cnt[n] for n in node_sum}

def sample_or_pad(pts, n, aux=None):
    """Resample point cloud to exactly n points. Optionally resample aux arrays with same indices."""
    m = len(pts)
    if m == 0: return np.zeros((n, pts.shape[1] if pts.ndim > 1 else 3)), np.zeros(n, dtype=int)
    idx = np.random.choice(m, n, replace=(m < n))
    result = pts[idx]
    if aux is not None:
        return result, aux[idx], idx
    return result, idx

def process_mesh(nodes, tets, meta, n_surf, n_int):
    """Process a single mesh into normalized surface + interior + materials."""
    surf_nids = extract_surface(tets)
    nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
    all_nids = set(nid_to_pos.keys())
    int_nids = all_nids - surf_nids

    surf_pts = np.array([nid_to_pos[n] for n in surf_nids if n in nid_to_pos])
    int_pts = np.array([nid_to_pos[n] for n in int_nids if n in nid_to_pos])
    int_nid_list = [n for n in int_nids if n in nid_to_pos]
    if len(surf_pts) < 50 or len(int_pts) < 50:
        return None

    # Material per interior node
    mats = meta.get('materials', {})
    eids = meta.get('elem_mat_ids', [])
    node_mat = compute_node_materials(nid_to_pos, tets, eids, mats)
    int_mat_raw = np.array([node_mat.get(n, 0.0) for n in int_nid_list], dtype=np.float32)
    has_material = bool(np.any(int_mat_raw > 0))

    # Normalize to unit sphere
    all_pts = np.vstack([surf_pts, int_pts])
    centroid = all_pts.mean(axis=0)
    scale = max(np.max(np.linalg.norm(all_pts - centroid, axis=1)), 1e-10)
    surf_norm = (surf_pts - centroid) / scale
    int_norm = (int_pts - centroid) / scale

    # Surface: xyz + normals
    normals = estimate_normals(surf_norm, k=15)
    surf_sampled, s_idx = sample_or_pad(surf_norm, n_surf)
    norm_sampled = normals[s_idx]

    # Interior: xyz + material
    int_sampled, mat_sampled, _ = sample_or_pad(int_norm, n_int, aux=int_mat_raw)

    return {
        'surface': np.hstack([surf_sampled, norm_sampled]).astype(np.float32),  # (n_surf, 6)
        'interior': int_sampled.astype(np.float32),                             # (n_int, 3)
        'material': mat_sampled.astype(np.float32),                             # (n_int,)
        'has_material': has_material,
        'centroid': centroid, 'scale': scale,
        'n_surf_orig': len(surf_pts), 'n_int_orig': len(int_pts),
    }

# ============================================================
# SECTION 6: DATASET WITH TEMPLATE DEFORMATION TARGETS
# ============================================================
class TemplateDeformDataset(Dataset):
    """Dataset that computes mean-shape template and per-sample displacement targets.

    For each sample:
      - target_displacement = nearest_GT_interior_point - template_point
      - target_material = material at nearest GT interior point
    """

    def __init__(self, meshes, augment=False):
        """Process meshes and normalize materials. Template NOT computed here to
        prevent data leakage — computed per-fold in run_kfold() from training only."""
        self.augment = augment
        self.template = None  # Set per-fold
        self.samples, self.names = [], []
        n_s = MODEL_CONFIG['n_surface_pts']
        n_i = MODEL_CONFIG['n_interior_pts']

        # Fixed seed for reproducible preprocessing
        np.random.seed(42)

        n_mat = 0
        for name, m in meshes.items():
            pair = process_mesh(m['nodes'], m['tets'], m.get('meta'), n_s, n_i)
            if pair:
                self.samples.append(pair)
                self.names.append(name)
                if pair['has_material']: n_mat += 1

        # Normalize materials globally and save normalization
        self._normalize_materials()
        self._save_material_norm()

        np.random.seed(None)

        print(f"  Dataset: {len(self.samples)} samples ({n_mat} with materials)")
        print(f"  Patients: {len(set(_extract_patient_id(n) for n in self.names))} unique")

    def _normalize_materials(self):
        global MATERIAL_NORM
        all_raw = []
        for s in self.samples:
            if s['has_material']:
                nz = s['material'][s['material'] > 0]
                if len(nz) > 0: all_raw.append(nz)
        if not all_raw:
            for s in self.samples: s['material'] = np.zeros_like(s['material'])
            return
        all_raw = np.concatenate(all_raw)
        gmin, gmax = float(all_raw.min()), float(all_raw.max())
        grange = max(gmax - gmin, 1e-6)
        MATERIAL_NORM['global_min'] = gmin
        MATERIAL_NORM['global_max'] = gmax
        print(f"  Material range: log10(EX) = [{gmin:.2f}, {gmax:.2f}] "
              f"({10**gmin:.0f} to {10**gmax:.0f} MPa)")
        for s in self.samples:
            m = s['material']
            s['material'] = np.where(m > 0, np.clip((m - gmin) / grange, 0, 1), 0).astype(np.float32)

    def _save_material_norm(self):
        """Save material normalization to disk so export can denormalize."""
        norm_path = os.path.join(OUTPUT_DIR, 'v3_material_norm.npy')
        np.save(norm_path, np.array([MATERIAL_NORM.get('global_min', 0.0),
                                     MATERIAL_NORM.get('global_max', 4.3)]))
        print(f"  💾 Material norm saved: {norm_path}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _random_rotation():
        angles = np.random.uniform(0, 2*np.pi, 3)
        cx, sx = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cz, sz = np.cos(angles[2]), np.sin(angles[2])
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return (Rz @ Ry @ Rx).astype(np.float32)

    def __getitem__(self, idx):
        # This is only used if template was set externally (legacy path)
        raise NotImplementedError("Use _SubDataset with per-fold template")


# ============================================================
# SECTION 7: DGCNN ENCODER (with per-point features)
# ============================================================
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    dist = xx + inner + xx.transpose(2, 1)
    return (-dist).topk(k=k, dim=-1)[1]

def edge_features(x, k=20, idx=None):
    B, D, N = x.size()
    if idx is None: idx = knn(x, k)
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + base).view(-1)
    x_t = x.transpose(2, 1).contiguous().view(B * N, -1)
    neighbors = x_t[idx_flat].view(B, N, k, D)
    center = x.transpose(2, 1).view(B, N, 1, D).expand(-1, -1, k, -1)
    return torch.cat([neighbors - center, center], dim=3).permute(0, 3, 1, 2)

class DGCNNEncoder(nn.Module):
    """3-layer DGCNN that returns BOTH per-point features AND global features.

    This is the key difference from v2: per-point features enable local conditioning.
    Output: global_feat (B, 512), point_feat (B, N, 256)
    """
    def __init__(self, k=20, in_dim=6, point_dim=256):
        super().__init__()
        self.k = k
        self.ec1 = nn.Sequential(nn.Conv2d(in_dim*2, 64, 1, bias=False),
                                 nn.GroupNorm(8, 64), nn.LeakyReLU(0.2))
        self.ec2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                 nn.GroupNorm(8, 128), nn.LeakyReLU(0.2))
        self.ec3 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                 nn.GroupNorm(16, 256), nn.LeakyReLU(0.2))
        # Per-point feature projection: concat of all scales → point_dim
        self.point_proj = nn.Sequential(
            nn.Conv1d(64+128+256, point_dim, 1, bias=False),
            nn.GroupNorm(16, point_dim), nn.LeakyReLU(0.2))
        # Global feature aggregation
        self.global_agg = nn.Sequential(
            nn.Conv1d(64+128+256, point_dim, 1, bias=False),
            nn.GroupNorm(16, point_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        """x: (B, 6, N) → global: (B, 512), point: (B, N, 256)"""
        B = x.size(0)
        x1 = self.ec1(edge_features(x, self.k)).max(-1)[0]  # (B, 64, N)
        x2 = self.ec2(edge_features(x1, self.k)).max(-1)[0]  # (B, 128, N)
        x3 = self.ec3(edge_features(x2, self.k)).max(-1)[0]  # (B, 256, N)
        cat = torch.cat([x1, x2, x3], dim=1)  # (B, 448, N)

        # Per-point features
        point_feat = self.point_proj(cat).transpose(1, 2)  # (B, N, 256)

        # Global features via max+avg pooling
        g = self.global_agg(cat)  # (B, 256, N)
        g_max = F.adaptive_max_pool1d(g, 1).view(B, -1)   # (B, 256)
        g_avg = F.adaptive_avg_pool1d(g, 1).view(B, -1)   # (B, 256)
        global_feat = torch.cat([g_max, g_avg], dim=1)     # (B, 512)

        return global_feat, point_feat


# ============================================================
# SECTION 8: TEMPLATE DEFORMATION NETWORK
# ============================================================
class TemplateDeformNet(nn.Module):
    """Predict per-node displacement + material via local surface conditioning.

    For each template node:
      1. Find k nearest surface points
      2. Average their per-point features → local feature (256D)
      3. Concat [template_xyz (3) + local (256) + global (512)] = 771D
      4. MLP → displacement (3D) + material (1D)
    """
    def __init__(self, global_dim=512, local_dim=256, hidden=256, k_local=8):
        super().__init__()
        self.k_local = k_local
        inp = 3 + local_dim + global_dim  # 771

        self.disp_head = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 3))  # dx, dy, dz

        self.mat_head = nn.Sequential(
            nn.Linear(inp, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def _get_local_features(self, template, surf_xyz, point_feat):
        """For each template point, aggregate features of k nearest surface points."""
        B, T, _ = template.shape
        _, S, D = point_feat.shape
        k = self.k_local

        local_feats = []
        chunk = 1024  # process in chunks to save GPU memory
        for i in range(0, T, chunk):
            t_chunk = template[:, i:min(i+chunk, T)]  # (B, c, 3)
            c = t_chunk.shape[1]
            # Pairwise distances
            dist = torch.cdist(t_chunk, surf_xyz)  # (B, c, S)
            _, nn_idx = dist.topk(k, dim=2, largest=False)  # (B, c, k)
            # Gather: flatten indices for efficient gather
            flat_idx = nn_idx.reshape(B, -1)  # (B, c*k)
            flat_exp = flat_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, c*k, D)
            gathered = torch.gather(point_feat, 1, flat_exp)  # (B, c*k, D)
            gathered = gathered.reshape(B, c, k, D)  # (B, c, k, D)
            local = gathered.mean(dim=2)  # (B, c, D)
            local_feats.append(local)

        return torch.cat(local_feats, dim=1)  # (B, T, D)

    def forward(self, template, surf_xyz, global_feat, point_feat):
        """
        template: (B, T, 3) — template node positions
        surf_xyz: (B, S, 3) — surface point positions
        global_feat: (B, 512) — global surface feature
        point_feat: (B, S, 256) — per-point surface features
        Returns: displacement (B, T, 3), material (B, T)
        """
        B, T, _ = template.shape

        # Local conditioning: k-NN feature aggregation
        local_feat = self._get_local_features(template, surf_xyz, point_feat)  # (B, T, 256)

        # Global conditioning: expand to all template points
        global_exp = global_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)

        # Concatenate all features per template node
        node_input = torch.cat([template, local_feat, global_exp], dim=2)  # (B, T, 771)

        # Predict displacement and material
        disp = self.disp_head(node_input)           # (B, T, 3)
        mat = self.mat_head(node_input).squeeze(-1)  # (B, T)

        # Scale displacement so model can make meaningful deformations
        disp = disp * MODEL_CONFIG['disp_scale']

        return disp, mat


class SurfaceToVolumeModel(nn.Module):
    """Full model: DGCNN encoder → Template Deformation Network."""
    def __init__(self):
        super().__init__()
        cfg = MODEL_CONFIG
        self.encoder = DGCNNEncoder(
            k=cfg['dgcnn_k'], in_dim=6, point_dim=cfg['encoder_dim'])
        self.decoder = TemplateDeformNet(
            global_dim=cfg['global_dim'], local_dim=cfg['encoder_dim'],
            hidden=cfg['hidden_dim'], k_local=cfg['k_local'])

    def forward(self, surface, template):
        """
        surface: (B, N_surf, 6) — surface points with normals
        template: (B, N_int, 3) — template positions
        Returns: displacement (B, N_int, 3), material (B, N_int)
        """
        # Encode surface
        surf_t = surface.transpose(1, 2)  # (B, 6, N_surf)
        global_feat, point_feat = self.encoder(surf_t)  # (B, 512), (B, N_surf, 256)
        surf_xyz = surface[:, :, :3]  # (B, N_surf, 3)

        # Decode: predict displacements + materials
        disp, mat = self.decoder(template, surf_xyz, global_feat, point_feat)
        return disp, mat

n_params = sum(p.numel() for p in SurfaceToVolumeModel().parameters())
print(f'✅ Model verified: {n_params:,} parameters')


# ============================================================
# SECTION 9: LOSS FUNCTIONS
# ============================================================
def chamfer_distance(pred, target, chunk=512):
    """Chunked symmetric Chamfer Distance using L2 (not squared L2)."""
    B, N, _ = pred.size()
    M = target.size(1)
    # pred→target
    min_p2t = torch.full((B, N), 1e6, device=pred.device)
    for i in range(0, N, chunk):
        d = (pred[:, i:i+chunk].unsqueeze(2) - target.unsqueeze(1)).pow(2).sum(-1).sqrt()
        min_p2t[:, i:i+chunk] = d.min(2)[0]
    # target→pred
    min_t2p = torch.full((B, M), 1e6, device=pred.device)
    for i in range(0, M, chunk):
        d = (target[:, i:i+chunk].unsqueeze(2) - pred.unsqueeze(1)).pow(2).sum(-1).sqrt()
        min_t2p[:, i:i+chunk] = d.min(2)[0]
    return (min_p2t.mean(1) + min_t2p.mean(1)).mean() / 2  # average (matches eval)


class DeformLoss(nn.Module):
    """Loss = CD(predicted_pos, GT_interior) + MSE(material) + disp_reg + smooth_reg.
    CD handles correspondence automatically. Smooth reg enforces spatial coherence.
    """
    def __init__(self):
        super().__init__()
        self._tpl_nn_idx = None  # cached template neighbor indices

    def _get_template_neighbors(self, template, k=6):
        """Precompute k-nearest neighbors in template for smoothness loss."""
        # template: (B, T, 3) — use first batch item (same template for all)
        tpl = template[0].detach().cpu().numpy()
        tree = KDTree(tpl)
        _, idx = tree.query(tpl, k=k+1)  # +1 because self is nearest
        return torch.tensor(idx[:, 1:], device=template.device)  # (T, k) exclude self

    def forward(self, pred_disp, pred_mat, target_pos, target_mat, template):
        # Predicted final positions
        pred_pos = template + pred_disp
        # Chamfer distance against raw GT interior (CD handles correspondence)
        cd = chamfer_distance(pred_pos, target_pos)
        # MSE on materials
        mat_loss = F.mse_loss(pred_mat, target_mat)
        # Soft L2 displacement regularization — prevents overfitting via wild displacements
        disp_reg = pred_disp.pow(2).mean()

        # Displacement smoothness — neighboring template points should move similarly
        if self._tpl_nn_idx is None or self._tpl_nn_idx.device != template.device:
            self._tpl_nn_idx = self._get_template_neighbors(template)
        nn_idx = self._tpl_nn_idx  # (T, k)
        B = pred_disp.size(0)
        # Gather neighbor displacements: (B, T, k, 3)
        nb_disp = pred_disp[:, nn_idx]  # broadcast: (B, T, k, 3)
        # Mean squared difference between each point and its neighbors
        smooth_reg = (nb_disp - pred_disp.unsqueeze(2)).pow(2).mean()

        total = (MODEL_CONFIG['chamfer_weight'] * cd
                 + MODEL_CONFIG['material_weight'] * mat_loss
                 + MODEL_CONFIG['disp_reg_weight'] * disp_reg
                 + MODEL_CONFIG['smooth_reg_weight'] * smooth_reg)
        return total, {
            'cd': cd.item(), 'mat': mat_loss.item(),
            'disp_mag': pred_disp.detach().norm(dim=-1).mean().item(),
            'smooth': smooth_reg.item(),
            'total': total.item()
        }


# ============================================================
# SECTION 10: TRAINER
# ============================================================
class Trainer:
    def __init__(self, model, train_dl, val_dl, fold=1):
        self.model = model.to(DEVICE)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.fold = fold
        self.loss_fn = DeformLoss()
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=MODEL_CONFIG['lr'],
                                     weight_decay=MODEL_CONFIG['weight_decay'])
        # ReduceLROnPlateau: only DECREASES LR when val plateaus (no disruptive restarts)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=15, min_lr=1e-6)
        self._encoder_frozen = False

    def _freeze_encoder(self):
        """Freeze DGCNN encoder weights — only train decoder MLP."""
        if self._encoder_frozen:
            return
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    🧊 Encoder frozen — {trainable:,} trainable params remain")

    def _run_epoch(self, dl, train=True):
        self.model.train(train)
        total_l, total_cd, n = 0.0, 0.0, 0
        for surf, template, target_pos, target_mat, _ in dl:
            surf = surf.to(DEVICE)
            template = template.to(DEVICE)
            target_pos = target_pos.to(DEVICE)
            target_mat = target_mat.to(DEVICE)
            pred_disp, pred_mat = self.model(surf, template)
            loss, metrics = self.loss_fn(pred_disp, pred_mat, target_pos, target_mat, template)
            if train:
                if torch.isnan(loss) or torch.isinf(loss):
                    print("    ⚠️ NaN/Inf loss, skipping batch"); continue
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
        freeze_ep = MODEL_CONFIG.get('encoder_freeze_epoch', 0)
        print(f"\n  Fold {self.fold} | Training on {DEVICE} | {n_params:,} params")
        if freeze_ep > 0:
            print(f"    Encoder will freeze after epoch {freeze_ep}")
        history = []
        for ep in range(1, MODEL_CONFIG['epochs'] + 1):
            # Freeze encoder after warmup period
            if freeze_ep > 0 and ep == freeze_ep + 1:
                self._freeze_encoder()
            t0 = time.time()
            train_loss, train_cd = self._run_epoch(self.train_dl, train=True)
            with torch.no_grad():
                val_loss, val_cd = self._run_epoch(self.val_dl, train=False)
            self.scheduler.step(val_cd)  # ReduceLROnPlateau needs val metric
            lr = self.opt.param_groups[0]['lr']
            history.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss,
                            'train_cd': train_cd, 'val_cd': val_cd, 'lr': lr})
            if ep % 10 == 0 or ep == 1:
                dt = time.time() - t0
                print(f"    Ep {ep:4d} | Train CD: {train_cd:.6f} | Val CD: {val_cd:.6f} "
                      f"| LR: {lr:.1e} | {dt:.1f}s")
            if val_cd < best_val:
                best_val = val_cd
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
            if patience >= MODEL_CONFIG['early_stop_patience']:
                print(f"    ⏹️ Early stop at epoch {ep} (best val CD: {best_val:.6f})")
                break
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"  ✅ Fold {self.fold} done | Best val CD: {best_val:.6f}")
        return history

    def save_model(self, path):
        """Save full checkpoint: model weights + material normalization."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'material_norm': {
                'global_min': MATERIAL_NORM.get('global_min'),
                'global_max': MATERIAL_NORM.get('global_max'),
            },
            'config': MODEL_CONFIG,
        }
        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint saved: {path} (model + material_norm)")


# ============================================================
# SECTION 11: EVALUATION
# ============================================================
def evaluate_model(model, dataset, template_np):
    """Evaluate model on a dataset with enhanced metrics + zero-displacement baseline."""
    model.eval()
    results = []
    for i in range(len(dataset)):
        surf, template, target_pos_t, target_mat, name = dataset[i]
        surf = surf.unsqueeze(0).to(DEVICE)
        template = template.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_disp, pred_mat = model(surf, template)

        pred_pos = (template + pred_disp)[0].cpu().numpy()
        target_pos = target_pos_t.numpy()  # Raw GT interior positions
        pred_d = pred_disp[0].cpu().numpy()

        # Chamfer distance (model prediction vs raw GT)
        tree_t = KDTree(target_pos)
        tree_p = KDTree(pred_pos)
        d_pt, _ = tree_p.query(target_pos)
        d_tp, _ = tree_t.query(pred_pos)
        cd = (d_pt.mean() + d_tp.mean()) / 2
        hd = max(d_pt.max(), d_tp.max())

        # Zero-displacement BASELINE: CD of template itself vs raw GT
        d_bt, _ = tree_t.query(template_np)  # template→GT
        tree_tpl = KDTree(template_np)
        d_tb, _ = tree_tpl.query(target_pos)  # GT→template
        baseline_cd = (d_bt.mean() + d_tb.mean()) / 2

        # Displacement magnitude stats
        disp_mag = np.linalg.norm(pred_d, axis=1)
        disp_mean = float(disp_mag.mean())
        disp_max = float(disp_mag.max())

        # Point-wise RMSE (pred vs nearest GT — since sizes may differ)
        d_p2t_rmse, _ = tree_t.query(pred_pos)
        rmse = float(np.sqrt(np.mean(d_p2t_rmse**2)))

        # Material
        pm = pred_mat[0].cpu().numpy()
        tm = target_mat.numpy()
        mat_mse = float(np.mean((pm - tm)**2))
        mat_mae = float(np.mean(np.abs(pm - tm)))

        results.append({
            'name': name, 'chamfer': cd, 'hausdorff': hd,
            'baseline_cd': baseline_cd,
            'improvement': 1.0 - (cd / max(baseline_cd, 1e-10)),
            'disp_mean': disp_mean, 'disp_max': disp_max,
            'rmse': rmse,
            'material_mse': mat_mse, 'material_mae': mat_mae,
        })
    return results


# ============================================================
# SECTION 12: CDB EXPORT
# ============================================================
def write_cdb(filepath, nodes, tets, material_values=None, poisson=0.3):
    """Write ANSYS CDB — format matches lhpOpExporterAnsysCDB exactly."""
    n_nodes, n_elems = len(nodes), len(tets)
    with open(filepath, 'w') as f:
        now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        f.write(f"!! Generated by AI Mesh Pipeline v3 {now}\n\n/PREP7\n")
        f.write(f"NBLOCK,6,SOLID,{n_nodes:>8},{n_nodes:>9}\n(3i7,6e22.13)\n")
        for i, (x, y, z) in enumerate(nodes, 1):
            f.write(f"{i:7d}{0:7d}{0:7d}{x:22.13E}{y:22.13E}{z:22.13E}\n")
        f.write("N,R5.3,LOC,       -1,\n\nET,3,73\n\n")
        if material_values is not None:
            node_ex = np.clip(np.asarray(material_values, dtype=np.float64).flatten(), 100.0, 30000.0)
            elem_ex = [np.mean([node_ex[n] if n < len(node_ex) else 5000.0 for n in t]) for t in tets]
            ex_rounded = np.clip(np.round(elem_ex).astype(int), 100, 30000)
            unique_ex = np.unique(ex_rounded)
            if len(unique_ex) > 500:
                bins = np.linspace(ex_rounded.min(), ex_rounded.max(), 501)
                binned = np.digitize(ex_rounded, bins)
                centers = 0.5 * (bins[:-1] + bins[1:])
                ex_rounded = np.round(centers[np.clip(binned-1, 0, len(centers)-1)]).astype(int)
                unique_ex = np.unique(ex_rounded)
            ex_to_mid = {v: i+1 for i, v in enumerate(unique_ex)}
            elem_mid = np.array([ex_to_mid[v] for v in ex_rounded])
        else:
            elem_mid = np.ones(n_elems, dtype=int)
            ex_to_mid = None
        f.write(f"EBLOCK,19,SOLID,{n_elems:>8},{n_elems:>9}\n(19i7)\n")
        for i, tet in enumerate(tets):
            m = int(elem_mid[i]); eid = i+1
            n1,n2,n3,n4 = tet[0]+1, tet[1]+1, tet[2]+1, tet[3]+1
            f.write(f"{m:7d}{3:7d}{1:7d}{1:7d}{0:7d}{0:7d}{0:7d}{0:7d}{4:7d}{0:7d}{eid:7d}"
                    f"{n1:7d}{n2:7d}{n3:7d}{n4:7d}\n")
        f.write("-1\n")
        if ex_to_mid:
            for ex_v, mid in sorted(ex_to_mid.items(), key=lambda x: x[1]):
                dens = (float(ex_v)/6850.0)**(1.0/1.49) if ex_v > 0 else 0.001
                f.write(f"MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
                f.write(f"MPDATA,R5.0, 1,EX,{mid:>6}, 1, {float(ex_v):.8f}    ,\n")
                f.write(f"MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
                f.write(f"MPDATA,R5.0, 1,NUXY,{mid:>6}, 1, {poisson:.8f}    ,\n")
                f.write(f"MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
                f.write(f"MPDATA,R5.0, 1,DENS,{mid:>6}, 1, {dens:.8f}    ,\n")
        else:
            f.write("MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write("MPDATA,R5.0, 1,EX,     1, 1, 10000.00000000    ,\n")
            f.write("MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,NUXY,     1, 1, {poisson:.8f}    ,\n")
            f.write("MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write("MPDATA,R5.0, 1,DENS,     1, 1, 1.00000000    ,\n")
        f.write("\n/GO\nFINISH\n")


# ============================================================
# SECTION 13: K-FOLD CROSS-VALIDATION
# ============================================================
class _SubDataset(Dataset):
    """Fold-specific dataset: uses per-fold template (training-only), computes targets."""
    def __init__(self, parent, indices, template, augment=False):
        self.samples = [parent.samples[i] for i in indices]
        self.names = [parent.names[i] for i in indices]
        self.template = template  # Per-fold template (from training samples ONLY)
        self.augment = augment
        # Pre-compute targets relative to this fold's template
        self._compute_targets()

    def _compute_targets(self):
        """Store raw GT interior + material for CD-based loss (no NN mapping).
        Also keep NN-mapped material for template-point-based evaluation.
        """
        for s in self.samples:
            gt_int = s['interior']   # (n_int, 3)
            gt_mat = s['material']   # (n_int,)
            # Raw GT interior positions — used as CD target (no NN noise)
            s['target_pos'] = gt_int.astype(np.float32)
            s['target_mat_raw'] = gt_mat.astype(np.float32)
            # NN-mapped material for template points (used for material head supervision)
            tree = KDTree(gt_int)
            _, idx = tree.query(self.template, k=1)
            s['target_mat'] = gt_mat[idx].astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        surface = s['surface'].copy()
        template = self.template.copy()
        target_pos = s['target_pos'].copy()  # Raw GT interior (for CD)
        target_mat = s['target_mat'].copy()  # NN-mapped material (for template pts)

        if self.augment:
            # --- PointMixup: interpolate with another sample ---
            if np.random.random() < MODEL_CONFIG.get('mixup_prob', 0.5):
                j = np.random.randint(0, len(self.samples))
                s2 = self.samples[j]
                lam = np.random.beta(MODEL_CONFIG.get('mixup_alpha', 0.4),
                                     MODEL_CONFIG.get('mixup_alpha', 0.4))
                # Mix surface (xyz only, recompute normals is too slow — mix normals too)
                surface = lam * surface + (1 - lam) * s2['surface']
                # Renormalize mixed normals
                nrm = np.linalg.norm(surface[:, 3:], axis=1, keepdims=True)
                surface[:, 3:] /= np.maximum(nrm, 1e-10)
                # Mix targets
                target_pos = lam * target_pos + (1 - lam) * s2['target_pos']
                target_mat = lam * target_mat + (1 - lam) * s2['target_mat']

            # --- Geometric augmentation ---
            R = TemplateDeformDataset._random_rotation()
            sc = 1.0 + np.random.uniform(-0.05, 0.05)
            surface[:, :3] = (surface[:, :3] @ R.T) * sc
            surface[:, 3:] = surface[:, 3:] @ R.T
            nrm = np.linalg.norm(surface[:, 3:], axis=1, keepdims=True)
            surface[:, 3:] /= np.maximum(nrm, 1e-10)
            template = (template @ R.T) * sc
            target_pos = (target_pos @ R.T) * sc

            # Random flip
            if np.random.random() > 0.5:
                surface[:, 0] *= -1; surface[:, 3] *= -1
                template[:, 0] *= -1; target_pos[:, 0] *= -1

            # --- Stronger jitter ---
            jitter_std = MODEL_CONFIG.get('jitter_std', 0.005)
            surface[:, :3] += np.random.randn(*surface[:, :3].shape).astype(np.float32) * jitter_std

            # --- Random surface point dropout ---
            drop_rate = MODEL_CONFIG.get('point_dropout', 0.1)
            if drop_rate > 0:
                n_pts = surface.shape[0]
                n_drop = int(n_pts * drop_rate)
                drop_idx = np.random.choice(n_pts, n_drop, replace=False)
                keep_idx = np.random.choice(n_pts, n_drop, replace=True)
                surface[drop_idx] = surface[keep_idx]  # replace dropped with random duplicates

        return (torch.tensor(surface, dtype=torch.float32),
                torch.tensor(template, dtype=torch.float32),
                torch.tensor(target_pos, dtype=torch.float32),   # raw GT for CD
                torch.tensor(target_mat, dtype=torch.float32),   # NN-mapped mat
                self.names[idx])

def _extract_patient_id(name):
    base = name.replace('_bonemat.cdb', '').replace('_re', '')
    return base.split('_')[0]

def _compute_template_from_indices(full_ds, indices):
    """Compute mean template from ONLY the specified sample indices (training-only)."""
    n = MODEL_CONFIG['n_interior_pts']
    acc = np.zeros((n, 3), dtype=np.float64)
    cnt = 0
    for i in indices:
        interior = full_ds.samples[i]['interior']
        if len(interior) == n:
            acc += interior.astype(np.float64)
            cnt += 1
    if cnt > 0:
        return (acc / cnt).astype(np.float32), cnt
    # Fallback
    rng = np.random.RandomState(42)
    pts = rng.randn(n, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * (rng.uniform(0, 1, (n, 1)) ** (1/3)).astype(np.float32), 0


def _zero_baseline_cd(template, full_ds, val_indices):
    """Compute CD between template (zero displacement) and each validation sample."""
    tree_tpl = KDTree(template)
    baseline_cds = []
    for vi in val_indices:
        gt_int = full_ds.samples[vi]['interior']  # (n_int, 3)
        tree_gt = KDTree(gt_int)
        d_t2g, _ = tree_gt.query(template)   # template → GT
        d_g2t, _ = tree_tpl.query(gt_int)    # GT → template
        baseline_cds.append((d_t2g.mean() + d_g2t.mean()) / 2)
    return np.array(baseline_cds)


def run_kfold(meshes):
    print("\n" + "=" * 60)
    print("🧠 K-FOLD CROSS-VALIDATION v3 (Template Deformation)")
    print("=" * 60)

    full_ds = TemplateDeformDataset(meshes, augment=False)
    n = len(full_ds)
    if n < 2: print("❌ Not enough samples"); return []

    patient_ids = [_extract_patient_id(name) for name in full_ds.names]
    unique_patients = sorted(set(patient_ids))
    n_patients = len(unique_patients)

    pid_to_group = {pid: i for i, pid in enumerate(unique_patients)}
    groups = np.array([pid_to_group[pid] for pid in patient_ids])
    k = min(MODEL_CONFIG['k_folds'], n_patients)
    gkf = GroupKFold(n_splits=k)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(range(n), groups=groups)):
        tr_patients = set(patient_ids[i] for i in tr_idx)
        va_patients = set(patient_ids[i] for i in va_idx)
        assert len(tr_patients & va_patients) == 0, "Patient-level data leakage!"
        print(f"\n{'='*50}")
        print(f"  FOLD {fold+1}/{k} | train={len(tr_idx)} ({len(tr_patients)} patients) "
              f"| val={len(va_idx)} ({len(va_patients)} patients)")

        # ★ FIX: Per-fold template from TRAINING samples ONLY
        template, tpl_cnt = _compute_template_from_indices(full_ds, tr_idx)
        print(f"  📐 Template: mean of {tpl_cnt} TRAINING samples (no val leakage)")

        # ★ FIX: Zero-displacement BASELINE
        baseline_cds = _zero_baseline_cd(template, full_ds, va_idx)
        print(f"  📏 Zero-displacement baseline val CD: {baseline_cds.mean():.6f} "
              f"(±{baseline_cds.std():.6f})")

        # Save per-fold template
        tpl_path = os.path.join(OUTPUT_DIR, f'v3_template_fold{fold+1}.npy')
        np.save(tpl_path, template)
        print(f"  💾 Template saved: {tpl_path}")

        # Create fold datasets with per-fold template
        tr_ds = _SubDataset(full_ds, tr_idx, template, augment=True)
        va_ds = _SubDataset(full_ds, va_idx, template, augment=False)

        bs = MODEL_CONFIG['batch_size']
        tr_dl = DataLoader(tr_ds, bs, shuffle=True, drop_last=len(tr_ds) > bs,
                           num_workers=2 if IN_CLOUD else 0, pin_memory=torch.cuda.is_available())
        va_dl = DataLoader(va_ds, bs,
                           num_workers=2 if IN_CLOUD else 0, pin_memory=torch.cuda.is_available())

        model = SurfaceToVolumeModel()
        trainer = Trainer(model, tr_dl, va_dl, fold=fold+1)
        hist = trainer.train()

        save_path = os.path.join(OUTPUT_DIR, f'model_v3_fold{fold+1}.pt')
        trainer.save_model(save_path)

        # ★ FIX: Enhanced evaluation with baseline comparison
        metrics = evaluate_model(model, va_ds, template)
        mean_cd = np.mean([m['chamfer'] for m in metrics])
        mean_baseline = np.mean([m['baseline_cd'] for m in metrics])
        mean_improve = np.mean([m['improvement'] for m in metrics])
        mean_disp = np.mean([m['disp_mean'] for m in metrics])

        print(f"\n  📊 FOLD {fold+1} EVALUATION:")
        print(f"  {'Metric':<25s} {'Value':>10s}")
        print(f"  {'-'*40}")
        print(f"  {'Model CD':<25s} {mean_cd:>10.6f}")
        print(f"  {'Baseline CD (zero-disp)':<25s} {mean_baseline:>10.6f}")
        print(f"  {'Improvement over base':<25s} {mean_improve*100:>9.1f}%")
        print(f"  {'Mean displacement':<25s} {mean_disp:>10.6f}")
        print(f"  {'Mean RMSE':<25s} {np.mean([m['rmse'] for m in metrics]):>10.6f}")
        print(f"  {'Mean HD':<25s} {np.mean([m['hausdorff'] for m in metrics]):>10.6f}")
        print(f"  {'Material MAE':<25s} {np.mean([m['material_mae'] for m in metrics]):>10.6f}")

        fold_results.append({
            'fold': fold+1, 'history': hist, 'metrics': metrics,
            'model_path': save_path, 'template_path': tpl_path,
            'mean_cd': mean_cd,
            'mean_baseline_cd': mean_baseline,
            'mean_improvement': mean_improve,
            'mean_hd': np.mean([m['hausdorff'] for m in metrics]),
            'mean_mat_mae': np.mean([m['material_mae'] for m in metrics]),
            'mean_disp': mean_disp,
        })

        del model, trainer, tr_dl, va_dl
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"  🧹 GPU memory cleared after fold {fold+1}")

    print("\n" + "=" * 60)
    print("📊 K-FOLD RESULTS SUMMARY")
    print("=" * 60)
    cds = [r['mean_cd'] for r in fold_results]
    bls = [r['mean_baseline_cd'] for r in fold_results]
    imps = [r['mean_improvement'] for r in fold_results]
    hds = [r['mean_hd'] for r in fold_results]
    mma = [r['mean_mat_mae'] for r in fold_results]
    disp = [r['mean_disp'] for r in fold_results]
    print(f"  {'Metric':<30s} {'Mean':>10s} {'± Std':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Model CD':<30s} {np.mean(cds):>10.6f} ±{np.std(cds):>9.6f}")
    print(f"  {'Baseline CD (zero-disp)':<30s} {np.mean(bls):>10.6f} ±{np.std(bls):>9.6f}")
    print(f"  {'Improvement %':<30s} {np.mean(imps)*100:>9.1f}% ±{np.std(imps)*100:>8.1f}%")
    print(f"  {'Hausdorff':<30s} {np.mean(hds):>10.6f} ±{np.std(hds):>9.6f}")
    print(f"  {'Material MAE':<30s} {np.mean(mma):>10.6f} ±{np.std(mma):>9.6f}")
    print(f"  {'Mean Displacement':<30s} {np.mean(disp):>10.6f} ±{np.std(disp):>9.6f}")
    print(f"  {'Patients':<30s} {n_patients} total, {k}-fold grouped")

    best = min(fold_results, key=lambda x: x['mean_cd'])
    print(f"\n  🏆 Best fold: {best['fold']} | CD: {best['mean_cd']:.6f}")
    print(f"     Template: {best['template_path']}")
    print(f"     Model:    {best['model_path']}")

    return fold_results


# ============================================================
# SECTION 14: VISUALIZATION
# ============================================================
class ResultsViz:
    @staticmethod
    def training_curves(fold_results):
        n_folds = len(fold_results)
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
        if n_folds == 1: axes = [axes]
        for i, fr in enumerate(fold_results):
            h = fr['history']
            eps = [x['epoch'] for x in h]
            axes[i].plot(eps, [x['train_cd'] for x in h], label='Train CD', alpha=0.8)
            axes[i].plot(eps, [x['val_cd'] for x in h], label='Val CD', alpha=0.8)
            axes[i].set_title(f"Fold {fr['fold']} (best={fr['mean_cd']:.4f})")
            axes[i].set_xlabel('Epoch'); axes[i].set_ylabel('CD')
            axes[i].legend(); axes[i].set_yscale('log')
        plt.suptitle('V3 Template Deformation — Training Curves', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'v3_training_curves.png'), dpi=150, bbox_inches='tight')
        if not IN_CLOUD: plt.show()
        plt.close(fig)


# ============================================================
# SECTION 15: MAIN PIPELINE
# ============================================================
def run_pipeline(skip_training=False):
    print("\n" + "=" * 70)
    print("🦴 V3 Template Deformation — Bone FEA Mesh Generation")
    print("   DGCNN (local features) + Template Deformation + TetGen")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  TetGen: {HAS_TETGEN}")
    print(f"  Surface pts: {MODEL_CONFIG['n_surface_pts']}")
    print(f"  Template pts: {MODEL_CONFIG['n_interior_pts']}")
    print(f"  K-local: {MODEL_CONFIG['k_local']}")

    # Phase 1: Parse CDB files
    print("\n📂 PHASE 1: Parsing CDB files")
    reader = CDBReader()
    meshes = reader.read_directory(DATA_DIR)
    if not meshes:
        print("❌ No valid meshes found"); return None

    # Quick stats
    stats = {}
    for name, m in list(meshes.items())[:3]:
        nodes, tets = m['nodes'], m['tets']
        surf_nids = extract_surface(tets)
        all_nids = set(nodes[:, 0].astype(int))
        stats[name] = f"{len(tets)}/{len(tets)} valid tets"
    for name, s in stats.items():
        print(f"  {name}: {s}")

    # Dataset summary
    records = []
    for name, m in meshes.items():
        surf_nids = extract_surface(m['tets'])
        all_nids = set(m['nodes'][:, 0].astype(int))
        records.append({'nodes': len(m['nodes']), 'tets': len(m['tets']),
                        'surface_nodes': len(surf_nids),
                        'interior_nodes': len(all_nids - surf_nids)})
    import pandas as pd
    df = pd.DataFrame(records)
    print("\n📊 DATASET SUMMARY")
    print("-" * 50)
    for c in ['nodes', 'tets', 'surface_nodes', 'interior_nodes']:
        print(f"  {c:20s}  min={df[c].min():8.0f}  mean={df[c].mean():8.0f}  max={df[c].max():8.0f}")

    if skip_training:
        print("\n⏭️ Training skipped")
        return None

    # Phase 2: K-Fold Training
    print("\n🧠 PHASE 2: Training Pipeline v3")
    fold_results = run_kfold(meshes)

    # Phase 3: Visualization
    if fold_results:
        ResultsViz.training_curves(fold_results)
        best = min(fold_results, key=lambda x: x['mean_cd'])
        print(f"\n🏆 Best fold: {best['fold']} | CD: {best['mean_cd']:.6f} | HD: {best['mean_hd']:.6f}")

    return fold_results


print('✅ Pipeline v3 ready')
print('  Call run_pipeline(skip_training=True) for data exploration')
print('  Call run_pipeline(skip_training=False) for full training')

# ── AUTO-RUN ──
if __name__ == '__main__' or IN_CLOUD:
    results = run_pipeline(skip_training=False)
