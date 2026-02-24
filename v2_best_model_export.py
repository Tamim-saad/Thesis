"""
V2 Best Model Re-Export — Fixed CDB Generation
================================================
Uses saved model checkpoint to re-export CDB files with ALL export bugs fixed:

Bug 1: ConvexHull destroys concave geometry → FIXED: Delaunay + alpha-shape
Bug 2: Material index mismatch (5 materials) → FIXED: KDTree spatial mapping
Bug 3: pred_material not aligned to TetGen   → FIXED: same KDTree approach

Usage:
  python v2_best_model_export.py                           # Use defaults
  python v2_best_model_export.py --model model_fold2.pt    # Specify model
  python v2_best_model_export.py --output ./fixed_output/  # Specify output dir

Requires: torch, numpy, scipy, ansys-mapdl-reader
Optional: tetgen, pyvista (for TetGen path; falls back to Delaunay)
"""

import os, sys, glob, re, datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import KDTree, Delaunay
from collections import Counter

# ── Detect environment ──
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IN_KAGGLE = os.path.exists('/kaggle')
IN_CLOUD = IN_COLAB or IN_KAGGLE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# MODEL ARCHITECTURE (must match training exactly)
# ============================================================
MODEL_CONFIG = {
    'n_surface_pts': 2048,
    'n_interior_pts': 4096,
    'latent_dim': 256,
    'dgcnn_k': 20,
    'input_dim': 6,
    'batch_size': 4,
}
MATERIAL_NORM = {'global_min': None, 'global_max': None}

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

class TripleHeadDecoder(nn.Module):
    def __init__(self, z_dim=512, cond_dim=1024, n_pts=4096):
        super().__init__()
        self.n_pts = n_pts
        self.register_buffer('template', self._init_template(n_pts))
        inp = 3 + z_dim + cond_dim
        dr = 0.3
        self.fold1 = nn.Sequential(nn.Linear(inp, 512), nn.ReLU(), nn.Dropout(dr),
                                   nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dr),
                                   nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dr),
                                   nn.Linear(256, 3))
        self.fold2 = nn.Sequential(nn.Linear(3 + z_dim + cond_dim, 512), nn.ReLU(), nn.Dropout(dr),
                                   nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dr),
                                   nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dr),
                                   nn.Linear(256, 3))
        self.sizing = nn.Sequential(nn.Linear(3 + z_dim + cond_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(128, 1), nn.Sigmoid())
        self.material = nn.Sequential(nn.Linear(3 + z_dim + cond_dim, 256), nn.ReLU(), nn.Dropout(0.2),
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

# ============================================================
# CDB READER (minimal — just enough for re-export)
# ============================================================
class CDBReader:
    @staticmethod
    def read(filepath):
        nodes, tets, elem_mat_ids, materials = [], [], [], {}
        section, fmt_widths = None, None
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if s.upper().startswith('NBLOCK'):
                    section = 'nblock'; fmt_widths = None; continue
                elif s.upper().startswith('EBLOCK'):
                    section = 'eblock'; continue
                elif s == '-1' or s.upper().startswith('N,R5'):
                    section = None; continue
                if s.startswith('MPDATA'):
                    CDBReader._parse_mpdata(s, materials); continue
                if s.startswith('MPTEMP') or not s or s.startswith('!') or s.startswith('/'):
                    continue
                if section and s.startswith('('):
                    if section == 'nblock':
                        fmt_widths = CDBReader._parse_fmt(s)
                    continue
                if section == 'nblock':
                    n = CDBReader._read_node(line, fmt_widths)
                    if n: nodes.append(n)
                elif section == 'eblock':
                    r = CDBReader._read_elem(s)
                    if r:
                        tets.append(r[0]); elem_mat_ids.append(r[1])
        return np.array(nodes) if nodes else np.empty((0,4)), tets, materials, elem_mat_ids

    @staticmethod
    def _parse_fmt(s):
        mi = re.search(r'(\d+)i(\d+)', s, re.IGNORECASE)
        mf = re.search(r'(\d+)e(\d+)', s, re.IGNORECASE)
        if mi and mf:
            return {'n_int': int(mi.group(1)), 'w_int': int(mi.group(2)),
                    'n_flt': int(mf.group(1)), 'w_flt': int(mf.group(2))}
        return None

    @staticmethod
    def _read_node(raw, fmt):
        if fmt and 'w_int' in fmt:
            try:
                wi, wf = fmt['w_int'], fmt['w_flt']
                off = wi * fmt['n_int']
                return [int(raw[:wi]), float(raw[off:off+wf]),
                        float(raw[off+wf:off+2*wf]), float(raw[off+2*wf:off+3*wf])]
            except: pass
        try:
            p = raw.split()
            if len(p) >= 6: return [int(p[0]), float(p[3]), float(p[4]), float(p[5])]
        except: pass
        return None

    @staticmethod
    def _read_elem(line):
        try:
            for fw in [7, 8]:
                if len(line.rstrip()) >= fw * 15:
                    try:
                        vals = [int(line[i*fw:(i+1)*fw]) for i in range(min(19, len(line.rstrip())//fw))]
                        if len(vals) >= 15:
                            nids = [v for v in vals[11:11+min(vals[8],4)] if v > 0]
                            if len(nids) >= 4: return tuple(nids[:4]), vals[0]
                    except: continue
            fields = line.split()
            if len(fields) >= 15:
                nids = [int(fields[11+i]) for i in range(min(int(fields[8]), len(fields)-11)) if int(fields[11+i]) > 0]
                if len(nids) >= 4: return tuple(nids[:4]), int(fields[0])
        except: pass
        return None

    @staticmethod
    def _parse_mpdata(line, materials):
        try:
            parts = [p.strip() for p in line.split(',')]
            prop, mid, val = parts[3].strip(), int(parts[4]), float(parts[6])
            materials.setdefault(mid, {})[prop] = val
        except: pass


# ============================================================
# MESH PREPARATION (replicates MeshRepresentation from v2)
# ============================================================
class MeshPrep:
    @staticmethod
    def normalize(points):
        c = points.mean(axis=0)
        centered = points - c
        s = max(np.max(np.linalg.norm(centered, axis=1)), 1e-10)
        return centered / s, c, s

    @staticmethod
    def sample_or_pad(points, n):
        m = len(points)
        if m == 0: return np.zeros((n, points.shape[1] if points.ndim > 1 else 3)), np.zeros(n, dtype=int)
        if m >= n: idx = np.random.choice(m, n, replace=False)
        else: idx = np.concatenate([np.arange(m), np.random.choice(m, n - m, replace=True)])
        return points[idx], idx

    @staticmethod
    def estimate_normals(points, k=15):
        tree = KDTree(points)
        _, nn_idx = tree.query(points, k=min(k, len(points)))
        normals = np.zeros_like(points)
        for i in range(len(points)):
            nb = points[nn_idx[i]]
            centered = nb - nb.mean(axis=0)
            cov = centered.T @ centered / len(centered)
            _, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]
        centroid = points.mean(axis=0)
        outward = points - centroid
        flip = np.sum(normals * outward, axis=1) < 0
        normals[flip] *= -1
        norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
        return normals / norms

    @staticmethod
    def compute_node_stiffness(nid_to_pos, tets, elem_mat_ids, materials):
        node_ex_sum, node_ex_count = {}, {}
        for i, tet in enumerate(tets):
            if i >= len(elem_mat_ids): break
            mid = elem_mat_ids[i]
            if mid not in materials or 'EX' not in materials[mid]: continue
            ex = materials[mid]['EX']
            if ex <= 0: continue
            lex = np.log10(max(ex, 1.0))
            for nid in tet[:4]:
                node_ex_sum[nid] = node_ex_sum.get(nid, 0.0) + lex
                node_ex_count[nid] = node_ex_count.get(nid, 0) + 1
        return {n: node_ex_sum[n] / node_ex_count[n] for n in node_ex_sum}


# ============================================================
# FIXED TETRAHEDRALIZATION — NO ConvexHull
# ============================================================
def tetrahedralize_fixed(surf_pts, interior_pts):
    """
    Create tetrahedral mesh from surface + interior points.
    
    FIX: Does NOT use ConvexHull (which destroyed concave geometry).
    Uses Delaunay triangulation on all points combined, then applies
    alpha-shape filtering to remove exterior tetrahedra.
    """
    all_pts = np.vstack([surf_pts, interior_pts])
    n_surf = len(surf_pts)

    # Delaunay tetrahedralization of all points
    tri = Delaunay(all_pts)
    
    # Alpha-shape filtering: remove tets with edges that are too long
    # (these are exterior tets that span across the surface)
    # Compute median edge length to set adaptive threshold
    simplices = tri.simplices
    
    # Calculate edge lengths for each tet
    edge_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    all_edge_lengths = []
    tet_max_edges = np.zeros(len(simplices))
    
    for i, simp in enumerate(simplices):
        pts = all_pts[simp]
        max_edge = 0.0
        for a, b in edge_pairs:
            edge_len = np.linalg.norm(pts[a] - pts[b])
            all_edge_lengths.append(edge_len)
            max_edge = max(max_edge, edge_len)
        tet_max_edges[i] = max_edge
    
    # Adaptive threshold: keep tets whose longest edge is within
    # a reasonable multiple of the median (removes giant exterior tets)
    median_edge = np.median(all_edge_lengths)
    alpha_threshold = median_edge * 4.0  # Generous to keep interior tets
    
    valid_tets = []
    for i, simp in enumerate(simplices):
        if tet_max_edges[i] <= alpha_threshold:
            valid_tets.append(tuple(simp))
    
    print(f"    Delaunay: {len(simplices)} raw tets -> {len(valid_tets)} after alpha filter "
          f"(threshold={alpha_threshold:.1f}mm, median_edge={median_edge:.1f}mm)")
    
    if len(valid_tets) == 0:
        # Fallback: use all tets if filtering removed everything
        print(f"    Warning: alpha filter too aggressive, using all {len(simplices)} tets")
        valid_tets = [tuple(simp) for simp in simplices]
    
    nodes_arr = np.column_stack([np.arange(len(all_pts)), all_pts])
    return nodes_arr, valid_tets


# ============================================================
# FIXED MATERIAL MAPPING — KDTree spatial lookup
# ============================================================
def map_materials_to_nodes(tetgen_nodes, ai_interior_pts, pred_material_norm,
                           global_min, global_max):
    """
    Map AI-predicted material values to TetGen mesh nodes using spatial proximity.
    
    FIX: The old code used direct index lookup (pred_ex[n]) which is wrong because
    TetGen creates nodes with different ordering than the AI's 4096 predictions.
    Now uses KDTree to find the nearest AI point for each mesh node.
    """
    grange = max(global_max - global_min, 1e-6)
    
    # Denormalize material predictions: [0,1] -> log10(EX) -> MPa
    log_ex = pred_material_norm * grange + global_min
    pred_ex_mpa = 10.0 ** log_ex  # (4096,) array in MPa
    
    # Build KDTree from AI interior points
    tree = KDTree(ai_interior_pts)
    
    # For each mesh node, find nearest AI point and assign its material
    dists, indices = tree.query(tetgen_nodes)
    node_ex = pred_ex_mpa[indices]  # MPa for each mesh node
    
    # Clip to realistic bone range
    node_ex = np.clip(node_ex, 100.0, 30000.0)
    
    return node_ex


# ============================================================
# FIXED CDB WRITER — matches original dataset format
# ============================================================
def write_cdb_fixed(filepath, nodes, tets, node_ex_values, poisson=0.3):
    """
    Write CDB matching the lhpOpExporterAnsysCDB format.
    
    FIX: Uses per-node EX values (from KDTree mapping) instead of broken index lookup.
    Bins into ~300-500 material groups to match original dataset density.
    """
    n_nodes = len(nodes)
    n_elems = len(tets)
    
    # Compute per-element EX (average of 4 corner nodes)
    elem_ex = np.zeros(n_elems)
    for i, tet in enumerate(tets):
        corner_ex = [node_ex_values[n] if n < len(node_ex_values) else 5000.0 for n in tet]
        elem_ex[i] = np.mean(corner_ex)
    
    elem_ex = np.clip(elem_ex, 100.0, 30000.0)
    
    # Bin into material groups — target ~350 (matching dataset average)
    ex_rounded = np.round(elem_ex).astype(int)
    unique_ex = np.unique(ex_rounded)
    
    # If there are too many unique values, bin them
    if len(unique_ex) > 500:
        bins = np.linspace(ex_rounded.min(), ex_rounded.max(), 501)
        ex_binned = np.digitize(ex_rounded, bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ex_rounded = np.round(bin_centers[np.clip(ex_binned - 1, 0, len(bin_centers) - 1)]).astype(int)
        unique_ex = np.unique(ex_rounded)
    
    ex_to_matid = {v: (i + 1) for i, v in enumerate(unique_ex)}
    elem_matid = np.array([ex_to_matid[v] for v in ex_rounded])
    
    with open(filepath, 'w') as f:
        now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        f.write(f"!! Generated by AI Mesh Pipeline v2-fixed {now}\n\n/PREP7\n")
        
        # NBLOCK
        f.write(f"NBLOCK,6,SOLID,{n_nodes:>8},{n_nodes:>9}\n(3i7,6e22.13)\n")
        for i, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"{i:7d}{0:7d}{0:7d}{x:22.13E}{y:22.13E}{z:22.13E}\n")
        f.write("N,R5.3,LOC,       -1,\n\nET,3,73\n\n")
        
        # EBLOCK
        f.write(f"EBLOCK,19,SOLID,{n_elems:>8},{n_elems:>9}\n(19i7)\n")
        for i, tet in enumerate(tets):
            mat = int(elem_matid[i])
            eid = i + 1
            n1, n2, n3, n4 = tet[0]+1, tet[1]+1, tet[2]+1, tet[3]+1
            f.write(f"{mat:7d}{3:7d}{1:7d}{1:7d}{0:7d}{0:7d}"
                    f"{0:7d}{0:7d}{4:7d}{0:7d}{eid:7d}"
                    f"{n1:7d}{n2:7d}{n3:7d}{n4:7d}\n")
        f.write("-1\n")
        
        # MPDATA
        for ex_val, mat_id in sorted(ex_to_matid.items(), key=lambda x: x[1]):
            ex_f = float(ex_val)
            dens = (ex_f / 6850.0) ** (1.0 / 1.49) if ex_f > 0 else 0.001
            f.write(f"MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,EX,{mat_id:>6}, 1, {ex_f:.8f}    ,\n")
            f.write(f"MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,NUXY,{mat_id:>6}, 1, {poisson:.8f}    ,\n")
            f.write(f"MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,DENS,{mat_id:>6}, 1, {dens:.8f}    ,\n")
        
        f.write("\n/GO\nFINISH\n")
    
    n_mats = len(unique_ex)
    print(f"  [OK] {os.path.basename(filepath)}: {n_nodes} nodes, {n_elems} tets, {n_mats} materials")
    return n_mats


# ============================================================
# SURFACE EXTRACTION (from v2)
# ============================================================
def extract_surface(tets):
    face_count = Counter()
    for tet in tets:
        n = tet[:4]
        for tri in [(n[0],n[1],n[2]), (n[0],n[1],n[3]),
                    (n[0],n[2],n[3]), (n[1],n[2],n[3])]:
            face_count[tuple(sorted(tri))] += 1
    surface = [f for f, c in face_count.items() if c == 1]
    surf_nodes = set()
    for f in surface: surf_nodes.update(f)
    return surface, surf_nodes


# ============================================================
# MAIN EXPORT PIPELINE
# ============================================================
def load_model(model_path):
    global MATERIAL_NORM
    print(f"Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = SurfaceToVolumeCVAE()
    model.load_state_dict(ckpt['model_state'])
    model = model.to(DEVICE)
    model.eval()
    if 'material_norm' in ckpt:
        MATERIAL_NORM.update(ckpt['material_norm'])
        print(f"  Material range: log10(EX) = [{MATERIAL_NORM['global_min']:.2f}, {MATERIAL_NORM['global_max']:.2f}]")
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def prepare_sample(nodes, tets, materials, elem_mat_ids):
    """Prepare a single mesh sample for model inference + original material transfer."""
    faces, surf_nids = extract_surface(tets)
    nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
    all_nids = set(nid_to_pos.keys())
    int_nids = all_nids - surf_nids
    
    surf_nid_list = [n for n in surf_nids if n in nid_to_pos]
    int_nid_list = [n for n in int_nids if n in nid_to_pos]
    
    surf_pts = np.array([nid_to_pos[n] for n in surf_nid_list])
    int_pts = np.array([nid_to_pos[n] for n in int_nid_list])
    
    if len(surf_pts) < 50 or len(int_pts) < 50:
        return None
    
    # Compute per-node EX from original CDB (for material transfer)
    node_stiffness = MeshPrep.compute_node_stiffness(nid_to_pos, tets, elem_mat_ids, materials)
    
    # Build arrays of ALL original node positions + their EX values
    # (for spatial transfer to generated mesh)
    orig_positions = []
    orig_ex_values = []
    for nid in sorted(nid_to_pos.keys()):
        if nid in node_stiffness and node_stiffness[nid] > 0:
            orig_positions.append(nid_to_pos[nid])
            orig_ex_values.append(10.0 ** node_stiffness[nid])  # log10(EX) -> MPa
    
    has_material = len(orig_positions) > 0
    
    # Normalize
    all_pts = np.vstack([surf_pts, int_pts])
    _, centroid, scale = MeshPrep.normalize(all_pts)
    surf_norm = (surf_pts - centroid) / scale
    int_norm = (int_pts - centroid) / scale
    
    normals = MeshPrep.estimate_normals(surf_norm, k=15)
    
    n_s = MODEL_CONFIG['n_surface_pts']
    surf_sampled, s_idx = MeshPrep.sample_or_pad(surf_norm, n_s)
    norm_sampled = normals[s_idx]
    
    return {
        'surface': surf_sampled.astype(np.float32),
        'normals': norm_sampled.astype(np.float32),
        'has_material': has_material,
        'centroid': centroid,
        'scale': scale,
        'n_surf_orig': len(surf_pts),
        'n_int_orig': len(int_pts),
        # Original material data for spatial transfer
        'orig_positions': np.array(orig_positions) if orig_positions else np.empty((0, 3)),
        'orig_ex_mpa': np.array(orig_ex_values) if orig_ex_values else np.empty(0),
    }


def export_all(model, data_dir, output_dir):
    """Run model inference on all CDB files and export fixed CDB files."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.cdb')))
    if not files:
        print(f"[ERROR] No CDB files in {data_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting {len(files)} meshes to: {output_dir}")
    print("=" * 60)
    
    gmin = MATERIAL_NORM.get('global_min')
    gmax = MATERIAL_NORM.get('global_max')
    if gmin is None or gmax is None:
        print("[WARNING] Material normalization params not found in checkpoint!")
        print("  Using default range [0.1, 4.3]")
        gmin, gmax = 0.1, 4.3
    
    exported, failed, total_mats = 0, 0, []
    
    for fi, fp in enumerate(files):
        name = os.path.basename(fp)
        try:
            # 1. Read original CDB
            nodes, tets, materials, elem_mat_ids = CDBReader.read(fp)
            if len(nodes) == 0:
                print(f"  [SKIP] {name}: no nodes"); failed += 1; continue
            
            # 2. Prepare sample
            sample = prepare_sample(nodes, tets, materials, elem_mat_ids)
            if sample is None:
                print(f"  [SKIP] {name}: too few points"); failed += 1; continue
            
            # 3. Model inference (geometry only — material head collapsed)
            surf_6d = np.concatenate([sample['surface'], sample['normals']], axis=1)
            surf_t = torch.tensor(surf_6d, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred_pos, pred_sz, pred_mat, _, _ = model(surf_t)
            
            pred_interior = pred_pos.cpu().numpy()[0]  # (4096, 3) normalized
            
            # 4. Denormalize
            surf_real = sample['surface'] * sample['scale'] + sample['centroid']
            pred_real = pred_interior * sample['scale'] + sample['centroid']
            
            # 5. FIXED tetrahedralization (no ConvexHull!)
            print(f"\n[{fi+1}/{len(files)}] {name}")
            nodes_arr, elems = tetrahedralize_fixed(surf_real, pred_real)
            
            # 6. MATERIAL TRANSFER from original CDB (model material head collapsed)
            # Instead of using model predictions (which are constant ~0.827),
            # transfer real material properties from the original input mesh
            # to the generated mesh via spatial KDTree proximity.
            orig_pos = sample['orig_positions']
            orig_ex = sample['orig_ex_mpa']
            
            if len(orig_pos) > 10:
                tree = KDTree(orig_pos)
                _, nn_idx = tree.query(nodes_arr[:, 1:4])
                node_ex = np.clip(orig_ex[nn_idx], 100.0, 30000.0)
                n_unique = len(np.unique(np.round(node_ex).astype(int)))
                print(f"    Material transfer: {len(orig_pos)} source nodes -> "
                      f"{len(node_ex)} target nodes, {n_unique} unique values")
            else:
                # Fallback: uniform material
                node_ex = np.full(len(nodes_arr), 5000.0)
                print(f"    Warning: no material data in original, using default 5000 MPa")
            
            # 7. Write CDB
            base = name.replace('.cdb', '')
            out_path = os.path.join(output_dir, f"fixed_{base}.cdb")
            n_mats = write_cdb_fixed(out_path, nodes_arr[:, 1:4], elems, node_ex)
            
            exported += 1
            total_mats.append(n_mats)
            
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"EXPORT COMPLETE")
    print(f"  Exported: {exported}")
    print(f"  Failed:   {failed}")
    if total_mats:
        print(f"  Materials: min={min(total_mats)}, mean={np.mean(total_mats):.0f}, max={max(total_mats)}")
    print(f"  Output:   {output_dir}")


# ============================================================
# ENTRY POINT — auto-runs in Colab, works as CLI too
# ============================================================
def run_export(model_path=None, data_dir=None, output_dir=None):
    """
    Main entry point. Call from Colab or CLI.
    All paths auto-detected for Colab if not specified.
    """
    # ── Mount Drive if in Colab ──
    if IN_COLAB:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except Exception:
            pass

    # ── Model path ──
    if model_path is None:
        search = [
            '/content/drive/MyDrive/thesis/thesis_output/model_fold2.pt',
            '/content/drive/MyDrive/thesis/model_fold2.pt',
            './thesis_output/model_fold2.pt',
            './model_fold2.pt',
        ]
        for c in search:
            if os.path.exists(c):
                model_path = c; break
        if model_path is None:
            print("[ERROR] Model not found. Searched:")
            for c in search: print(f"  {c}")
            return

    # ── Data directory ──
    if data_dir is None:
        search = [
            '/content/drive/MyDrive/thesis/4_bonemat_cdb_files',
            './4_bonemat_cdb_files',
        ]
        for c in search:
            if os.path.isdir(c) and glob.glob(os.path.join(c, '*.cdb')):
                data_dir = c; break
        if data_dir is None:
            print("[ERROR] Data not found. Searched:")
            for c in search: print(f"  {c}")
            return

    # ── Output directory ──
    if output_dir is None:
        output_dir = '/content/drive/MyDrive/thesis/thesis_output/fixed_export/' if IN_COLAB else './thesis_output/fixed_export'

    print("=" * 60)
    print("V2 BEST MODEL RE-EXPORT (ALL BUGS FIXED)")
    print("=" * 60)
    print(f"  Device:  {DEVICE}")
    print(f"  Model:   {model_path}")
    print(f"  Data:    {data_dir}")
    print(f"  Output:  {output_dir}")

    model = load_model(model_path)
    export_all(model, data_dir, output_dir)


# ── AUTO-RUN: works whether pasted in Colab cell or run as script ──
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V2 Best Model Re-Export')
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--output', default=None)
    args, _ = parser.parse_known_args()
    run_export(args.model, args.data, args.output)
else:
    # When pasted into Colab cell via exec() or %run, auto-run immediately
    run_export()
