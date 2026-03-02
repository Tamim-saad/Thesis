"""
V4 Export Script: Enhanced Template Deformation Model → CDB Files
=================================================================
Loads trained V4 model, predicts interior points + materials,
runs TetGen for tet meshing, writes ANSYS CDB files.

Usage (Colab):
  exec(open('/content/drive/MyDrive/thesis/v4_export.py').read())

Usage (local):
  python v4_export.py --model model_v4_fold3.pt --data ./4_bonemat_cdb_files --output ./v4_output/
"""

import os, sys, subprocess, glob, re, datetime, argparse
import numpy as np

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IN_CLOUD = IN_COLAB or os.path.exists('/kaggle')

if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except: pass
    # Install pyvista for surface-enclosed tet filtering
    try:
        import pyvista
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista'])
        print('  ✅ Installed pyvista')

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from collections import Counter

try:
    import tetgen as _tetgen_lib
    HAS_TETGEN = True
except ImportError:
    HAS_TETGEN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# MODEL CONFIG (must match training exactly)
# ============================================================
MODEL_CONFIG = {
    'n_surface_pts': 4096,
    'n_interior_pts': 8192,
    'k_local': 8,
    'encoder_dim': 256,
    'global_dim': 512,
    'hidden_dim': 256,
    'dgcnn_k': 20,
    'disp_scale': 0.3,
    'n_refine_steps': 3,         # V4: must match training (was 2)
}
MATERIAL_NORM = {'global_min': None, 'global_max': None}

# ============================================================
# MODEL ARCHITECTURE (must match training)
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
    def __init__(self, k=20, in_dim=6, point_dim=256):
        super().__init__()
        self.k = k
        self.ec1 = nn.Sequential(nn.Conv2d(in_dim*2, 64, 1, bias=False),
                                 nn.GroupNorm(8, 64), nn.LeakyReLU(0.2))
        self.ec2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                 nn.GroupNorm(8, 128), nn.LeakyReLU(0.2))
        self.ec3 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                 nn.GroupNorm(16, 256), nn.LeakyReLU(0.2))
        self.point_proj = nn.Sequential(
            nn.Conv1d(64+128+256, point_dim, 1, bias=False),
            nn.GroupNorm(16, point_dim), nn.LeakyReLU(0.2))
        self.global_agg = nn.Sequential(
            nn.Conv1d(64+128+256, point_dim, 1, bias=False),
            nn.GroupNorm(16, point_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        B = x.size(0)
        x1 = self.ec1(edge_features(x, self.k)).max(-1)[0]
        x2 = self.ec2(edge_features(x1, self.k)).max(-1)[0]
        x3 = self.ec3(edge_features(x2, self.k)).max(-1)[0]
        cat = torch.cat([x1, x2, x3], dim=1)
        point_feat = self.point_proj(cat).transpose(1, 2)
        g = self.global_agg(cat)
        g_max = F.adaptive_max_pool1d(g, 1).view(B, -1)
        g_avg = F.adaptive_avg_pool1d(g, 1).view(B, -1)
        return torch.cat([g_max, g_avg], dim=1), point_feat

class TemplateDeformNet(nn.Module):
    def __init__(self, global_dim=512, local_dim=256, hidden=256, k_local=8):
        super().__init__()
        self.k_local = k_local
        inp = 3 + local_dim + global_dim
        self.disp_fc1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.disp_fc2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.disp_skip = nn.Linear(inp, hidden)
        self.disp_out = nn.Linear(hidden, 3)
        self.mat_head = nn.Sequential(
            nn.Linear(inp, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def _get_local_features(self, template, surf_xyz, point_feat):
        """V4: Distance-weighted aggregation of k nearest surface features."""
        B, T, _ = template.shape
        _, S, D = point_feat.shape
        k = self.k_local
        local_feats = []
        chunk = 1024
        for i in range(0, T, chunk):
            t_chunk = template[:, i:min(i+chunk, T)]
            c = t_chunk.shape[1]
            dist = torch.cdist(t_chunk, surf_xyz)
            topk_dist, nn_idx = dist.topk(k, dim=2, largest=False)
            flat_idx = nn_idx.reshape(B, -1)
            flat_exp = flat_idx.unsqueeze(-1).expand(-1, -1, D)
            gathered = torch.gather(point_feat, 1, flat_exp)
            gathered = gathered.reshape(B, c, k, D)
            # V4: Inverse distance weights
            weights = 1.0 / (topk_dist + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            local = (gathered * weights.unsqueeze(-1)).sum(dim=2)
            local_feats.append(local)
        return torch.cat(local_feats, dim=1)

    def forward(self, template, surf_xyz, global_feat, point_feat):
        B, T, _ = template.shape
        local_feat = self._get_local_features(template, surf_xyz, point_feat)
        global_exp = global_feat.unsqueeze(1).expand(-1, T, -1)
        node_input = torch.cat([template, local_feat, global_exp], dim=2)
        # V4: Residual displacement
        h = self.disp_fc1(node_input)
        h = self.disp_fc2(h) + self.disp_skip(node_input)
        disp = self.disp_out(h)
        mat = self.mat_head(node_input).squeeze(-1)
        disp = disp * MODEL_CONFIG['disp_scale']
        return disp, mat

class SurfaceToVolumeModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = MODEL_CONFIG
        self.encoder = DGCNNEncoder(k=cfg['dgcnn_k'], in_dim=6, point_dim=cfg['encoder_dim'])
        self.decoder = TemplateDeformNet(
            global_dim=cfg['global_dim'], local_dim=cfg['encoder_dim'],
            hidden=cfg['hidden_dim'], k_local=cfg['k_local'])
        self.n_refine_steps = cfg.get('n_refine_steps', 1)

    def forward(self, surface, template):
        # Encode surface ONCE
        surf_t = surface.transpose(1, 2)
        global_feat, point_feat = self.encoder(surf_t)
        surf_xyz = surface[:, :, :3]
        # Iterative refinement (must match training)
        pos = template.clone()
        mat = None
        for step in range(self.n_refine_steps):
            disp, mat = self.decoder(pos, surf_xyz, global_feat, point_feat)
            pos = pos + disp
        total_disp = pos - template
        return total_disp, mat

# ============================================================
# CDB READER (same as training)
# ============================================================
class CDBReader:
    def read(self, filepath):
        nodes, tets, elem_mat_ids, materials = [], [], [], {}
        section = None; fmt = None
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if s.upper().startswith('NBLOCK'): section = 'nblock'; fmt = None; continue
                elif s.upper().startswith('EBLOCK'): section = 'eblock'; continue
                elif s == '-1' or s.upper().startswith('N,R5'): section = None; continue
                if s.startswith('MPDATA'): self._parse_mpdata(s, materials); continue
                if s.startswith('MPTEMP'): continue
                if section and s.startswith('('):
                    if section == 'nblock':
                        mi = re.search(r'(\d+)i(\d+)', s, re.I)
                        mf = re.search(r'(\d+)e(\d+)', s, re.I)
                        fmt = {'n_int': int(mi.group(1)), 'w_int': int(mi.group(2)),
                               'n_flt': int(mf.group(1)), 'w_flt': int(mf.group(2))} if mi and mf else None
                    continue
                if not s or s.startswith('!') or s.startswith('/'): continue
                if section == 'nblock':
                    nd = self._read_node(line, fmt)
                    if nd: nodes.append(nd)
                elif section == 'eblock':
                    r = self._read_elem(s)
                    if r: tets.append(r[0]); elem_mat_ids.append(r[1])
        nodes = np.array(nodes) if nodes else np.empty((0, 4))
        base = os.path.basename(filepath).replace('_bonemat.cdb', '').replace('_re', '')
        return nodes, tets, {'patient_id': base.split('_')[0],
            'side': 'left' if 'left' in filepath.lower() else 'right',
            'filepath': filepath, 'materials': materials, 'elem_mat_ids': elem_mat_ids}

    def _read_node(self, raw, fmt):
        if fmt:
            try:
                wi, wf = fmt['w_int'], fmt['w_flt']; off = wi * fmt['n_int']
                return [int(raw[:wi]), float(raw[off:off+wf]), float(raw[off+wf:off+2*wf]), float(raw[off+2*wf:off+3*wf])]
            except: pass
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
                            if len(nids) >= 4: return tuple(nids[:4]), vals[0]
                    except (ValueError, IndexError): continue
            # Fallback: whitespace split
            fields = line.split()
            if len(fields) >= 15:
                nids = [int(fields[11+i]) for i in range(min(4, len(fields)-11)) if int(fields[11+i]) > 0]
                if len(nids) >= 4:
                    return tuple(nids[:4]), int(fields[0])
        except (ValueError, IndexError): pass
        return None

    @staticmethod
    def _parse_mpdata(line, materials):
        try:
            parts = [p.strip() for p in line.split(',')]
            materials.setdefault(int(parts[4]), {})[parts[3].strip()] = float(parts[6])
        except: pass

# ============================================================
# PROCESSING UTILITIES
# ============================================================
def extract_surface(tets):
    face_count = Counter()
    for tet in tets:
        n = tet[:4]
        for tri in [(n[0],n[1],n[2]), (n[0],n[1],n[3]), (n[0],n[2],n[3]), (n[1],n[2],n[3])]:
            face_count[tuple(sorted(tri))] += 1
    return {n for f, c in face_count.items() if c == 1 for n in f}

def extract_surface_faces(tets):
    """Extract boundary faces (triangles) from tet connectivity.
    Returns list of (n1, n2, n3) tuples — faces appearing exactly once."""
    face_count = Counter()
    face_original = {}  # sorted_key → original (unsorted) face
    for tet in tets:
        n = tet[:4]
        for tri in [(n[0],n[1],n[2]), (n[0],n[1],n[3]), (n[0],n[2],n[3]), (n[1],n[2],n[3])]:
            key = tuple(sorted(tri))
            face_count[key] += 1
            face_original[key] = tri
    return [face_original[f] for f, c in face_count.items() if c == 1]

def estimate_normals(points, k=15):
    tree = KDTree(points)
    _, nn_idx = tree.query(points, k=min(k, len(points)))
    normals = np.zeros_like(points)
    for i in range(len(points)):
        nb = points[nn_idx[i]]
        cov = (nb - nb.mean(0)).T @ (nb - nb.mean(0)) / len(nb)
        _, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]
    outward = points - points.mean(0)
    normals[np.sum(normals * outward, axis=1) < 0] *= -1
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    return normals

def sample_or_pad(pts, n):
    m = len(pts)
    idx = np.random.choice(m, n, replace=(m < n))
    return pts[idx], idx

# ============================================================
# CDB WRITER
# ============================================================
def write_cdb(filepath, nodes, tets, material_values=None, poisson=0.3):
    n_nodes, n_elems = len(nodes), len(tets)
    with open(filepath, 'w') as f:
        now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        f.write(f"!! Generated by AI Mesh Pipeline v4 {now}\n\n/PREP7\n")
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
            elem_mid = np.ones(n_elems, dtype=int); ex_to_mid = None
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
            f.write("MPTEMP,R5.0, 1, 1,  0.00000000    ,\nMPDATA,R5.0, 1,EX,     1, 1, 10000.00000000    ,\n")
            f.write("MPTEMP,R5.0, 1, 1,  0.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,NUXY,     1, 1, {poisson:.8f}    ,\n")
            f.write("MPTEMP,R5.0, 1, 1,  0.00000000    ,\nMPDATA,R5.0, 1,DENS,     1, 1, 1.00000000    ,\n")
        f.write("\n/GO\nFINISH\n")

# ============================================================
# MAIN EXPORT FUNCTION
# ============================================================
def run_export(model_path=None, data_dir=None, output_dir=None):
    if IN_COLAB:
        model_path = model_path or '/content/drive/MyDrive/thesis/thesis_output/model_v4_fold1.pt'
        data_dir = data_dir or '/content/drive/MyDrive/thesis/4_bonemat_cdb_files'
        output_dir = output_dir or '/content/drive/MyDrive/thesis/thesis_output/v4_export/'
    else:
        model_path = model_path or './thesis_output/model_v4_fold1.pt'
        data_dir = data_dir or './4_bonemat_cdb_files'
        output_dir = output_dir or './thesis_output/v4_export/'

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("V4 ENHANCED TEMPLATE DEFORMATION — EXPORT")
    print("=" * 60)
    print(f"  Device:  {DEVICE}")
    print(f"  Model:   {model_path}")
    print(f"  Data:    {data_dir}")
    print(f"  Output:  {output_dir}")

    # Load model
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Handle both old (raw state_dict) and new (full checkpoint) formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'material_norm' in checkpoint:
            mn = checkpoint['material_norm']
            MATERIAL_NORM['global_min'] = mn.get('global_min')
            MATERIAL_NORM['global_max'] = mn.get('global_max')
            print(f"  Material norm from checkpoint: [{mn.get('global_min'):.2f}, {mn.get('global_max'):.2f}]")
    else:
        state_dict = checkpoint

    model = SurfaceToVolumeModel()
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded ({n_params:,} params)")

    # Load template — prefer per-fold template matching the model fold
    model_base = os.path.basename(model_path)
    fold_num = None
    import re as _re
    m = _re.search(r'fold(\d+)', model_base)
    if m:
        fold_num = int(m.group(1))

    template = None
    model_dir = os.path.dirname(model_path)

    # Priority 1: per-fold template
    if fold_num:
        fold_tpl = os.path.join(model_dir, f'v4_template_fold{fold_num}.npy')
        if os.path.exists(fold_tpl):
            template = np.load(fold_tpl)
            print(f"  Template (fold {fold_num}): {fold_tpl} ({template.shape})")

    # Priority 2: global template
    if template is None:
        global_tpl = os.path.join(model_dir, 'v4_template.npy')
        if os.path.exists(global_tpl):
            template = np.load(global_tpl)
            print(f"  Template (global): {global_tpl} ({template.shape})")
            print("  ⚠️ Using global template — for best results, retrain with per-fold templates")

    # Read all CDB files
    reader = CDBReader()
    cdb_files = sorted(glob.glob(os.path.join(data_dir, '*.cdb')))
    print(f"\nExporting {len(cdb_files)} meshes to: {output_dir}")
    print("=" * 60)

    # Priority 3: compute from data (worst option)
    if template is None:
        print("  ⚠️ No template found — computing from all CDB files...")
        all_interiors = []
        n_int = MODEL_CONFIG['n_interior_pts']
        for fi, fp in enumerate(cdb_files):
            try:
                nodes, tets, meta = reader.read(fp)
                if len(nodes) == 0 or len(tets) == 0:
                    continue
                nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
                surf_nids = extract_surface(tets)
                valid_surf = [n for n in surf_nids if n in nid_to_pos]
                valid_int = [n for n in nid_to_pos if n not in surf_nids]
                if len(valid_surf) < 50 or len(valid_int) < 50:
                    continue
                surf_pts = np.array([nid_to_pos[n] for n in valid_surf])
                int_pts = np.array([nid_to_pos[n] for n in valid_int])
                all_p = np.vstack([surf_pts, int_pts])
                c = all_p.mean(0)
                s = max(np.linalg.norm(all_p - c, axis=1).max(), 1e-10)
                int_norm = (int_pts - c) / s
                sampled, _ = sample_or_pad(int_norm, n_int)
                all_interiors.append(sampled)
                if (fi+1) % 50 == 0 or fi == 0:
                    print(f"    [{fi+1}/{len(cdb_files)}] ✅ processed")
            except Exception as e:
                print(f"    Error: {e}")
        if all_interiors:
            template = np.mean(all_interiors, axis=0).astype(np.float32)
            tpl_save = os.path.join(model_dir, 'v4_template.npy')
            np.save(tpl_save, template)
            print(f"  📐 Template computed from {len(all_interiors)} samples")
        else:
            print("  ❌ Could not compute template")
            return

    # Detect material range
    gmin = MATERIAL_NORM.get('global_min')
    gmax = MATERIAL_NORM.get('global_max')
    if gmin is None:
        # Try loading from saved file
        norm_path = os.path.join(os.path.dirname(model_path), 'v4_material_norm.npy')
        if os.path.exists(norm_path):
            norm_arr = np.load(norm_path)
            gmin, gmax = float(norm_arr[0]), float(norm_arr[1])
            print(f"  Material norm from file: [{gmin:.2f}, {gmax:.2f}]")
    if gmin is None:
        # Last resort: scan CDB files
        all_logex = []
        print("  Scanning CDB files for material range...")
        for fp in cdb_files:
            nodes, tets, meta = reader.read(fp)
            for mid, props in meta['materials'].items():
                if 'EX' in props and props['EX'] > 0:
                    all_logex.append(np.log10(max(props['EX'], 1.0)))
        if all_logex:
            gmin, gmax = min(all_logex), max(all_logex)
        else:
            gmin, gmax = 0.1, 4.3
    print(f"  Material range: log10(EX) = [{gmin:.2f}, {gmax:.2f}]")
    grange = max(gmax - gmin, 1e-6)

    success, failed = 0, 0
    for fi, fp in enumerate(cdb_files):
        name = os.path.basename(fp)
        print(f"\n[{fi+1}/{len(cdb_files)}] {name}")
        try:
            # 1. Read original CDB (only surface used as model INPUT)
            nodes, tets, meta = reader.read(fp)
            nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
            surf_nids = extract_surface(tets)
            surf_faces = extract_surface_faces(tets)  # for inside/outside test
            valid_surf = [n for n in surf_nids if n in nid_to_pos]
            valid_int = [n for n in nid_to_pos if n not in surf_nids]

            if len(valid_surf) < 50 or len(valid_int) < 50:
                print(f"    ⚠️ Skipped (too few points)")
                continue

            surf_pts = np.array([nid_to_pos[n] for n in valid_surf])
            int_pts = np.array([nid_to_pos[n] for n in valid_int])

            # 2. Normalize surface for model input
            all_pts = np.vstack([surf_pts, int_pts])
            centroid = all_pts.mean(axis=0)
            scale = max(np.linalg.norm(all_pts - centroid, axis=1).max(), 1e-10)
            surf_norm = (surf_pts - centroid) / scale

            # 3. Prepare surface input (xyz + normals)
            normals_est = estimate_normals(surf_norm)
            n_s = MODEL_CONFIG['n_surface_pts']
            surf_sampled, s_idx = sample_or_pad(surf_norm, n_s)
            norm_sampled = normals_est[s_idx]
            surf_input = np.hstack([surf_sampled, norm_sampled]).astype(np.float32)

            # 4. Model forward pass → predicted interior positions + materials
            surf_t = torch.tensor(surf_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            temp_t = torch.tensor(template, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_disp, pred_mat = model(surf_t, temp_t)

            pred_pos = (temp_t + pred_disp)[0].cpu().numpy()   # (8192, 3) normalized
            pred_mat_vals = pred_mat[0].cpu().numpy()           # (8192,) in [0,1]

            # 5. Denormalize MODEL OUTPUT to real coordinates
            pred_real = pred_pos * scale + centroid

            # 6. Denormalize materials: [0,1] → log10(EX) → EX
            mat_log = pred_mat_vals * grange + gmin
            mat_ex = 10.0 ** mat_log  # actual EX in MPa

            # 7. Combine surface (input) + predicted interior (model output)
            combined_pts = np.vstack([surf_pts, pred_real])
            n_surf = len(surf_pts)

            # 8. Delaunay tetrahedralization
            from scipy.spatial import Delaunay
            tri = Delaunay(combined_pts)
            simp = tri.simplices

            # --- Stage 1: Surface-enclosed centroid test ---
            # Build watertight surface mesh from original CDB faces.
            # Test each Delaunay tet centroid: is it INSIDE the bone?
            # This correctly handles concavities (femoral neck, trochanter).
            try:
                import pyvista as pv
                # Build node-id → sequential index mapping for surface faces
                surf_nid_list = sorted(valid_surf)
                nid_to_idx = {nid: idx for idx, nid in enumerate(surf_nid_list)}
                surf_pts_ordered = np.array([nid_to_pos[n] for n in surf_nid_list])

                # Convert face node-IDs to sequential indices
                pv_faces = []
                for f in surf_faces:
                    if all(int(n) in nid_to_idx for n in f):
                        pv_faces.append([3, nid_to_idx[int(f[0])],
                                            nid_to_idx[int(f[1])],
                                            nid_to_idx[int(f[2])]])
                if len(pv_faces) > 100:
                    pv_faces_arr = np.array(pv_faces, dtype=np.int64).ravel()
                    surface_mesh = pv.PolyData(surf_pts_ordered, pv_faces_arr)

                    # Compute tet centroids
                    tet_verts = combined_pts[simp]  # (N_tets, 4, 3)
                    centroids = tet_verts.mean(axis=1)  # (N_tets, 3)

                    # Inside/outside test via signed distance
                    centroid_cloud = pv.PolyData(centroids)
                    try:
                        # Try modern API first
                        result = centroid_cloud.select_enclosed_points(
                            surface_mesh, check_surface=False)
                        inside_mask = result['SelectedPoints'].astype(bool)
                    except Exception:
                        # Fallback: signed distance approach
                        # Points with negative implicit distance are inside
                        dists = centroid_cloud.compute_implicit_distance(surface_mesh)
                        inside_mask = dists['implicit_distance'] < 0

                    filtered_tets = simp[inside_mask]
                    filter_method = 'surface-enclosed'
                    print(f"    Surface mesh: {len(pv_faces)} faces → "
                          f"{inside_mask.sum()}/{len(simp)} tets inside")
                else:
                    raise ValueError("Too few valid surface faces")

            except Exception as e:
                # --- Fallback: IQR on max-edge length ---
                print(f"    ⚠️ Surface test failed ({e}), using edge-length filter")
                edge_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
                tet_pts = combined_pts[simp]
                max_edges = np.zeros(len(simp))
                for i, j in edge_pairs:
                    el = np.linalg.norm(tet_pts[:, i] - tet_pts[:, j], axis=1)
                    max_edges = np.maximum(max_edges, el)
                # Use combined-point spacing for reference
                comb_tree = KDTree(combined_pts)
                comb_nn, _ = comb_tree.query(combined_pts, k=2)
                med_spacing = np.median(comb_nn[:, 1])
                q3 = np.percentile(max_edges, 75)
                iqr = q3 - np.percentile(max_edges, 25)
                alpha_threshold = min(q3 + 1.5 * iqr, 4.0 * med_spacing)
                valid_mask = max_edges < alpha_threshold
                filtered_tets = simp[valid_mask]
                filter_method = 'edge-IQR'
                if len(filtered_tets) < 100:
                    alpha_threshold = np.percentile(max_edges, 95)
                    filtered_tets = simp[max_edges < alpha_threshold]

            tet_nodes = combined_pts
            tet_list = [tuple(row) for row in filtered_tets]

            # 9. Map predicted materials to ALL nodes
            pred_tree = KDTree(pred_real)
            _, nn_idx = pred_tree.query(tet_nodes)
            node_ex = mat_ex[np.clip(nn_idx, 0, len(mat_ex)-1)]

            # Tet quality: aspect ratio check
            edge_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
            kept_pts = combined_pts[filtered_tets]  # (N_kept, 4, 3)
            aspect_ratios = []
            for ti in range(min(len(filtered_tets), 2000)):
                p = kept_pts[ti]
                edges = [np.linalg.norm(p[i]-p[j]) for i,j in edge_pairs]
                max_e, min_e = max(edges), min(edges)
                aspect_ratios.append(max_e / max(min_e, 1e-10))
            ar = np.array(aspect_ratios)

            pct_kept = len(filtered_tets) / len(simp) * 100
            print(f"    ✅ {filter_method}: {len(tet_list)} tets "
                  f"(kept {pct_kept:.0f}% of {len(simp)})")
            print(f"    Nodes: {len(tet_nodes)} ({n_surf} surface + {len(pred_real)} predicted)")
            print(f"    Aspect ratio: mean={ar.mean():.1f}, p95={np.percentile(ar,95):.1f}")
            print(f"    Material range: {node_ex.min():.0f} - {node_ex.max():.0f} MPa")

            # 10. Write CDB
            out_path = os.path.join(output_dir, f'v4_{name}')
            write_cdb(out_path, tet_nodes, tet_list, node_ex)
            print(f"    ✅ Exported: {out_path}")
            success += 1

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE: {success} success, {failed} failed")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


# ── AUTO-RUN ──
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--output', default=None)
    args, _ = parser.parse_known_args()
    run_export(args.model, args.data, args.output)
else:
    run_export()
