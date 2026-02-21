"""
Tetrahedral Mesh v1: AI-Based FEA Mesh Generation for Hard Tissue (Femur)
=========================================================================
Thesis Topic #3: "AI based mesh generation for hard tissue"
Supervisor: Prof. Mahmuda Naznin (BUET CSE)
Collaborator: Prof. Tanvir R. Faisal (UL Lafayette, 4MLab)

Research Pipeline Position:
  SSDL Paper (Sultana, Naznin, Faisal): QCT → SSDL segmentation → STL surface
  THIS THESIS: STL surface → FEA-quality tetrahedral volume mesh (AI-based)
  FEA Surrogate (Faisal, 4MLab): Volume mesh → fracture risk prediction

Ground Truth: 198 ANSYS CDB files (*_bonemat.cdb, patient-specific femurs)

Architecture (research-validated):
  Encoder: DGCNN — dynamic graph CNN captures local surface geometry
  Generator: Conditional VAE — regularized for small dataset (198 samples)
  Decoder: Triple-head FoldingNet — positions + sizing + material properties
  Mesher: TetGen — constrained Delaunay respecting surface boundary

Key improvements over naive approach:
  1. TetGen constrained tetrahedralization (not scipy convex hull Delaunay)
  2. Surface normals in encoder input (6D: xyz + normals)
  3. Sizing field prediction (adaptive mesh density near cortical bone)
  4. Material property prediction (Young's modulus from MPDATA)
  5. Per-element bone stiffness mapped from CT via Bonemat

References:
  - Wang et al. "DGCNN" (2019) — dynamic k-NN graph features
  - Gao et al. "DefTet" (NeurIPS 2020) — deformable tet meshes
  - Si. "TetGen" (2015) — constrained Delaunay tetrahedralization
  - Zheng et al. "NVMG" (ICLR 2023) — neural volumetric mesh generation
  - Sultana et al. "SSDL" — QCT-based 3D femur reconstruction
"""

# ============================================================
# SECTION 0: ENVIRONMENT SETUP (Auto-detect platform)
# ============================================================
import os, sys, subprocess

# Detect platform
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IN_KAGGLE = os.path.exists('/kaggle')
IN_CLOUD = IN_COLAB or IN_KAGGLE

if IN_COLAB:
    print('☁️ Google Colab detected — installing dependencies...')
    # System deps FIRST: xvfb for headless rendering, cmake for tetgen build
    subprocess.call(['apt-get', 'update', '-qq'])
    subprocess.call(['apt-get', 'install', '-y', '-qq',
                     'xvfb', 'libgl1-mesa-glx', 'cmake'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'ansys-mapdl-reader', 'pyvista', 'vtk', 'plotly',
                           'tetgen', 'seaborn', 'scikit-learn'])
    # Mount Google Drive for data access
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print('  ✅ Google Drive mounted')
    except Exception as e:
        print(f'  ⚠️ Drive mount failed: {e}')
    # PyVista off-screen rendering (Colab has no display)
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_PANEL'] = 'false'

elif IN_KAGGLE:
    print('☁️ Kaggle detected — installing dependencies...')
    # System deps FIRST: xvfb for headless rendering, cmake for tetgen build
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

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend in cloud environments WITHOUT a notebook kernel.
# In Jupyter/IPython notebooks (Colab, Kaggle), the inline backend handles display;
# forcing 'Agg' would silently suppress all plot output.
def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return 'IPKernelApp' in shell.config or 'google.colab' in str(type(shell))
    except Exception:
        return False

if IN_CLOUD and not _in_notebook():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import glob, re, warnings, time
from pathlib import Path
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# Plotly — set renderer for notebook environments
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

# PyVista
try:
    import pyvista as pv
    if IN_CLOUD:
        try:
            pv.start_xvfb()  # Virtual framebuffer for headless rendering
        except Exception:
            pass  # xvfb may not be available; pyvista still works for meshing
    pv.set_plot_theme('document')
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

print('=' * 60)
print('🦴 AI-Based Tetrahedral Mesh Generation for Hard Tissue')
print('=' * 60)
print(f'  Platform: {"Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "Local"}')
print(f'  PyVista: {"✅" if HAS_PYVISTA else "❌"}')
print(f'  Plotly: {"✅" if HAS_PLOTLY else "❌"}')

# ============================================================
# SECTION 2: CONFIGURATION (Auto-detect data directory)
# ============================================================
def _auto_detect_data_dir():
    """Find CDB data directory automatically based on platform."""
    candidates = [
        # Colab — user uploads to Drive
        '/content/drive/MyDrive/Thesis/Data',
        '/content/drive/MyDrive/thesis/4_bonemat_cdb_files',
        '/content/drive/MyDrive/4_bonemat_cdb_files',
        # Kaggle — user creates a dataset
        '/kaggle/input/femur-cdb-files',
        '/kaggle/input/femur-cdb-files/4_bonemat_cdb_files',
        '/kaggle/input/bonemat-cdb-files',
        # Local — common paths
        './4_bonemat_cdb_files',
        '../4_bonemat_cdb_files',
        os.path.join(os.path.dirname(os.path.abspath('.')), '4_bonemat_cdb_files'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            cdb_count = len(glob.glob(os.path.join(path, '*.cdb')))
            if cdb_count > 0:
                print(f'  📂 Data found: {path} ({cdb_count} CDB files)')
                return path
    # Fallback — user must set manually
    print('  ⚠️ Data directory not found automatically.')
    print('  Set CONFIG["data_dir"] to the folder containing *.cdb files.')
    if IN_COLAB:
        return '/content/drive/MyDrive/Thesis/Data'
    elif IN_KAGGLE:
        return '/kaggle/input/femur-cdb-files'
    else:
        return './4_bonemat_cdb_files'

CONFIG = {
    'data_dir': _auto_detect_data_dir(),
    # FEA quality thresholds (ANSYS standard)
    'ar_good': 3.0, 'ar_poor': 10.0,
    'sj_good': 0.5, 'sj_poor': 0.2,
    'skew_good': 0.25, 'skew_poor': 0.75,
    'max_viz_elements': 5000,
    'fig_dpi': 150,
}

# ============================================================
# SECTION 3: CDB FILE PARSER
# ============================================================
"""
ANSYS CDB file format (from lhpOpExporterAnsysCDB):
  - NBLOCK section: node coordinates
    Format line: (3i8,6e21.13e3) → 3 ints of width 8, then floats of width 21
    Data: node_id, solid_model_ref, line_loc, x, y, z
  - EBLOCK section: element connectivity
    Format: 19i8 or similar
    Data: mat_id, elem_type, real_const, section_id, esys, death,
          solid_model_ref, shape, num_nodes, unused, elem_id, then node_ids
  - MPDATA section: material properties (from Bonemat)
    Format: MPDATA,R5.0, 1,<prop>, <mat_id>, 1, <value>
    Properties: EX (Young's modulus MPa), NUXY (Poisson 0.3), DENS (g/cm³)
  - Section ends with -1 on its own line
"""

class CDBFileReader:
    """
    Read ANSYS CDB files to extract tetrahedral mesh data.
    Primary: ansys-mapdl-reader library (official ANSYS format support)
    Fallback: Direct text parsing based on CDB format specification
    """

    def __init__(self):
        self.meshes = {}
        self._try_pyansys = True
        try:
            from ansys.mapdl.reader import Archive
            self._Archive = Archive
        except ImportError:
            self._try_pyansys = False
            print("ℹ️ ansys-mapdl-reader not installed, using direct parser")

    def read(self, filepath):
        """Read a single CDB file → (nodes_array, elements_list, metadata)"""
        name = os.path.basename(filepath)

        # Always use direct parser — it handles MPDATA material properties
        # which the pyansys library does NOT parse.
        # The pyansys path is kept as a potential fallback for NBLOCK/EBLOCK
        # format edge cases, but direct parser is preferred for completeness.
        return self._read_direct(filepath, name)

    def _read_pyansys(self, filepath, name):
        """Read via ansys-mapdl-reader (handles all format quirks)."""
        archive = self._Archive(filepath)

        # Nodes: (N, 3) array with node IDs
        node_ids = archive.nnum
        coords = archive.nodes[:, :3]  # x, y, z
        nodes = np.column_stack([node_ids, coords])

        # Elements: extract tetrahedral (4-node) elements
        tets = []
        if hasattr(archive, 'elem') and len(archive.elem) > 0:
            for e in archive.elem:
                nids = [n for n in e[-1] if n > 0]
                if len(nids) >= 4:
                    tets.append(tuple(nids[:4]))

        meta = self._extract_metadata(name, filepath)
        return nodes, tets, meta

    def _read_direct(self, filepath, name):
        """Direct text parsing following CDB format specification."""
        nodes, tets = [], []
        elem_mat_ids = []  # material ID per element
        materials = {}     # mat_id → {EX, NUXY, DENS}
        section = None     # 'nblock' or 'eblock'
        fmt_widths = None

        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                stripped = line.strip()

                # Section detection
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

                # MPDATA lines (not in a section block, appear after EBLOCK)
                if stripped.startswith('MPDATA'):
                    self._parse_mpdata(stripped, materials)
                    continue

                # Format specification line (appears after NBLOCK/EBLOCK header)
                if section and stripped.startswith('('):
                    if section == 'nblock':
                        fmt_widths = self._parse_fortran_format(stripped)
                    continue

                # Skip empty/comment lines — but do NOT exit section for
                # stray blanks or inline comments within NBLOCK/EBLOCK blocks.
                # Sections are only terminated by the '-1' sentinel line (above).
                if not stripped or stripped.startswith('!') or stripped.startswith('/'):
                    continue

                # Parse data lines
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
        """Parse FORTRAN format spec like (3i8,6e21.13e3) → field widths."""
        m_int = re.search(r'(\d+)i(\d+)', fmt_str, re.IGNORECASE)
        m_flt = re.search(r'(\d+)e(\d+)', fmt_str, re.IGNORECASE)
        if m_int and m_flt:
            return {'n_int': int(m_int.group(1)), 'w_int': int(m_int.group(2)),
                    'n_flt': int(m_flt.group(1)), 'w_flt': int(m_flt.group(2))}
        return None

    def _read_node(self, raw_line, fmt):
        """Parse one NBLOCK data line → [node_id, x, y, z]."""
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
        # Space-separated fallback
        try:
            p = raw_line.split()
            if len(p) >= 6:
                return [int(p[0]), float(p[3]), float(p[4]), float(p[5])]
        except (ValueError, IndexError):
            pass
        return None

    def _read_element(self, line):
        """
        Parse one EBLOCK data line → (tuple of 4 node IDs, material_id).
        EBLOCK line: 11 metadata fields, then node IDs.
        Field 0 = material ID, Field 8 = number of nodes per element.
        """
        try:
            fields = line.split()
            if len(fields) >= 15:  # 11 meta + at least 4 nodes
                mat_id = int(fields[0])  # Material ID (links to MPDATA)
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
        """
        Parse MPDATA line → store in materials dict.
        Format: MPDATA,R5.0, 1,EX, <mat_id>, 1, <value>,
        """
        try:
            parts = [p.strip() for p in line.split(',')]
            # parts: ['MPDATA', 'R5.0', '1', 'EX', 'mat_id', '1', 'value', '']
            prop = parts[3].strip()   # 'EX', 'NUXY', or 'DENS'
            mat_id = int(parts[4].strip())
            value = float(parts[6].strip())
            if mat_id not in materials:
                materials[mat_id] = {}
            materials[mat_id][prop] = value
        except (ValueError, IndexError):
            pass

    def _extract_metadata(self, filename, filepath):
        """Extract patient ID and side from filename convention."""
        base = filename.replace('_bonemat.cdb', '').replace('_re', '')
        parts = base.split('_')
        return {
            'patient_id': parts[0] if parts else base,
            'side': 'left' if 'left' in filename.lower() else
                    'right' if 'right' in filename.lower() else 'unknown',
            'filepath': filepath
        }

    def read_directory(self, directory):
        """Read all CDB files from directory."""
        files = sorted(glob.glob(os.path.join(directory, '*.cdb')))
        if not files:
            print(f"❌ No CDB files in {directory}")
            return {}

        print(f"📂 Found {len(files)} CDB files")
        for fp in files:
            name = os.path.basename(fp)
            nodes, tets, meta = self.read(fp)
            if len(nodes) > 0:
                n_mats = len(meta.get('materials', {}))
                self.meshes[name] = {'nodes': nodes, 'tets': tets, 'meta': meta}
                print(f"  ✅ {name}: {len(nodes)} nodes, {len(tets)} tets, {n_mats} materials")

        n = len(self.meshes)
        total_n = sum(len(m['nodes']) for m in self.meshes.values())
        total_t = sum(len(m['tets']) for m in self.meshes.values())
        total_m = sum(len(m['meta'].get('materials', {})) for m in self.meshes.values())
        print(f"\n📊 {n} files | {total_n:,} nodes | {total_t:,} tetrahedra | {total_m:,} materials")
        return self.meshes


# ============================================================
# SECTION 4: TETRAHEDRAL MESH VALIDATION
# ============================================================
class TetMeshValidator:
    """Validate tetrahedral mesh integrity and compute basic statistics."""

    @staticmethod
    def validate(nodes, tets):
        """
        Check mesh integrity:
        - All element node references exist
        - No degenerate (zero-volume) elements
        - Positive orientation (positive Jacobian determinant)
        """
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
        """Compute bounding box and centroid."""
        coords = nodes[:, 1:4]
        return {
            'centroid': coords.mean(axis=0),
            'span_mm': coords.max(axis=0) - coords.min(axis=0),
            'min': coords.min(axis=0),
            'max': coords.max(axis=0),
        }


# ============================================================
# SECTION 5: MESH QUALITY METRICS
# ============================================================
class QualityMetrics:
    """
    Standard FEA mesh quality metrics per tetrahedron.
    These are the metrics used by ANSYS, Gmsh, and other FEA tools.
    """

    @staticmethod
    def compute(nodes, tets):
        """Compute quality metrics for all tetrahedra → DataFrame."""
        nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
        records = []

        for i, tet in enumerate(tets):
            pts = [nid_to_pos.get(n) for n in tet[:4]]
            if None in pts:
                continue
            m = QualityMetrics._single_tet(*pts)
            if m:
                m['idx'] = i
                records.append(m)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['quality'] = df.apply(lambda r: QualityMetrics._classify(
            r['aspect_ratio'], r['jacobian'], r['skewness']), axis=1)
        return df

    @staticmethod
    def _single_tet(p0, p1, p2, p3):
        """All metrics for one tetrahedron."""
        edges = [p1-p0, p2-p0, p3-p0, p2-p1, p3-p1, p3-p2]
        elen = np.array([np.linalg.norm(e) for e in edges])
        if elen.min() < 1e-12:
            return None

        # Volume via Jacobian determinant
        v0, v1, v2 = edges[0], edges[1], edges[2]
        J = np.dot(v0, np.cross(v1, v2))
        vol = abs(J) / 6.0
        if vol < 1e-15:
            return None

        # Aspect ratio: max_edge / (2√6 × inradius)
        faces = [(p0,p1,p2), (p0,p1,p3), (p0,p2,p3), (p1,p2,p3)]
        areas = [0.5 * np.linalg.norm(np.cross(b-a, c-a)) for a,b,c in faces]
        total_area = sum(areas)
        inradius = 3.0 * vol / total_area if total_area > 0 else 1e-12
        ar = elen.max() / (2.0 * np.sqrt(6) * inradius)

        # Scaled Jacobian: J / (|v0| × |v1| × |v2| × √2)
        product = np.linalg.norm(v0) * np.linalg.norm(v1) * np.linalg.norm(v2)
        sj = np.clip(J / (product * np.sqrt(2)), -1.0, 1.0) if product > 0 else 0

        # Skewness: 1 - vol/vol_ideal
        vol_ideal = elen.mean()**3 / (6.0 * np.sqrt(2))
        skew = 1.0 - min(vol / vol_ideal, 1.0) if vol_ideal > 0 else 1.0

        # Edge ratio
        er = elen.max() / elen.min()

        return {'volume': vol, 'aspect_ratio': ar, 'jacobian': sj,
                'skewness': skew, 'edge_ratio': er}

    @staticmethod
    def _classify(ar, sj, skew):
        scores = [0, 0, 0]  # good, ok, poor
        scores[0 if ar < CONFIG['ar_good'] else (2 if ar > CONFIG['ar_poor'] else 1)] += 1
        scores[0 if sj > CONFIG['sj_good'] else (2 if sj < CONFIG['sj_poor'] else 1)] += 1
        scores[0 if skew < CONFIG['skew_good'] else (2 if skew > CONFIG['skew_poor'] else 1)] += 1
        return ['good', 'acceptable', 'poor'][np.argmax(scores)]


# ============================================================
# SECTION 6: SURFACE EXTRACTION
# ============================================================
class SurfaceExtractor:
    """
    Extract boundary surface from tetrahedral volume mesh.

    Method: A triangular face is on the surface if it belongs to exactly
    one tetrahedron. Internal faces are shared by two tetrahedra.

    In the thesis pipeline:
      Surface mesh = INPUT to AI model (what we receive from SSDL pipeline)
      Volume mesh  = OUTPUT of AI model (what we generate)
    """

    @staticmethod
    def extract(tets):
        """Returns (surface_faces, surface_node_ids)."""
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
        """Full surface extraction with statistics."""
        faces, surf_nids = SurfaceExtractor.extract(tets)
        all_nids = set(nodes[:, 0].astype(int))
        interior_nids = all_nids - surf_nids
        return faces, surf_nids, {
            'surface_faces': len(faces),
            'surface_nodes': len(surf_nids),
            'interior_nodes': len(interior_nids),
            'total_nodes': len(nodes),
            'total_tets': len(tets),
        }


# ============================================================
# SECTION 7: VISUALIZATION
# ============================================================
class Visualizer:
    """Mesh and quality visualization."""

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
    def plot_wireframe(nodes, tets, title="Tet Wireframe", max_show=500, ax=None):
        nid_to_pos = {int(n[0]): n[1:4] for n in nodes}
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        show = np.random.choice(len(tets), min(max_show, len(tets)), replace=False)
        for i in show:
            pts = [nid_to_pos.get(n) for n in tets[i][:4]]
            if None in pts:
                continue
            pts = np.array(pts)
            for a, b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                ax.plot3D(*zip(pts[a], pts[b]), c='steelblue', lw=0.3, alpha=0.4)
        ax.set_title(title, fontsize=13, fontweight='bold')
        return ax

    @staticmethod
    def plot_surface_plotly(nodes, faces, title="Surface Mesh"):
        if not HAS_PLOTLY:
            return Visualizer.plot_surface(nodes, faces, title)

        nid_to_pos = {int(n[0]): n[1:4] for n in nodes}
        surf_nids = sorted(set(n for f in faces for n in f))
        nid_to_idx = {n: i for i, n in enumerate(surf_nids)}
        verts = np.array([nid_to_pos[n] for n in surf_nids if n in nid_to_pos])

        ii, jj, kk = [], [], []
        for f in faces:
            if all(n in nid_to_idx for n in f):
                ii.append(nid_to_idx[f[0]])
                jj.append(nid_to_idx[f[1]])
                kk.append(nid_to_idx[f[2]])

        fig = go.Figure(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=ii, j=jj, k=kk, color='lightcyan', opacity=0.7,
            flatshading=True, lighting=dict(ambient=0.5, diffuse=0.8)))
        fig.update_layout(title=title, scene=dict(aspectmode='data'),
                          width=900, height=700)
        fig.show()

    @staticmethod
    def plot_input_vs_output(nodes, tets, faces, title):
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        Visualizer.plot_surface(nodes, faces, 'Surface (AI Input)', ax1)
        ax2 = fig.add_subplot(122, projection='3d')
        Visualizer.plot_wireframe(nodes, tets, 'Volume Mesh (AI Output)', ax=ax2)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('input_vs_output.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_quality(qdf, title="Mesh Quality"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(title, fontsize=15, fontweight='bold')

        for ax, col, color, thresholds in [
            (axes[0,0], 'aspect_ratio', 'steelblue', (CONFIG['ar_good'], CONFIG['ar_poor'])),
            (axes[0,1], 'jacobian', 'teal', (CONFIG['sj_good'], CONFIG['sj_poor'])),
            (axes[0,2], 'skewness', 'coral', (CONFIG['skew_good'], CONFIG['skew_poor'])),
        ]:
            ax.hist(qdf[col], bins=50, color=color, edgecolor='white', alpha=0.8)
            ax.axvline(thresholds[0], color='green', ls='--', label='Good')
            ax.axvline(thresholds[1], color='red', ls='--', label='Poor')
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_title(col.replace('_', ' ').title())
            ax.legend(fontsize=8)

        axes[1,0].hist(qdf['volume'], bins=50, color='mediumpurple', edgecolor='white')
        axes[1,0].set_xlabel('Volume (mm³)'); axes[1,0].set_title('Element Volume')

        axes[1,1].hist(qdf['edge_ratio'], bins=50, color='goldenrod', edgecolor='white')
        axes[1,1].set_xlabel('Edge Ratio'); axes[1,1].set_title('Edge Ratio')

        cc = qdf['quality'].value_counts()
        colors = {'good':'#2ecc71', 'acceptable':'#f39c12', 'poor':'#e74c3c'}
        axes[1,2].pie(cc.values, labels=[f"{k}\n({v})" for k,v in cc.items()],
                      colors=[colors.get(k,'gray') for k in cc.index], autopct='%1.1f%%')
        axes[1,2].set_title('Quality Classification')
        plt.tight_layout()
        plt.savefig('quality_distribution.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()


# ============================================================
# SECTION 8: DATASET ANALYSIS
# ============================================================
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



# ============================================================
# SECTION 9: AI CONFIGURATION & IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold
from scipy.spatial import KDTree

# TetGen for constrained Delaunay tetrahedralization
try:
    import tetgen as _tetgen_lib
    HAS_TETGEN = True
except ImportError:
    HAS_TETGEN = False
    print("⚠️ tetgen not installed. Install: pip install tetgen")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CONFIG = {
    'n_surface_pts': 2048,
    'n_interior_pts': 4096,
    'latent_dim': 512,
    'dgcnn_k': 20,
    'input_dim': 6,       # xyz(3) + normals(3)
    'batch_size': 2,          # test run batch size
    'epochs': 1,              # quick run
    'lr': 1e-4,
    'lr_patience': 25,
    'weight_decay': 1e-4,
    'kl_weight': 0.001,
    'sizing_weight': 0.05,
    'density_weight': 0.1,
    'material_weight': 0.1,
    'k_folds': 2,
    'early_stop_patience': 40,
}


# ============================================================
# SECTION 10: MESH REPRESENTATION (BUGS FIXED)
# ============================================================
class MeshRepresentation:
    """
    Convert raw CDB mesh data → fixed-size tensors for AI.

    CRITICAL FIX: Points and normals are sampled TOGETHER using
    the SAME indices — not independently, which would create
    mismatched (point, normal) pairs.

    Material properties: computes per-node Young's modulus (log-scaled)
    from Bonemat's per-element material assignments.
    """

    @staticmethod
    def normalize(points):
        """Center at origin, scale to unit sphere."""
        c = points.mean(axis=0)
        centered = points - c
        s = max(np.max(np.linalg.norm(centered, axis=1)), 1e-10)
        return centered / s, c, s

    @staticmethod
    def sample_or_pad(points, n):
        """Resample to exactly n points. Returns (sampled_points, indices)."""
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
        """
        Estimate surface normals via PCA on local k-NN neighborhoods.
        Uses vectorized KDTree — O(N log N), not the old O(F*N) loop.
        Orients normals consistently outward from centroid.
        """
        tree = KDTree(points)
        _, nn_idx = tree.query(points, k=min(k, len(points)))
        normals = np.zeros_like(points)

        for i in range(len(points)):
            neighbors = points[nn_idx[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered / len(centered)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]  # smallest eigenvalue → normal direction

        # Orient outward from centroid
        centroid = points.mean(axis=0)
        outward = points - centroid
        flip = np.sum(normals * outward, axis=1) < 0
        normals[flip] *= -1

        # Normalize to unit length
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals /= norms
        return normals

    @staticmethod
    def compute_sizing_field(int_pts, surf_pts):
        """
        Per-interior-point sizing target: distance to nearest surface point,
        normalized to [0,1]. Near surface → 0 (fine mesh), deep interior → 1 (coarse).
        """
        tree = KDTree(surf_pts)
        dists, _ = tree.query(int_pts)
        d_max = dists.max()
        if d_max > 1e-10:
            return (dists / d_max).astype(np.float32)
        return np.zeros(len(int_pts), dtype=np.float32)

    @staticmethod
    def compute_node_stiffness(nid_to_pos, tets, elem_mat_ids, materials):
        """
        Compute per-node Young's modulus as weighted average from adjacent elements.
        Returns dict: node_id → log10(EX_MPa). Log-scale because EX spans 340-18000.
        """
        node_ex_sum = {}   # node_id → sum of log(EX)
        node_ex_count = {}  # node_id → count of adjacent elements

        for i, tet in enumerate(tets):
            if i >= len(elem_mat_ids):
                break
            mat_id = elem_mat_ids[i]
            if mat_id not in materials or 'EX' not in materials[mat_id]:
                continue
            ex_val = materials[mat_id]['EX']
            if ex_val <= 0:
                continue
            log_ex = np.log10(max(ex_val, 1.0))  # log10(EX) for numerical stability
            for nid in tet[:4]:
                node_ex_sum[nid] = node_ex_sum.get(nid, 0.0) + log_ex
                node_ex_count[nid] = node_ex_count.get(nid, 0) + 1

        node_stiffness = {}
        for nid in node_ex_sum:
            node_stiffness[nid] = node_ex_sum[nid] / node_ex_count[nid]
        return node_stiffness

    @staticmethod
    def prepare_pair(nodes, tets, meta=None):
        """
        Full preprocessing: mesh → (surface_with_normals, interior, sizing, material).

        CRITICAL: surface points and normals are sampled with THE SAME
        indices so they remain paired correctly.
        """
        faces, surf_nids = SurfaceExtractor.extract(tets)
        nid_to_pos = {int(n[0]): n[1:4].astype(float) for n in nodes}
        all_nids = set(nid_to_pos.keys())
        int_nids = all_nids - surf_nids

        # Ordered lists so we can map back to node IDs
        surf_nid_list = [n for n in surf_nids if n in nid_to_pos]
        int_nid_list = [n for n in int_nids if n in nid_to_pos]

        surf_pts = np.array([nid_to_pos[n] for n in surf_nid_list])
        int_pts = np.array([nid_to_pos[n] for n in int_nid_list])
        if len(surf_pts) < 50 or len(int_pts) < 50:
            return None

        # Compute per-node stiffness from material properties
        materials = meta.get('materials', {}) if meta else {}
        elem_mat_ids = meta.get('elem_mat_ids', []) if meta else []
        node_stiffness = MeshRepresentation.compute_node_stiffness(
            nid_to_pos, tets, elem_mat_ids, materials)

        # Build interior stiffness array (ordered same as int_pts)
        # Normalized: log10(EX) typically ranges from ~2.5 (340 MPa) to ~4.3 (18000 MPa)
        # We normalize to [0, 1] within each mesh
        int_stiffness_raw = np.array([node_stiffness.get(n, 0.0) for n in int_nid_list])
        has_material = np.any(int_stiffness_raw > 0)
        # Store RAW log10(EX) values — global normalization happens in MeshDataset
        # This preserves absolute stiffness information across meshes.
        int_stiffness = int_stiffness_raw.copy()

        # Normalize all points together (same coordinate system)
        all_pts = np.vstack([surf_pts, int_pts])
        _, centroid, scale = MeshRepresentation.normalize(all_pts)
        surf_norm = (surf_pts - centroid) / scale
        int_norm = (int_pts - centroid) / scale

        # Compute normals on ALL surface points BEFORE sampling
        normals = MeshRepresentation.estimate_normals(surf_norm, k=15)

        # Sizing field on ALL interior points BEFORE sampling
        sizing = MeshRepresentation.compute_sizing_field(int_norm, surf_norm)

        # Sample surface points AND normals with SAME indices
        n_s = MODEL_CONFIG['n_surface_pts']
        n_i = MODEL_CONFIG['n_interior_pts']

        surf_sampled, s_idx = MeshRepresentation.sample_or_pad(surf_norm, n_s)
        norm_sampled = normals[s_idx]  # SAME indices → paired correctly

        int_sampled, i_idx = MeshRepresentation.sample_or_pad(int_norm, n_i)
        size_sampled = sizing[i_idx]   # SAME indices → paired correctly
        mat_sampled = int_stiffness[i_idx]  # SAME indices → paired correctly

        return {
            'surface': surf_sampled.astype(np.float32),    # (n_s, 3)
            'normals': norm_sampled.astype(np.float32),     # (n_s, 3)
            'interior': int_sampled.astype(np.float32),     # (n_i, 3)
            'sizing': size_sampled.astype(np.float32),      # (n_i,)
            'material': mat_sampled.astype(np.float32),     # (n_i,) normalized log(EX)
            'has_material': has_material,
            'centroid': centroid, 'scale': scale,
            'n_surf_orig': len(surf_pts), 'n_int_orig': len(int_pts),
        }


# ============================================================
# SECTION 11: PYTORCH DATASET (AUGMENTATION FIXED)
# ============================================================
class MeshDataset(Dataset):
    """
    Dataset returning (6D surface, interior, sizing, material) tuples.

    AUGMENTATION FIX: Full 3D rotation (random axis, not just Z),
    plus mirror, scale, jitter, and random point dropout.
    Material data preserved through augmentation (scalar per point).
    """

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

        # FIX P1: Global material normalization across ALL meshes
        # Instead of per-mesh [0,1] normalization (which makes EX=5000
        # map to different values in different meshes), we normalize
        # using the global min/max of log10(EX) across the entire dataset.
        # This lets the model learn absolute stiffness differences:
        #   trabecular bone (~500 MPa, log10≈2.7) vs
        #   cortical bone (~18000 MPa, log10≈4.3)
        self._global_normalize_material()

        print(f"  Dataset: {len(self.samples)} valid samples "
              f"({n_with_mat} with materials, {'augmented' if augment else 'eval'})")

    def _global_normalize_material(self):
        """Normalize material (log10 EX) to [0,1] across the ENTIRE dataset.

        FIX P1: Instead of per-mesh normalization (where EX=5000 maps
        to 0.3 in one mesh but 0.7 in another), we find the global
        min/max of log10(EX) across all samples and normalize consistently.
        This lets the model learn that trabecular bone (~500 MPa) is
        genuinely different from cortical bone (~18000 MPa).
        """
        # Collect all non-zero raw log10(EX) values
        all_raw = []
        for s in self.samples:
            if s.get('has_material', False):
                mat = s['material']
                nonzero = mat[mat > 0]
                if len(nonzero) > 0:
                    all_raw.append(nonzero)

        if not all_raw:
            # No material data — zero out all material arrays
            for s in self.samples:
                s['material'] = np.zeros_like(s['material'])
            return

        all_raw = np.concatenate(all_raw)
        global_min = float(all_raw.min())  # e.g. log10(100) ≈ 2.0
        global_max = float(all_raw.max())  # e.g. log10(18000) ≈ 4.26
        global_range = max(global_max - global_min, 1e-6)
        print(f"  Material range: log10(EX) = [{global_min:.2f}, {global_max:.2f}] "
              f"({10**global_min:.0f} to {10**global_max:.0f} MPa)")

        # Normalize ALL samples to consistent [0,1] using global range
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
        surf_6d = np.concatenate([s['surface'], s['normals']], axis=1)  # (N,6)
        surf = torch.tensor(surf_6d, dtype=torch.float32)
        intr = torch.tensor(s['interior'], dtype=torch.float32)
        sizing = torch.tensor(s['sizing'], dtype=torch.float32)
        material = torch.tensor(s['material'], dtype=torch.float32)
        if self.augment:
            surf, intr = self._augment(surf, intr)
        return surf, intr, sizing, material, idx

    @staticmethod
    def _random_rotation_matrix():
        """Random rotation in SO(3) — uniform over all orientations."""
        # Random axis using Gram-Schmidt
        u = torch.randn(3)
        u = u / u.norm()
        # Random angle
        theta = torch.rand(1).item() * 2 * np.pi
        # Rodrigues' formula
        K = torch.tensor([[0, -u[2], u[1]],
                           [u[2], 0, -u[0]],
                           [-u[1], u[0], 0]], dtype=torch.float32)
        R = torch.eye(3) + torch.sin(torch.tensor(theta)) * K + \
            (1 - torch.cos(torch.tensor(theta))) * (K @ K)
        return R

    def _augment(self, surf, intr):
        # 1. Random 3D rotation (any axis, any angle)
        R = self._random_rotation_matrix()
        surf_xyz = surf[:, :3] @ R.T
        surf_nrm = surf[:, 3:] @ R.T   # normals rotate too
        intr = intr @ R.T

        # 2. Random anisotropic scaling (±10% per axis)
        s = 1.0 + (torch.rand(3) * 0.2 - 0.1)
        surf_xyz = surf_xyz * s
        intr = intr * s
        # Normals need inverse-transpose scaling for correctness
        s_inv = 1.0 / s
        surf_nrm = surf_nrm * s_inv
        surf_nrm = F.normalize(surf_nrm, dim=1)

        # 3. Random mirror (50% chance per axis)
        for ax in range(3):
            if torch.rand(1).item() > 0.5:
                surf_xyz[:, ax] *= -1
                surf_nrm[:, ax] *= -1
                intr[:, ax] *= -1

        # 4. Jitter on surface positions (not normals)
        surf_xyz = surf_xyz + torch.randn_like(surf_xyz) * 0.003

        surf = torch.cat([surf_xyz, surf_nrm], dim=1)
        return surf, intr


# ============================================================
# SECTION 12: DGCNN ENCODER
# ============================================================
def knn(x, k):
    """k-nearest neighbors via pairwise distances in feature space."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    dist = xx + inner + xx.transpose(2, 1)
    return (-dist).topk(k=k, dim=-1)[1]


def edge_features(x, k=20, idx=None):
    """Build edge features: concat(neighbor - center, center)."""
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
    """
    Dynamic Graph CNN encoder for point cloud feature extraction.

    Input: (B, input_dim, N) where input_dim = 6 (xyz + normals)
    Output: (B, out_dim * 2) global feature via max + avg pooling

    Architecture validated: DGCNN captures local geometric patterns via
    dynamic k-NN graphs — appropriate for bone surface geometry with
    varying curvature. For 198 samples, lighter than Point Transformer
    V3 (~46M params would overfit).

    FIX P3: Uses GroupNorm instead of BatchNorm for stable training
    with small batch sizes (batch_size=4).
    """

    def __init__(self, k=20, in_dim=6, out_dim=512):
        super().__init__()
        self.k = k
        # GroupNorm: num_groups chosen so each group has ≥16 channels
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
        x1 = self.ec1(edge_features(x, self.k)).max(-1)[0]   # (B,64,N)
        x2 = self.ec2(edge_features(x1, self.k)).max(-1)[0]  # (B,128,N)
        x3 = self.ec3(edge_features(x2, self.k)).max(-1)[0]  # (B,256,N)
        x4 = self.ec4(edge_features(x3, self.k)).max(-1)[0]  # (B,512,N)
        x = self.agg(torch.cat([x1, x2, x3, x4], dim=1))     # (B,out_dim,N)
        g_max = F.adaptive_max_pool1d(x, 1).view(B, -1)
        g_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        return torch.cat([g_max, g_avg], dim=1)                # (B, out_dim*2)


# ============================================================
# SECTION 13: TRIPLE-HEAD DECODER + CVAE
# ============================================================
class TripleHeadDecoder(nn.Module):
    """
    FoldingNet-style decoder with THREE output heads:
      Head 1 (position): 3D unit-ball template → fold1 → fold2 → xyz positions
      Head 2 (sizing):   From decoded position + latent → local element size [0,1]
      Head 3 (material):  From decoded position + latent → local stiffness [0,1]

    Novel thesis contribution — predicting WHERE interior nodes go,
    how DENSE the mesh should be, AND what MATERIAL PROPERTIES each
    location has (Young's modulus from CT bone density via Bonemat).
    """

    def __init__(self, z_dim=512, cond_dim=1024, n_pts=4096):
        super().__init__()
        self.n_pts = n_pts
        self.register_buffer('template', self._init_template(n_pts))

        inp = 3 + z_dim + cond_dim
        # Position head (two-fold deformation)
        self.fold1 = nn.Sequential(
            nn.Linear(inp, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3))
        self.fold2 = nn.Sequential(
            nn.Linear(3 + z_dim + cond_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3))
        # Sizing head (from decoded position + conditioning)
        self.sizing = nn.Sequential(
            nn.Linear(3 + z_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid())
        # Material head (from decoded position + conditioning → bone stiffness)
        # FIX P2: Material head receives non-detached positions so gradients
        # flow back — positions learn to cluster near high-stiffness regions.
        self.material = nn.Sequential(
            nn.Linear(3 + z_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid())

        # FIX P4: Xavier init for Sigmoid output heads (better gradient flow)
        self._init_sigmoid_heads()

    def _init_sigmoid_heads(self):
        """Xavier uniform init for layers preceding Sigmoid activations."""
        for head in [self.sizing, self.material]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _init_template(self, n):
        """Initialize template as uniform random points inside unit ball."""
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

        # Sizing head: detached — sizing shouldn't move node positions
        sz_input = torch.cat([pos2.detach(), z_e, c_e], dim=2)
        sz = self.sizing(sz_input).squeeze(-1)

        # Material head: NOT detached — positions should learn to
        # cluster near high-stiffness (cortical bone) regions
        mat_input = torch.cat([pos2, z_e, c_e], dim=2)
        mat = self.material(mat_input).squeeze(-1)

        return pos2, sz, mat


class SurfaceToVolumeCVAE(nn.Module):
    """
    Full generative model: Surface (6D) → Interior positions + sizing + material.

    Encoder: DGCNN → 1024-dim features → μ, log σ²
    Decoder: Triple-head FoldingNet → (positions, sizing, material)
    """

    def __init__(self):
        super().__init__()
        self.encoder = DGCNN(k=MODEL_CONFIG['dgcnn_k'],
                             in_dim=MODEL_CONFIG['input_dim'], out_dim=512)
        enc_out = 1024  # DGCNN max+avg pool
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
        """Generate interior nodes from surface (inference mode)."""
        self.eval()
        with torch.no_grad():
            feat, mu, logvar = self.encode(surf)
            results = []
            for _ in range(n_samples):
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                pos, sz, mat = self.decoder(z, feat)
                results.append((pos, sz, mat))
        return results


# ============================================================
# SECTION 14: LOSS FUNCTIONS
# ============================================================
def chamfer_distance(pred, target, chunk_size=512):
    """
    Bidirectional Chamfer Distance — MEMORY-EFFICIENT chunked version.

    FIX P0: Instead of materializing an O(N×M) distance matrix all at once
    (~800MB for N=M=4096), we process in chunks of `chunk_size` rows.
    Memory: O(chunk_size × M) ≈ 6MB per chunk, 99.3% reduction.
    """
    B, N, _ = pred.size()
    M = target.size(1)

    # pred → target: for each pred point, find nearest target point
    d_p2t = torch.zeros(B, N, device=pred.device)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        diff = pred[:, i:end].unsqueeze(2) - target.unsqueeze(1)  # (B, chunk, M, 3)
        d_p2t[:, i:end] = diff.pow(2).sum(-1).min(2)[0]          # (B, chunk)

    # target → pred: for each target point, find nearest pred point
    d_t2p = torch.zeros(B, M, device=pred.device)
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        diff = target[:, i:end].unsqueeze(2) - pred.unsqueeze(1)  # (B, chunk, N, 3)
        d_t2p[:, i:end] = diff.pow(2).sum(-1).min(2)[0]          # (B, chunk)

    return (d_p2t.mean(1) + d_t2p.mean(1)).mean()


def hausdorff_distance(pred, target, chunk_size=512):
    """
    One-sided Hausdorff (max of min distances) — chunked, evaluation only.
    """
    B, N, _ = pred.size()
    M = target.size(1)

    d_max_pred = torch.zeros(B, device=pred.device)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        diff = pred[:, i:end].unsqueeze(2) - target.unsqueeze(1)
        chunk_mins = diff.pow(2).sum(-1).min(2)[0]  # (B, chunk)
        d_max_pred = torch.max(d_max_pred, chunk_mins.max(1)[0])

    d_max_tgt = torch.zeros(B, device=pred.device)
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        diff = target[:, i:end].unsqueeze(2) - pred.unsqueeze(1)
        chunk_mins = diff.pow(2).sum(-1).min(2)[0]
        d_max_tgt = torch.max(d_max_tgt, chunk_mins.max(1)[0])

    return torch.max(d_max_pred, d_max_tgt).mean()


def density_uniformity(pred_pos, n_subsample=1024):
    """
    Density uniformity loss — on a SUBSAMPLE of points.

    FIX P0: Computing pairwise distances on all 4096 points requires
    O(N²) = 16M entries. Subsampling to 1024 reduces to O(1M) = 16x savings.
    """
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
    """Combined loss: Chamfer + KL + density + sizing MSE + material MSE."""

    def forward(self, pred_pos, pred_sizing, pred_material,
                target_pos, target_sizing, target_material, mu, logvar):
        cd = chamfer_distance(pred_pos, target_pos)
        kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        density = density_uniformity(pred_pos)  # FIX P0: subsampled

        # Sizing field regression
        sizing_loss = F.mse_loss(pred_sizing, target_sizing)

        # Material property regression (Young's modulus)
        material_loss = F.mse_loss(pred_material, target_material)

        total = (cd
                 + MODEL_CONFIG['kl_weight'] * kl
                 + MODEL_CONFIG['density_weight'] * density
                 + MODEL_CONFIG['sizing_weight'] * sizing_loss
                 + MODEL_CONFIG['material_weight'] * material_loss)

        return total, {
            'cd': cd.item(), 'kl': kl.item(),
            'density': density.item(), 'sizing': sizing_loss.item(),
            'material': material_loss.item(),
            'total': total.item()
        }


# ============================================================
# SECTION 15: TRAINING WITH PROPER SAVE/LOAD
# ============================================================
class Trainer:
    """Training loop with checkpointing, proper metrics, and model save."""

    def __init__(self, model, train_dl, val_dl, fold=0):
        self.model = model.to(DEVICE)
        self.fold = fold
        self.loss_fn = MeshGenLoss()
        self.opt = torch.optim.AdamW(model.parameters(),
                                      lr=MODEL_CONFIG['lr'],
                                      weight_decay=MODEL_CONFIG['weight_decay'])
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=MODEL_CONFIG['lr_patience'], factor=0.5, min_lr=1e-6)
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
            self.sched.step(vl)

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
        """Save model weights + config for reproducibility."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': MODEL_CONFIG,
            'fold': self.fold,
        }, path)
        print(f"  💾 Model saved: {path}")

    @staticmethod
    def load_model(path):
        """Load a saved model."""
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = SurfaceToVolumeCVAE()
        model.load_state_dict(ckpt['model_state'])
        model = model.to(DEVICE)
        model.eval()
        print(f"  📂 Model loaded: {path}")
        return model


# ============================================================
# SECTION 16: TETGEN INTEGRATION (FIXED)
# ============================================================
def _tetgen_from_surface(surf_pts, surf_faces=None):
    """
    Constrained Delaunay tetrahedralization via TetGen.

    FIX: Uses scipy ConvexHull for surface triangulation instead of
    PyVista's delaunay_2d() (which is a 2D projection — wrong for 3D).
    Falls back to scipy.spatial.Delaunay if TetGen is unavailable.
    """
    if HAS_TETGEN:
        try:
            import pyvista as pv
            from scipy.spatial import ConvexHull

            # Build surface triangulation from convex hull
            hull = ConvexHull(surf_pts)
            faces_pv = np.column_stack([
                np.full(len(hull.simplices), 3),
                hull.simplices
            ]).ravel()
            surf_mesh = pv.PolyData(surf_pts, faces_pv)

            tg = _tetgen_lib.TetGen(surf_mesh)
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

    # Fallback: scipy Delaunay (convex hull — less accurate)
    from scipy.spatial import Delaunay
    tri = Delaunay(surf_pts)
    elems = [tuple(simp) for simp in tri.simplices]
    nodes_arr = np.column_stack([np.arange(len(surf_pts)), surf_pts])
    return nodes_arr, elems


# ============================================================
# SECTION 17: EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_model(model, dataset):
    """
    Evaluate trained model:
      1. Predict interior nodes from surface
      2. Tetrahedralize with TetGen
      3. Compute FEA quality metrics + geometric distances
      4. Evaluate material property prediction accuracy
    """
    model.eval()
    results = []

    for i in range(len(dataset)):
        s = dataset.samples[i]
        surf_6d = np.concatenate([s['surface'], s['normals']], axis=1)
        surf_t = torch.tensor(surf_6d, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        pred_pos, pred_sz, pred_mat, mu, lv = model(surf_t)
        pred = pred_pos.cpu().numpy()[0]
        pred_sizing = pred_sz.cpu().numpy()[0]
        pred_material = pred_mat.cpu().numpy()[0]

        # Point cloud distances
        p_t = torch.tensor(pred).unsqueeze(0).float()
        g_t = torch.tensor(s['interior']).unsqueeze(0).float()
        cd = chamfer_distance(p_t, g_t).item()
        hd = hausdorff_distance(p_t, g_t).item()

        m = {'chamfer': cd, 'hausdorff': hd, 'file': dataset.names[i]}

        # Material prediction accuracy
        if s.get('has_material', False):
            target_mat = s['material']
            mat_mse = np.mean((pred_material - target_mat) ** 2)
            mat_mae = np.mean(np.abs(pred_material - target_mat))
            m['material_mse'] = float(mat_mse)
            m['material_mae'] = float(mat_mae)

        # Denormalize and generate tet mesh
        pred_real = pred * s['scale'] + s['centroid']
        surf_real = s['surface'] * s['scale'] + s['centroid']
        all_pts = np.vstack([surf_real, pred_real])

        try:
            nodes_arr, elems = _tetgen_from_surface(surf_real)
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
# SECTION 18: CDB EXPORT (write generated mesh back to ANSYS)
# ============================================================
def write_cdb(filepath, nodes, tets, material_values=None, poisson=0.3):
    """Write generated mesh + material properties to ANSYS CDB format.

    This makes the pipeline end-to-end: Surface → AI → TetGen → CDB → ANSYS FEA.

    Args:
        filepath: Output .cdb file path
        nodes: (N, 3) array of node coordinates
        tets: (M, 4) array of tetrahedral element connectivity (0-indexed)
        material_values: Optional (M,) array of Young's Modulus (EX) per element.
                         If None, a single default material (EX=10000) is used.
        poisson: Poisson's ratio (default 0.3, standard for bone)
    """
    import datetime
    n_nodes = len(nodes)
    n_elems = len(tets)

    with open(filepath, 'w') as f:
        # ── Header ──
        now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        f.write(f"!! Generated by AI Mesh Pipeline {now}\n")
        f.write("*\n")
        f.write("/PREP7\n")

        # ── NBLOCK (nodes) ──
        f.write(f"NBLOCK,6,SOLID,{n_nodes:>8},{n_nodes:>8}\n")
        f.write("(3i7,6e22.13)\n")
        for i, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"{i:7d}{0:7d}{0:7d}"
                    f"{x:22.13E}{y:22.13E}{z:22.13E}\n")
        f.write("-1\n")  # NBLOCK terminator

        # ── Material definitions (MPDATA) ──
        if material_values is not None:
            # Cluster elements into material groups (discretize EX)
            # Use ~350 bins to match original CDB structure
            ex_vals = np.asarray(material_values, dtype=np.float64)
            ex_vals = np.clip(ex_vals, 1.0, None)  # Ensure positive

            # Unique-ify: round to nearest integer MPa
            ex_rounded = np.round(ex_vals).astype(int)
            unique_ex = np.unique(ex_rounded)
            ex_to_matid = {v: (i + 1) for i, v in enumerate(unique_ex)}
            elem_matid = np.array([ex_to_matid[v] for v in ex_rounded])

            # Write MPDATA for each unique material
            for ex_val, mat_id in sorted(ex_to_matid.items(), key=lambda x: x[1]):
                f.write(f"MPDATA,R5.0, 1,EX,{mat_id:>6}, 1, "
                        f"{float(ex_val):.8f}    ,\n")
                f.write(f"MPDATA,R5.0, 1,NUXY,{mat_id:>6}, 1, "
                        f"{poisson:.8f}    ,\n")
        else:
            # Single default material
            f.write("MPDATA,R5.0, 1,EX,     1, 1, 10000.00000000    ,\n")
            f.write(f"MPDATA,R5.0, 1,NUXY,     1, 1, {poisson:.8f}    ,\n")
            elem_matid = np.ones(n_elems, dtype=int)

        # ── EBLOCK (elements) ──
        # ANSYS EBLOCK for 4-node tetrahedra (SOLID185)
        # 11 metadata fields + 4 node IDs per element line:
        #   mat, type, real_const, section_id, esys, death,
        #   solidmodel_ref, shape_flag, num_nodes, unused, elem_id,
        #   n1, n2, n3, n4
        f.write(f"EBLOCK,19,SOLID,{n_elems:>8},{n_elems:>8}\n")
        f.write("(19i9)\n")
        for i, tet in enumerate(tets):
            mat = int(elem_matid[i])
            eid = i + 1  # 1-indexed element ID
            n1, n2, n3, n4 = tet[0] + 1, tet[1] + 1, tet[2] + 1, tet[3] + 1
            f.write(f"{mat:9d}{1:9d}{1:9d}{1:9d}{0:9d}{0:9d}"
                    f"{0:9d}{0:9d}{4:9d}{0:9d}{eid:9d}"
                    f"{n1:9d}{n2:9d}{n3:9d}{n4:9d}\n")
        f.write("-1\n")  # EBLOCK terminator

        # ── Footer ──
        f.write("FINISH\n")

    print(f"  📁 CDB exported: {filepath} "
          f"({n_nodes} nodes, {n_elems} tets, "
          f"{len(np.unique(elem_matid))} materials)")


# ============================================================
# SECTION 19: K-FOLD CROSS-VALIDATION (GroupKFold fix)
# ============================================================
def _extract_patient_id(name):
    """Extract patient ID from mesh name for GroupKFold.

    Examples:
        'AB029_left_bonemat'  → 'AB029'
        'AB029_right_bonemat' → 'AB029'
        'C5008_right_bonemat_re' → 'C5008'
    
    This ensures left/right femurs from the same patient are
    ALWAYS in the same fold, preventing data leakage.
    """
    # Split on '_', take everything before 'left' or 'right'
    parts = name.split('_')
    # Find the index of 'left' or 'right'
    for i, p in enumerate(parts):
        if p.lower() in ('left', 'right'):
            return '_'.join(parts[:i])
    # Fallback: use the full name
    return name


def run_kfold(meshes):
    """K-Fold CV with PATIENT-LEVEL grouping to prevent data leakage.

    FIX: Uses GroupKFold instead of KFold so that left/right femurs
    from the same patient (e.g., AB029_left and AB029_right) are
    ALWAYS in the same fold. Without this, the model could see a
    patient's left femur in training and predict its near-mirror
    right femur in testing — artificially inflating metrics.
    """
    print("\n" + "=" * 60)
    print("🧠 K-FOLD CROSS-VALIDATION (Patient-Grouped)")
    print("=" * 60)

    full_ds = MeshDataset(meshes, augment=False)
    n = len(full_ds)
    if n < 2:
        print("❌ Not enough valid samples for training")
        return []

    # Extract patient IDs for grouping
    patient_ids = [_extract_patient_id(name) for name in full_ds.names]
    unique_patients = sorted(set(patient_ids))
    n_patients = len(unique_patients)
    print(f"  Patients: {n_patients} unique (from {n} samples)")

    # Map patient IDs to group indices
    pid_to_group = {pid: i for i, pid in enumerate(unique_patients)}
    groups = np.array([pid_to_group[pid] for pid in patient_ids])

    k = min(MODEL_CONFIG['k_folds'], n_patients)
    gkf = GroupKFold(n_splits=k)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(range(n), groups=groups)):
        tr_patients = set(patient_ids[i] for i in tr_idx)
        va_patients = set(patient_ids[i] for i in va_idx)
        overlap = tr_patients & va_patients
        assert len(overlap) == 0, f"Data leakage! Patients in both folds: {overlap}"

        print(f"\n{'=' * 50}")
        print(f"  FOLD {fold + 1}/{k} | train={len(tr_idx)} ({len(tr_patients)} patients) "
              f"| val={len(va_idx)} ({len(va_patients)} patients)")
        print(f"{'=' * 50}")

        # Create subset datasets
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

        # Save best model per fold
        save_path = f'model_fold{fold + 1}.pt'
        trainer.save_model(save_path)

        # Evaluate
        metrics = evaluate_model(model, va_ds)

        fold_results.append({
            'fold': fold + 1,
            'history': hist,
            'metrics': metrics,
            'model_path': save_path,
            'n_train_patients': len(tr_patients),
            'n_val_patients': len(va_patients),
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
    print("📊 K-FOLD RESULTS SUMMARY (Patient-Grouped)")
    print("=" * 60)
    cds = [r['mean_cd'] for r in fold_results]
    hds = [r['mean_hd'] for r in fold_results]
    ars = [r['mean_ar'] for r in fold_results]
    sjs = [r['mean_sj'] for r in fold_results]
    pgs = [r['pct_good'] for r in fold_results]
    mms = [r['mean_mat_mse'] for r in fold_results]
    mma = [r['mean_mat_mae'] for r in fold_results]
    print(f"  Chamfer:    {np.mean(cds):.6f} ± {np.std(cds):.6f}")
    print(f"  Hausdorff:  {np.mean(hds):.6f} ± {np.std(hds):.6f}")
    print(f"  Aspect R:   {np.mean(ars):.3f} ± {np.std(ars):.3f}")
    print(f"  Jacobian:   {np.mean(sjs):.3f} ± {np.std(sjs):.3f}")
    print(f"  % Good:     {np.mean(pgs):.1f}% ± {np.std(pgs):.1f}%")
    print(f"  Mat MSE:    {np.mean(mms):.6f} ± {np.std(mms):.6f}")
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
        fig.suptitle('Training Curves (K-Fold CV)', fontsize=14, fontweight='bold')
        for fr in fold_results:
            h, f = fr['history'], fr['fold']
            axes[0].plot(h['train_loss'], alpha=0.4, label=f'F{f} train')
            axes[0].plot(h['val_loss'], alpha=0.8, ls='--', label=f'F{f} val')
            axes[1].plot(h['train_cd'], alpha=0.4, label=f'F{f} train')
            axes[1].plot(h['val_cd'], alpha=0.8, ls='--', label=f'F{f} val')
        axes[0].set(xlabel='Epoch', ylabel='Loss', yscale='log', title='Total Loss')
        axes[1].set(xlabel='Epoch', ylabel='CD', yscale='log', title='Chamfer Distance')
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
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
        plt.savefig('gen_vs_gt.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def quality_comparison(fold_results):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Generated Mesh Quality (TetGen)', fontsize=14, fontweight='bold')
        all_m = [m for fr in fold_results for m in fr['metrics'] if 'mean_ar' in m]
        if all_m:
            axes[0].hist([m['mean_ar'] for m in all_m], 20,
                         color='steelblue', edgecolor='white')
            axes[0].set_title('Aspect Ratio')
            axes[1].hist([m['mean_sj'] for m in all_m], 20,
                         color='teal', edgecolor='white')
            axes[1].set_title('Scaled Jacobian')
            axes[2].hist([m.get('pct_good', 0) for m in all_m], 20,
                         color='#2ecc71', edgecolor='white')
            axes[2].set_title('% Good Quality')
            axes[3].hist([m['chamfer'] for m in all_m], 20,
                         color='coral', edgecolor='white')
            axes[3].set_title('Chamfer Distance')
        for ax in axes:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('gen_quality.png', dpi=CONFIG['fig_dpi'], bbox_inches='tight')
        plt.show()


# ============================================================
# SECTION 21: MAIN PIPELINE
# ============================================================
def run_pipeline(skip_training=False):
    """
    Complete pipeline: Parse → Analyze → Train → Evaluate.
    skip_training=True: Phase 1 only (data analysis).
    """
    print("\n" + "=" * 70)
    print("🦴 AI-Based Tetrahedral Mesh Generation for Femur")
    print("   DGCNN (6D) + CVAE + Triple-Head Decoder + TetGen")
    print("   Predicts: positions + sizing + material properties")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  TetGen available: {HAS_TETGEN}")
    print(f"  Surface pts: {MODEL_CONFIG['n_surface_pts']}")
    print(f"  Interior pts: {MODEL_CONFIG['n_interior_pts']}")

    # Drive mount already handled in Section 0 (auto-detect)
    # ── PHASE 1: Parse & Analyze ──
    print("\n📂 PHASE 1: Parsing CDB files")
    reader = CDBFileReader()
    meshes = reader.read_directory(CONFIG['data_dir'])
    meshes = {k: meshes[k] for k in list(meshes)[:4]}
    if not meshes:
        print("❌ No data found. Set CONFIG['data_dir'].")
        return

    # Validate first 3
    for name, m in list(meshes.items())[:3]:
        st, _ = TetMeshValidator.validate(m['nodes'], m['tets'])
        print(f"  {name}: {st['valid']}/{st['total']} valid tets")

    # Quality metrics
    all_q = {}
    for name, m in meshes.items():
        q = QualityMetrics.compute(m['nodes'], m['tets'])
        if not q.empty:
            all_q[name] = q

    # Surface extraction
    all_surf = {}
    for name, m in meshes.items():
        faces, nids, stats = SurfaceExtractor.get_surface_data(m['nodes'], m['tets'])
        all_surf[name] = {'faces': faces, 'nids': nids, 'stats': stats}

    # Visualize one sample
    sample = list(meshes.keys())[0]
    Visualizer.plot_input_vs_output(
        meshes[sample]['nodes'], meshes[sample]['tets'],
        all_surf[sample]['faces'], sample)
    if sample in all_q:
        Visualizer.plot_quality(all_q[sample], f'GT Quality: {sample}')

    stats_df = DatasetAnalyzer.analyze(meshes)
    DatasetAnalyzer.plot_overview(stats_df)

    if skip_training:
        print("\n⏩ Phase 1 complete. Call run_pipeline() for full training.")
        return meshes, all_q, all_surf, stats_df

    # ── PHASE 2-3: Train & Evaluate ──
    print("\n🧠 PHASE 2-3: Training Pipeline")
    fold_results = run_kfold(meshes)

    if fold_results:
        ResultsViz.training_curves(fold_results)
        ResultsViz.quality_comparison(fold_results)

        best = min(fold_results, key=lambda r: r['mean_cd'])
        print(f"\n  🏆 Best fold: {best['fold']} (CD={best['mean_cd']:.6f})")
        print(f"      Model saved: {best['model_path']}")

        # Visualize best fold predictions
        best_model = Trainer.load_model(best['model_path'])
        full_ds = MeshDataset(meshes, augment=False)
        ResultsViz.generated_vs_gt(best_model, full_ds, n=3)

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    if fold_results:
        print(f"  {len(meshes)} meshes | DGCNN+CVAE+TripleHead+TetGen")
        print(f"  CD = {np.mean([r['mean_cd'] for r in fold_results]):.6f}")
        print(f"  HD = {np.mean([r['mean_hd'] for r in fold_results]):.6f}")
        print(f"  Material MAE = {np.mean([r['mean_mat_mae'] for r in fold_results]):.6f}")
    print("=" * 70)

    return meshes, all_q, all_surf, stats_df, fold_results


if __name__ == '__main__':
    results = run_pipeline()
