# Thesis Context — AI-Based Mesh Generation for Hard Tissue

> **Last updated:** 2026-02-18
> **Author:** Tamim Saad
> **Program:** B.Sc. in CSE, KUET (Khulna University of Engineering & Technology)
> **Supervisor:** Prof. Mahmuda Naznin, BUET CSE
> **Collaborator:** Prof. Tanvir R. Faisal, University of Louisiana at Lafayette (4MLab)

---

## Table of Contents

1. [Thesis Topic & Position in the Research Pipeline](#1-thesis-topic--position-in-the-research-pipeline)
2. [The Complete Research Pipeline](#2-the-complete-research-pipeline)
3. [Supervisor & Collaborator Details](#3-supervisor--collaborator-details)
4. [Dataset Description](#4-dataset-description)
5. [CDB File Format Deep Dive](#5-cdb-file-format-deep-dive)
6. [Architecture: DGCNN + CVAE + Triple-Head FoldingNet + TetGen](#6-architecture-dgcnn--cvae--triple-head-foldingnet--tetgen)
7. [The AI Learning Task (Formally)](#7-the-ai-learning-task-formally)
8. [Component SOTA Validation (2024-2025)](#8-component-sota-validation-2024-2025)
9. [Implementation History & Fixes](#9-implementation-history--fixes)
10. [Known Issues & Remaining Work](#10-known-issues--remaining-work)
11. [Key References](#11-key-references)
12. [File Structure](#12-file-structure)

---

## 1. Thesis Topic & Position in the Research Pipeline

### Thesis Topic

> **#3: "AI based mesh generation for hard tissue"**
> — from Prof. Mahmuda Naznin's list of thesis topics (Research Area: AI, Deep Learning, Medical Imaging)

### What This Means

The thesis builds a **deep learning model** that takes a bone's 3D surface geometry and generates:

1. **Interior volume node positions** (filling the bone's interior)
2. **Local element sizing field** (controlling mesh density)
3. **Material properties** (Young's modulus per node — bone stiffness)
4. A final **tetrahedral volume mesh** (via TetGen) suitable for Finite Element Analysis (FEA)

### Why This Matters

Traditional FEA mesh creation for patient-specific bone analysis requires:

- **HyperMesh** (or similar) for manual tetrahedral meshing → hours per specimen
- **Bonemat** for mapping CT-derived material properties → additional manual steps
- **Deformetrica** for statistical shape analysis → complex setup

This thesis replaces **all three** with a single AI model, reducing the process from hours to seconds.

---

## 2. The Complete Research Pipeline

The supervisor's research group is building an **end-to-end pipeline** for femur fracture risk assessment:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE RESEARCH PIPELINE                            │
│                                                                          │
│  Steps 1-3: QCT Image → Segmentation → 3D Surface Reconstruction        │
│            ✅ COMPLETED by Jamalia Sultana / Fabliha                     │
│            Published: MBEC (Springer), 2024                              │
│            Method: 3D U-Net → polynomial spline interpolation            │
│            Result: Dice Score 91.8%, Volume error 6.61%                  │
│            Output: 3D reconstructed femur surface from sparse QCT        │
│                                                                          │
│  Step 4:   3D Surface → TETRAHEDRAL VOLUME MESH + MATERIAL PROPERTIES    │
│            🔴 THIS IS THE THESIS (Tamim Saad)                            │
│            Topic: "AI based mesh generation for hard tissue"             │
│            Method: DGCNN + CVAE + Triple-Head FoldingNet + TetGen        │
│            Input: Surface mesh (from Step 3)                             │
│            Output: FEA-quality tet mesh + per-node Young's modulus        │
│            Ground truth: 198 CDB files from collaborator                 │
│                                                                          │
│  Steps 5-7: FE Mesh → FEA Simulation → Fracture Risk Prediction         │
│            ✅ COMPLETED by Rabina Awal                                   │
│            Published: Expert Systems with Applications, 2025             │
│            Method: CatBoost ML surrogate model                           │
│            Result: 76% fracture risk prediction accuracy                 │
│            Assumption: FEA-quality mesh ALREADY exists                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### The Gap This Thesis Fills

> Steps 1-3 output a **3D surface**. Steps 5-7 require a **FEA volume mesh with material properties**.
> The question was: **who converts the surface into a volume mesh?**
> Answer: **This thesis** — using AI instead of manual tools like HyperMesh + Bonemat.

---

## 3. Supervisor & Collaborator Details

### Supervisor: Prof. Mahmuda Naznin

| Detail            | Info                                                                              |
| ----------------- | --------------------------------------------------------------------------------- |
| **Position**      | Professor, Dept. of CSE, BUET (Bangladesh University of Engineering & Technology) |
| **Research Area** | AI, Deep Learning, Medical Imaging, Ubiquitous Computing                          |
| **Thesis Topics** | 8 topics listed; topic #3 = "AI based mesh generation for hard tissue"            |
| **Group Papers**  | SSDL (MBEC 2024), FEA Surrogate (Expert Systems 2025)                             |
| **Role**          | Defines the overall pipeline vision; supervises all three thesis components       |

### Collaborator: Prof. Tanvir R. Faisal

| Detail                | Info                                                                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Position**          | Assistant Professor, University of Louisiana at Lafayette                                                                                      |
| **Lab**               | 4MLab (Medical Mechanics and Machine Learning Lab)                                                                                             |
| **Prior Affiliation** | PhD/Postdoc at McGill University (verified email: @mail.mcgill.ca)                                                                             |
| **Research Focus**    | QCT-based FEA for hip fracture risk assessment                                                                                                 |
| **Key Tools**         | Bonemat v3.0, Deformetrica, ANSYS, HyperMesh                                                                                                   |
| **Key Publication**   | "Study of the significance of parameters and their interaction on assessing femoral fracture risk by quantitative statistical analysis" (2022) |
| **Role in Thesis**    | Provided the 198 CDB dataset; his manual pipeline (Bonemat + Deformetrica) is what the AI replaces                                             |

### Tanvir Faisal's Manual Pipeline (What We Replace)

```
QCT scan → Segment bone → Deformetrica (shape analysis) → HyperMesh (tet meshing)
         → Bonemat (material mapping from HU → Young's Modulus) → ANSYS CDB export
```

Our AI model replaces the middle 3 steps (Deformetrica + HyperMesh + Bonemat) with a single neural network.

### Related Researchers in the Group

| Researcher      | Contribution                                                   | Step       |
| --------------- | -------------------------------------------------------------- | ---------- |
| Jamalia Sultana | SSDL — semi-supervised 3D femur reconstruction from sparse QCT | Steps 1-3  |
| Fabliha         | Co-contributor to SSDL paper                                   | Steps 1-3  |
| Rabina Awal     | FEA surrogate model for fracture risk prediction               | Steps 5-7  |
| **Tamim Saad**  | **AI-based mesh generation (this thesis)**                     | **Step 4** |

---

## 4. Dataset Description

### Overview

| Property               | Value                                                    |
| ---------------------- | -------------------------------------------------------- | -------------------- |
| **Total Files**        | 198 CDB files                                            |
| **Location**           | `4_bonemat_cdb_files/`                                   |
| **Naming**             | `{PatientID}\_{left                                      | right}\_bonemat.cdb` |
| **Unique Patients**    | ~100 (left + right femur per patient)                    |
| **Re-meshed Variants** | 3 files with `_re.cdb` suffix (AB029, C5008, CV041)      |
| **File Size**          | ~7-17 MB each (avg ~10 MB)                               |
| **Generator**          | `lhpOpExporterAnsysCDB` (Living Human Project framework) |
| **Generation Date**    | February 2023                                            |
| **Source**             | Prof. Tanvir R. Faisal's lab                             |

### Per-File Contents

| Block            | Content                                               | Typical Count            |
| ---------------- | ----------------------------------------------------- | ------------------------ |
| **NBLOCK**       | Node coordinates (x, y, z)                            | ~16,000–21,000 nodes     |
| **EBLOCK**       | Tetrahedral element connectivity (4 node IDs per tet) | ~80,000–120,000 elements |
| **MPDATA(EX)**   | Young's Modulus per material ID (MPa)                 | ~350 unique materials    |
| **MPDATA(NUXY)** | Poisson's Ratio (constant 0.3)                        | ~350 entries             |
| **MPDATA(DENS)** | Bone density (g/cm³)                                  | ~350 entries             |

### Bonemat Configuration (from `newMAFfile.xml`)

```xml
DensitySelection="Mean"
PoissonRatio="0.3"
RhoUsage="rhoQCT"
```

This means:

- **Mean density** used per element (averaged from CT voxels within element)
- **Constant Poisson's ratio** of 0.3 (standard for bone)
- **QCT-derived density** (rhoQCT) for material property calculation

### Material Property Ranges

| Property                   | Min        | Max         | Physical Meaning               |
| -------------------------- | ---------- | ----------- | ------------------------------ |
| **EX (Young's Modulus)**   | ~100 MPa   | ~18,400 MPa | Trabecular → Cortical bone     |
| **NUXY (Poisson's Ratio)** | 0.3        | 0.3         | Constant for all elements      |
| **DENS (Density)**         | ~0.1 g/cm³ | ~1.9 g/cm³  | Spongy → Dense bone            |
| **log₁₀(EX)**              | ~2.0       | ~4.26       | What the model actually learns |

### Data Leakage Warning

> **CRITICAL:** Left and right femurs from the same patient (e.g., `AB029_left` and `AB029_right`)
> are near-mirror images. Cross-validation MUST use `GroupKFold` with patient ID as the group
> to prevent the model from seeing one side in training and the other in testing.

---

## 5. CDB File Format Deep Dive

### File Header

```
!! Generated by lhpOpExporterAnsysCDB Wed Feb  1 23:07:53 2023
*
/PREP7
```

### NBLOCK (Node Block)

```
NBLOCK,6,SOLID,   21023,    20982
(3i7,6e22.13)
      1      0      0   1.4369985961914E+02  -1.7612020874023E+02   1.2686087646484E+03
      2      0      0   1.4162065124512E+02  -1.7570407104492E+02   1.2686207275391E+03
```

- Format: `node_id, 0, 0, x, y, z` (in millimeters, using scientific notation)
- 6 fields per line, 22 chars each, 13 decimal places

### EBLOCK (Element Block)

- Contains tetrahedral element definitions
- Each element: 4 node IDs forming a tetrahedron
- Material ID assigned to each element

### MPDATA (Material Properties)

```
MPDATA,R5.0, 1,EX,     1, 1, 18382.71988488    ,
MPDATA,R5.0, 1,NUXY,     1, 1, 0.30000000    ,
MPDATA,R5.0, 1,DENS,     1, 1, 1.27029542    ,
```

- Format: `MPDATA,version, table, property, material_id, row, value`
- **EX** = Young's Modulus (MPa) — varies per material (100–18400 MPa)
- **NUXY** = Poisson's Ratio — constant 0.3
- **DENS** = Bone density (g/cm³) — varies (0.1–1.9)
- ~350 unique material IDs per file, each with 3 properties = ~1050 MPDATA lines

### What the Code Parses

The `MeshRepresentation` class in `tetrahedral_mesh_v1.py`:

1. Reads NBLOCK → extracts node coordinates
2. Reads EBLOCK → extracts tetrahedral connectivity + material IDs per element
3. Reads MPDATA → extracts EX per material ID
4. Computes **per-node stiffness** = log₁₀(mean EX of adjacent elements)
5. Separates surface nodes (boundary faces) from interior nodes
6. Stores: surface (xyz + normals), interior (xyz), sizing field, material (log₁₀ EX)

---

## 6. Architecture: DGCNN + CVAE + Triple-Head FoldingNet + TetGen

### Pipeline Overview

```
           Surface Mesh (2048 pts × 6D)
                    ↓
          ┌─────────────────┐
          │     DGCNN        │  Encoder — captures local bone geometry
          │  (k=20 neighbors)│  via dynamic graph convolution
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │      CVAE        │  Generative model — learns a latent space
          │  (dim=512)       │  of bone interior distributions
          │  μ + log(σ²)     │  KL divergence regularization
          └────────┬────────┘
                   ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
   ┌─────────┐ ┌────────┐ ┌─────────┐
   │Position │ │Sizing  │ │Material │  Triple-Head FoldingNet Decoder
   │Head     │ │Head    │ │Head     │
   │(xyz)    │ │(σ)     │ │(EX)     │
   └────┬────┘ └───┬────┘ └────┬────┘
        ↓          ↓           ↓
   4096 interior  sizing    log₁₀(EX)
   node positions  field    per node
        ↓
   ┌─────────────────┐
   │    TetGen        │  Constrained Delaunay Tetrahedralization
   │  (traditional)   │  Converts predicted points → tet mesh
   └──────────────────┘
        ↓
   FEA-quality tetrahedral mesh
   with per-element material properties
```

### Component Details

#### Encoder: DGCNN (Dynamic Graph CNN)

| Parameter     | Value                                                |
| ------------- | ---------------------------------------------------- |
| Input         | 2048 surface points × 6D (xyz + normals)             |
| k-NN          | k=20 dynamic neighbors                               |
| Channels      | 64 → 128 → 256 → 512                                 |
| Normalization | GroupNorm (not BatchNorm — stable with batch_size=4) |
| Output        | 512D global feature per mesh                         |

Why DGCNN over alternatives:

- **PointNet**: No local structure capture
- **PointNet++**: Good but DGCNN's dynamic graph is better for bone geometry
- **Point Transformer v3**: Would overfit on 198 samples (needs >10K)

#### Generative Model: CVAE

| Parameter      | Value                                |
| -------------- | ------------------------------------ |
| Latent dim     | 512                                  |
| KL weight      | 0.001 (annealed)                     |
| Regularization | KL divergence prevents mode collapse |

Why CVAE over alternatives:

- **Flow Matching**: Too data-hungry for 198 samples
- **Diffusion Models**: Need >10K samples for stable training
- **GAN**: Mode collapse risk with small datasets
- **VAE/CVAE**: Built-in regularization, works well with small datasets

#### Decoder: Triple-Head FoldingNet

Three separate prediction heads, all sharing the latent code:

| Head         | Output                         | Activation | Gradient                                         |
| ------------ | ------------------------------ | ---------- | ------------------------------------------------ |
| **Position** | 4096 × 3 (xyz)                 | tanh       | Full backprop                                    |
| **Sizing**   | 4096 × 1 (element size)        | Sigmoid    | **Detached** from position                       |
| **Material** | 4096 × 1 (normalized log₁₀ EX) | Sigmoid    | **Not detached** — positions learn from material |

The material head is NOT detached so gradients can encourage nodes to cluster near high-stiffness (cortical bone) regions.

#### Mesher: TetGen

| Parameter | Value                                   |
| --------- | --------------------------------------- |
| Input     | Predicted surface + interior points     |
| Method    | Constrained Delaunay tetrahedralization |
| Output    | Tetrahedral mesh suitable for ANSYS FEA |

TetGen is the standard in biomechanics FEA. Neural alternatives (DefTet, FlexiCubes) are for different input types.

### Loss Functions

```python
L_total = L_chamfer + λ_kl * L_KL + λ_density * L_density + λ_sizing * L_sizing + λ_material * L_material
```

| Component          | Weight | Purpose                                         |
| ------------------ | ------ | ----------------------------------------------- |
| Chamfer Distance   | 1.0    | Geometric accuracy of predicted interior points |
| KL Divergence      | 0.001  | Latent space regularization                     |
| Density Uniformity | 0.1    | Even distribution of interior nodes             |
| Sizing MSE         | 0.05   | Element size field accuracy                     |
| Material MSE       | 0.1    | Young's modulus prediction accuracy             |

### Hyperparameters

```python
MODEL_CONFIG = {
    'n_surface_pts': 2048,      # Surface points sampled per mesh
    'n_interior_pts': 4096,     # Interior points predicted
    'latent_dim': 512,          # CVAE latent space
    'dgcnn_k': 20,              # k-NN for DGCNN
    'input_dim': 6,             # xyz(3) + normals(3)
    'batch_size': 4,            # Small — 198 samples total
    'epochs': 300,              # Training epochs
    'lr': 1e-4,                 # Learning rate
    'lr_patience': 25,          # LR scheduler patience
    'weight_decay': 1e-4,       # L2 regularization
    'kl_weight': 0.001,         # KL divergence weight
    'sizing_weight': 0.05,      # Sizing loss weight
    'density_weight': 0.1,      # Density uniformity weight
    'material_weight': 0.1,     # Material prediction weight
    'k_folds': 5,               # Cross-validation folds
    'early_stop_patience': 40,  # Early stopping patience
}
```

---

## 7. The AI Learning Task (Formally)

### Input Representation

```
Surface S = {(x_i, y_i, z_i, nx_i, ny_i, nz_i)} for i = 1..2048
```

- 2048 points sampled from the bone's boundary surface
- Each point has 3D position + estimated surface normal
- Normalized to unit sphere (zero-centered, unit scale)

### Output Representation

```
f(S) → (P, σ, E) where:
  P = {(x_j, y_j, z_j)} for j = 1..4096  — interior node positions
  σ = {s_j} for j = 1..4096               — local element sizing
  E = {e_j} for j = 1..4096               — normalized log₁₀(Young's Modulus)
```

### Training Paradigm

The model is a **conditional generative model** — it generates an interior point cloud conditioned on the surface boundary. The CVAE framework:

1. **Encodes** the surface into a global feature
2. **Samples** from a learned latent distribution
3. **Decodes** into interior positions + sizing + material

### Ground Truth

Each CDB file provides:

- Surface nodes (extracted from boundary faces of the tet mesh)
- Interior nodes (all non-surface nodes)
- Per-node sizing (estimated from neighboring element volumes)
- Per-node material (log₁₀ of mean EX from adjacent elements)

After generation, **TetGen** converts the predicted point cloud into an actual tetrahedral mesh with element connectivity.

---

## 8. Component SOTA Validation (2024-2025)

All components have been validated against the latest literature:

### Encoder: DGCNN ✅

| Evidence                                                         | Source                |
| ---------------------------------------------------------------- | --------------------- |
| DGCNN evaluated for mammalian bone 3D point cloud classification | ResearchGate 2025     |
| EdgeConv captures local geometry via dynamic k-NN graphs         | MDPI 2024 review      |
| Appropriate for small datasets (<200 samples)                    | Architecture analysis |

### Generative Model: CVAE ✅

| Evidence                                                     | Source               |
| ------------------------------------------------------------ | -------------------- |
| CLAY (SIGGRAPH 2024) uses multi-resolution VAE for 3D assets | SIGGRAPH proceedings |
| CVAE with transformer for mesh recovery from point clouds    | CVPR 2023            |
| Diffusion/Flow models need >10K samples — 198 too small      | Multiple sources     |

### Decoder: Triple-Head FoldingNet ✅

| Evidence                                                                     | Source             |
| ---------------------------------------------------------------------------- | ------------------ |
| Multi-head decoders standard in 2024 3D generation papers                    | Multiple papers    |
| FoldingNet concept (template deformation) still used in modern architectures | Confirmed          |
| Separate position/sizing/material heads = physically interpretable           | Novel contribution |

### Mesher: TetGen ✅

| Evidence                                                           | Source                             |
| ------------------------------------------------------------------ | ---------------------------------- |
| Standard in biomechanics FEA workflows                             | Bonemat docs, Tanvir Faisal's work |
| Produces constrained Delaunay tetrahedralization                   | Correct for FEA                    |
| Neural alternatives (DefTet, FlexiCubes) for different input types | Not applicable                     |

### Material Prediction ✅ (Novel!)

| Evidence                                                                                     | Source            |
| -------------------------------------------------------------------------------------------- | ----------------- |
| BPNN for Young's modulus prediction validated at 13% stress accuracy                         | ResearchGate 2025 |
| Deep learning for bone density from CT scan                                                  | NIH 2024          |
| Bonemat v4 (cortical-specific mapping) released 2024                                         | ResearchGate      |
| **No published work combines surface→volume generation with per-node material in one model** | **Novelty claim** |

### Alternative Approaches Considered

| Method                     | What It Does                                      | Why Not Used               |
| -------------------------- | ------------------------------------------------- | -------------------------- |
| MeshingNet                 | NN predicts local mesh density → guides mesher    | Doesn't predict material   |
| LAMG                       | NN maps solution estimate → adaptive sizing field | Requires solution estimate |
| Generator NN (SSRN)        | Directly generates refined meshes                 | No material prediction     |
| FeaGPT                     | AI agent automates entire FEA workflow            | Too complex, agent-based   |
| Neural Vol Mesh Gen (NVMG) | Diffusion-based voxel → tet mesh                  | Needs more data            |

---

## 9. Implementation History & Fixes

### Evolution: v5 → v1 (Current)

The project went through a major architectural shift:

| Version          | Architecture                           | Problem                                                                              |
| ---------------- | -------------------------------------- | ------------------------------------------------------------------------------------ |
| v5 (old)         | PointNet AutoEncoder on raw NBLOCK     | Only reconstructed point clouds, ignored element connectivity, material. Loss = NaN. |
| **v1 (current)** | **DGCNN + CVAE + TripleHead + TetGen** | Full pipeline: surface → interior + sizing + material → tet mesh                     |

### What v5 Got Wrong (and v1 Fixed)

| v5 Issue                             | v1 Fix                                                            |
| ------------------------------------ | ----------------------------------------------------------------- |
| Parsed only NBLOCK (node coords)     | Parses NBLOCK + EBLOCK + MPDATA                                   |
| Treated mesh as point cloud          | Separates surface vs. interior; uses connectivity                 |
| No material properties               | Predicts per-node Young's modulus                                 |
| PointNet encoder (no local features) | DGCNN with dynamic k-NN graphs                                    |
| Single decoder head                  | Triple-head (position + sizing + material)                        |
| No mesh generation                   | TetGen for actual tetrahedral meshing                             |
| Chamfer-only loss                    | Multi-component loss (Chamfer + KL + density + sizing + material) |
| Training diverged (NaN loss)         | Stable training with GroupNorm + proper init                      |

### Phase 1: Initial Bug Fixes (5 issues)

| Bug                           | Fix                                             |
| ----------------------------- | ----------------------------------------------- |
| Normals/points index mismatch | `sample_or_pad` returns shared indices          |
| O(F×N) normal estimation      | KDTree O(N log N)                               |
| Wrong TetGen surface input    | ConvexHull surface triangulation                |
| Z-only rotation augmentation  | Full SO(3) + mirror + anisotropic scale         |
| No model save/load            | `save_model/load_model` + `generate()` function |

### Phase 2: Material Property Integration

| Component                  | Change                                                |
| -------------------------- | ----------------------------------------------------- |
| `_parse_mpdata()`          | **New** — parses EX/NUXY/DENS per material ID         |
| `_read_element()`          | Returns `(tet_connectivity, mat_id)`                  |
| `read()`                   | Always uses direct parser (pyansys can't read MPDATA) |
| `compute_node_stiffness()` | **New** — per-node log₁₀(EX) from element materials   |
| `prepare_pair()`           | Stores raw material + `has_material` flag             |
| `TripleHeadDecoder`        | 3rd head predicts normalized bone stiffness [0,1]     |
| `MeshGenLoss`              | Adds `material_weight × MSE(pred, target)`            |

### Phase 3: SOTA Review & Critical Fixes (P0-P4)

| Fix                                   | What Changed                                                      | Impact                                        |
| ------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------- |
| **P0: Chunked Chamfer/Hausdorff**     | O(chunk×M) per chunk instead of O(N²)                             | **~800MB → ~6MB** per batch                   |
| **P0: Subsampled density**            | 1024-pt subsample instead of 4096                                 | **64MB → 4MB**                                |
| **P1: Global material normalization** | Raw log₁₀(EX) stored; dataset-wide min/max normalization to [0,1] | Consistent learning across patients           |
| **P2: Material gradient flow**        | Material head gets non-detached positions                         | Positions learn to cluster near cortical bone |
| **P3: GroupNorm**                     | Replaces BatchNorm in DGCNN (stabler with batch_size=4)           | Stable training                               |
| **P4: Xavier init**                   | Sigmoid output heads get Xavier uniform initialization            | Faster convergence, better gradient flow      |

### Current Code State

```
File: tetrahedral_mesh_v1.py
Lines: 1794
Syntax: ✅ Verified (ast.parse)
BatchNorm: 0 instances (GroupNorm throughout)
GroupNorm: 7 references (all DGCNN layers)
detach(): Only on sizing head, NOT material head
Xavier init: Applied to sizing + material Sigmoid heads
chunk_size: 12 references (chunked distance calculations)
Global material normalization: Implemented with range printing
```

---

## 10. Known Issues & Remaining Work

### 🔴 CRITICAL: Data Leakage in Cross-Validation

**Current:** `KFold(n_splits=5, shuffle=True)` — random splitting
**Problem:** Left/right femurs from same patient can split across folds
**Fix needed:** `GroupKFold` with patient ID as group key

### 🟠 HIGH: No CDB Export After Generation

The pipeline generates interior nodes + sizing + material → TetGen → tet mesh.
But it never writes the result back to CDB format for ANSYS FEA.
Need a `write_cdb()` function to produce valid CDB files.

### 🟡 MEDIUM: Missing References in resources.md

Current `resources.md` doesn't cite the papers this implementation is based on:

- DGCNN (Wang et al., 2019)
- FoldingNet (Yang et al., 2018)
- CVAE literature
- Bonemat (Taddei et al., 2007)
- Tanvir Faisal et al. (2022)
- SSDL paper (Jamalia et al., 2024)
- FEA Surrogate paper (Rabina et al., 2025)

### Future Work

1. **Run training** on university GPU machine
2. **Evaluate** generated meshes vs. ground truth CDB files
3. **Mesh quality metrics**: aspect ratio, Jacobian, skewness per tetrahedron
4. **Hyperparameter tuning**: based on initial training loss curves
5. **FEA validation**: run ANSYS simulation on generated mesh, compare stress/strain

---

## 11. Key References

### Papers Directly Used in Implementation

| #   | Paper                                                                                | Relevance                       |
| --- | ------------------------------------------------------------------------------------ | ------------------------------- |
| 1   | Wang et al., "DGCNN: Dynamic Graph CNN for Learning on Point Clouds" (2019)          | **Encoder architecture**        |
| 2   | Yang et al., "FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation" (2018) | **Decoder backbone**            |
| 3   | Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)                           | **CVAE framework**              |
| 4   | Si, "TetGen: A Quality Tetrahedral Mesh Generator" (2015)                            | **Meshing tool**                |
| 5   | Taddei et al., "Bonemat: An Open-Source Tool..." (2007, updated 2024)                | **What material head replaces** |

### Collaborator & Group Publications

| #   | Paper                                                                     | Authors                         | Venue                            |
| --- | ------------------------------------------------------------------------- | ------------------------------- | -------------------------------- |
| 6   | "Study of the significance of parameters... femoral fracture risk" (2022) | Tanvir R. Faisal et al.         | ResearchGate                     |
| 7   | SSDL: Semi-supervised 3D femur reconstruction from sparse QCT (2024)      | Jamalia Sultana, Naznin, Faisal | MBEC Springer                    |
| 8   | FEA Surrogate model for fracture risk prediction (2025)                   | Rabina Awal, Naznin, Faisal     | Expert Systems with Applications |

### SOTA References (2024-2025)

| #   | Topic                                                     | Source            |
| --- | --------------------------------------------------------- | ----------------- |
| 9   | Generator NN for mesh refinement (75% faster)             | SSRN 2024         |
| 10  | BPNN for Young's modulus prediction (13% stress accuracy) | ResearchGate 2025 |
| 11  | DGCNN for mammalian bone classification                   | ResearchGate 2025 |
| 12  | CLAY: Large-scale 3D generative model with multi-res VAE  | SIGGRAPH 2024     |
| 13  | BonematV4: Cortical-specific material mapping             | ResearchGate 2024 |
| 14  | Multi-NN for bone microstructure from clinical CT         | CSTAM 2024        |

---

## 12. File Structure

```
thesis/
├── tetrahedral_mesh_v1.py              # Main pipeline (1794 lines)
│   ├── Sections 1-9: CDB parsing, data structures, element/node handling
│   ├── Section 10: MeshRepresentation (surface/interior separation, material)
│   ├── Section 11: MeshDataset (augmentation, global material normalization)
│   ├── Section 12: DGCNN encoder (with GroupNorm)
│   ├── Section 13: TripleHeadDecoder (position + sizing + material)
│   ├── Section 14: SurfaceToVolumeCVAE model
│   ├── Section 15: Loss functions (chunked Chamfer, KL, density, sizing, material)
│   ├── Section 16: Training loop (K-fold CV, early stopping, LR scheduling)
│   └── Section 17: Main entry point (run_pipeline / run_kfold)
│
├── 4_bonemat_cdb_files/                # Dataset (198 CDB files)
│   ├── AB029_left_bonemat.cdb          # Example: Patient AB029, left femur
│   ├── AB029_right_bonemat.cdb         # Example: Patient AB029, right femur
│   ├── ...                             # ~100 patients × 2 sides
│   └── newMAFfile.xml                  # Bonemat configuration
│
├── context/                            # Thesis context files
│   ├── Thesis topics.png               # Supervisor's thesis topic list
│   ├── dataset-list.jpg                # Screenshot of CDB file listing
│   ├── dataset-sample-ss.jpg           # Screenshot of CDB file contents
│   └── gpt.md                          # ChatGPT conversation explaining thesis
│
├── ss/                                 # Screenshots
│   ├── dataset-list.jpg
│   └── dataset-sample-ss.jpg
│
├── resources.md                        # Reference papers (needs updating)
├── context.md                          # THIS FILE — comprehensive thesis context
└── .agent/                             # Agent workflow configs
```

### Dependencies

```python
# Core
torch, torch.nn, torch.optim           # Deep learning framework
numpy, pandas, scipy, sklearn           # Data handling, ML utilities

# 3D Processing
pytorch3d                               # Efficient Chamfer/Hausdorff distance
tetgen                                  # Tetrahedral mesh generation
pyvista                                 # 3D visualization

# ANSYS Integration
ansys-mapdl-reader                      # CDB file parsing (fallback)

# Custom (in tetrahedral_mesh_v1.py)
# Direct CDB parser (used instead of pyansys for MPDATA support)
```

---

> **This document should be updated whenever significant changes are made to the pipeline,
> architecture, or research direction. It serves as the single source of truth for anyone
> (human or AI) who needs to understand this thesis project.**
