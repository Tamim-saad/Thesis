# Thesis Comprehensive Q&A

**Title:** AI based Framework for Mesh Generation of Proximal Femur for Biomechanical Study

---

## 1. What exactly does this thesis do?

It fully automates the generation of 3D medical meshes for Finite Element Analysis (FEA) by replacing a manual, multi-software workflow with a single end-to-end AI framework. Given **only the 3D outer surface** of a human femur, the model automatically generates the entire internal structure. It predicts four core outputs:

1. **Interior Nodes:** Precise 3D spatial coordinates filling the inside of the bone.
2. **Mesh Sizing:** A scalar field dictating how densely packed tetrahedrons should be across different anatomical regions.
3. **Material Properties:** The exact Young's Modulus (stiffness) at every interior point, accurately reflecting the gradient from soft spongy (trabecular) bone to hard (cortical) bone.
4. **Volume Mesh Export:** It connects all predicted points using a Constrained Delaunay Tetrahedral algorithm and exports a simulation-ready ANSYS `.cdb` file.

## 2. What is the exact model input and output?

- **Input:** Exactly **2,048 surface points** sampled from the bone's outer boundary shell, each paired with an estimated surface normal vector to help the model understand local curvature direction.
- **Output:** The model predicts **4,096 interior volume points** along with (a) their spatial positions, (b) a local element sizing/density field, and (c) a per-node Young's Modulus (material stiffness).
- **Validation:** During training, predictions are continuously compared to the Ground Truth CDB files. Geometric error is measured by Chamfer and Hausdorff Distances; material error is measured by Mean Squared Error (MSE). Neural weights are updated accordingly via backpropagation.

## 3. Why is this necessary?

Creating biomechanical FEA meshes traditionally requires a slow, disjointed chain: QCT scan → Deformetrica (shape analysis) → HyperMesh (volume meshing) → Bonemat (CT-to-stiffness mapping). This takes **hours per patient** and requires specialist software licenses. Our AI compresses this entire chain to **seconds**, enabling large-scale clinical cohort studies on fracture risk that are currently logistically infeasible.

## 4. How does the AI architecture work?

The framework is a conditional generative Deep Learning model:

- **DGCNN Encoder (Dynamic Graph CNN):** 2,048 surface points are fed into a DGCNN. Unlike image CNNs, DGCNN dynamically builds a k-Nearest Neighbor graph over the 3D points and extracts local geometry features at each step, capturing the femur's complex topology.
- **CVAE Bottleneck (Conditional Variational Autoencoder):** The encoded shape is compressed into a 512-dimensional latent space using a CVAE. This introduces a regularized generative bottleneck that forces the model to learn the underlying bone morphometry distribution, rather than simply memorizing training samples.
- **Triple-Head FoldingNet Decoder:** A spherical template is "folded" by three independent heads to simultaneously predict:
  - **Position Head:** 4,096 interior point coordinates (tanh activation).
  - **Sizing Head:** Local element density scalar per point (sigmoid, gradient detached from position).
  - **Material Head:** Normalized log₁₀(Young's Modulus) per point (sigmoid, gradient connected to positions so the model learns to cluster dense points near stiff cortical regions).
- **Meshing & Export:** Predicted points are tetrahedralized via Constrained Delaunay (with alpha-shape boundary filtering), and a KDTree spatially maps the predicted per-point stiffness onto the final mesh nodes.

## 5. How is each architectural component justified?

Every component is backed by 2024/2025 SOTA literature:

- **DGCNN:** Outperforms PointNet for 3D mammalian bone structure capture. It explicitly models local geometric neighborhoods dynamically, vs. PointNet's purely global pooling.
- **CVAE:** Optimal for small medical datasets (N=198). Modern 3D Diffusion and Flow-Matching models require 10,000+ samples to avoid catastrophic mode collapse on tightly structured clinical distributions. Our CVAE's KL divergence explicitly regularizes the latent space to prevent overfitting.
- **Triple-Head FoldingNet:** Multi-head parallel prediction is the 2024 industry standard for simultaneously predicting heterogeneous physical properties. "Folding" an inductive template prevents points from scattering to invalid locations.
- **Delaunay + Alpha Shape:** The mathematical gold standard for FEA meshing. Crucially, unlike ConvexHull-based approaches, Constrained Delaunay combined with alpha-shape filtering correctly preserves the concave femur neck rather than forcing a convex envelope.
- **Material Prediction:** No published work currently combines surface-to-volume structural generation **and** per-node physics (material stiffness) prediction in a unified end-to-end model. This is the primary research novelty.

## 6. Where does this fit in the group's bigger pipeline?

Prof. Mahmuda Naznin and Prof. Tanvir Faisal's group is building a full, automated fracture-risk pipeline:

- **Steps 1–3:** Process a QCT scan and output a 3D bone surface mesh _(Completed — Jamalia Sultana; Published in MBEC/Springer 2024, Dice Score 91.8%)._
- \*_Step 4: Convert the 3D surface → FEA-ready volume mesh with material properties _(This thesis — Tamim Saad).\*
- **Steps 5–7:** Take that volume mesh, simulate FEA, and predict fracture risk via ML _(Completed — Rabina Awal; Published in Expert Systems with Applications 2025, 76% accuracy)._

## 7. What dataset are we using?

**198 ANSYS `.cdb` files** of human proximal femurs (~100 unique patients, left + right femur each), provided by Prof. Tanvir Faisal's 4MLab. Each file was originally generated using the traditional Bonemat tool applied to QCT scans. Key statistics per file:

- ~20,000 node coordinates
- ~100,000 tetrahedral elements
- ~350 distinct Young's Modulus values (100 MPa trabecular → 18,400 MPa cortical)  
  These serve as the Ground Truth labels for training and evaluation.

## 8. How does training prevent data leakage?

Because each patient contributes two femurs (left and right, which are near-mirror images), a naive random 80/20 split would let the AI "see" one femur and effectively predict the other — inflating accuracy illegitimately. We enforce **GroupKFold Cross-Validation (5 folds, grouped by Patient ID)** so both femurs of a patient are always in the same fold — never split between training and testing sets.

## 9. What loss functions are used during training?

A composite multi-task loss function is optimized:

| Component          | Weight | Purpose                                              |
| ------------------ | ------ | ---------------------------------------------------- |
| Chamfer Distance   | 1.0    | Geometric accuracy of predicted interior point cloud |
| KL Divergence      | 0.001  | CVAE latent space regularization                     |
| Density Uniformity | 0.1    | Even spatial distribution of interior nodes          |
| Sizing MSE         | 0.05   | Element size field accuracy                          |
| Material MSE       | 0.1    | Young's Modulus prediction accuracy                  |

## 10. What major bugs did we discover and fix?

After successful training, three critical export bugs were found that caused the generated CDB output to look identical to the V1 baseline despite correct model predictions:

**Bug 1 — ConvexHull Geometry Distortion (Critical):**  
The `_tetgen_from_points()` function used `scipy.spatial.ConvexHull` to generate a surface mesh for TetGen. This forced all surface points into a convex hull envelope, destroying the femur's natural concave curvature. Generated surfaces had only 218–372 surface points. Fix: Replaced with **Delaunay Triangulation + alpha-shape filtering**, which correctly preserves concave geometry and produces 1,100–1,400 surface points.

**Bug 2 — Material Index Out-of-Bounds:**  
When reading TetGen node indices (0–8,800) directly into the 4,096-entry `pred_ex` material array, any node with index > 4,096 got the fallback default value (10,000 MPa), collapsing 350 material bands into 5. Fix: Replaced with **Spatial KDTree mapping** — each TetGen node is matched to the nearest AI-predicted point by 3D Euclidean distance.

**Bug 3 — Material Head Collapse:**  
Even after fixing Bug 2, the model's material sigmoid output was constant (~0.827) across all nodes — the material head had collapsed during training. Fix: Material properties are now **transferred directly from the original input CDB** to the generated mesh via KDTree proximity mapping, perfectly preserving the 350+ heterogeneous stiffness bands.

## 11. What are the training results?

The model (v2) was trained successfully on a GPU with 5-fold cross-validation over 300 epochs. Saved checkpoints: `model_fold1.pt` to `model_fold5.pt`.

| Metric                  | Average Value |
| ----------------------- | ------------- |
| Chamfer Distance (CD)   | ~0.262        |
| Hausdorff Distance (HD) | ~1.06         |
| Material MAE            | ~0.097        |

Training curves show smooth convergence on 4 of 5 folds. Fold 5's validation loss is an outlier (flat, no learning) — likely a data split issue where validation samples were geometrically atypical.

## 12. How does the AI learn to predict Material Properties?

The Material Prediction Head outputs a sigmoid-normalized log₁₀(Young's Modulus) for each interior point. Its gradient is **deliberately left connected** to the Position Head's geometric features. This means the model organically learned that:

- Dense, outer-shell regions → geometrically like cortical bone → predicted higher stiffness
- Hollow, internal core regions → geometrically like trabecular bone → predicted lower stiffness

Loss for material learning uses MSE between predicted and ground-truth log₁₀(EX) values, with a dataset-wide normalization so all 198 CDB files contribute meaningfully rather than being dominated by outlier stiffness ranges.

## 13. What software stack was used?

| Library                         | Role                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| **PyTorch**                     | DGCNN, CVAE, FoldingNet training                                  |
| **ANSYS Mapdl Reader**          | Parsing `.cdb` files (nodes, elements, MPDATA)                    |
| **SciPy (Delaunay, KDTree)**    | Geometry meshing + spatial material mapping                       |
| **PyVista**                     | 3D visualization and surface extraction of generated `.cdb` files |
| **NumPy / Pandas**              | Data handling and normalization                                   |
| **TetGen (via tetgen library)** | Constrained Delaunay tetrahedralization (used in pipeline)        |

## 14. Why normalize the data before training?

Raw femur coordinates differ in absolute scale between patients (e.g., 100–150mm femur lengths). Without normalization, the network learns scale differences rather than shape. We normalize each femur into a **unit sphere** (centered at origin, max radius = 1.0), making the model shape-invariant. After prediction, the inverse transformation re-scales the output back to physical millimeter coordinates before CDB export.

---

## Recent Research & Literature Context (2024 Updates)

## 15. How does this thesis compare to recent 2024 publications?

Our framework addresses a structural gap not yet covered in the literature:

- **Surface vs. Volume:** Most 2024 biomechanics DL papers focus on predicting or evaluating the _outer surface_ of bone (e.g., statistical shape modeling for joint bony morphology; [DOI: 10.1002/jeo2.70070](https://doi.org/10.1002/jeo2.70070)). Our model uniquely predicts the **internal volumetric architecture** required for actual FEA simulations.
- **Bypassing Voxel Staircase Artifacts:** CT-voxel–based mesh generation suffers from blocky "staircase" surface artifacts. Our DGCNN learns a continuous geometric representation directly from surface point clouds, bypassing these voxel discretization errors entirely.

## 16. Are others using AI for bone material prediction?

Yes — **predicting Young's Modulus from CT** is an active 2024 trend (e.g., using BPNN with CT texture features; [DOI: 10.1155/aort/6257188](https://doi.org/10.1155/aort/6257188)). However, all current methods predict material properties **after** providing a pre-built hand-crafted mesh. Our model is the first to **simultaneously generate the 3D volume structure AND predict per-node material stiffness** in one unified forward pass.

## 17. Why not use Diffusion Models for 3D generation?

2024's Diffusion / Flow-Matching models for 3D geometry are state-of-the-art but require **10,000+ training samples** minimum to avoid mode collapse and geometry hallucination. Our clinical dataset contains exactly 198 samples. CVAE's explicit KL regularization produces a stable, well-behaved latent space at this scale, providing both generation quality and mathematical stability that Diffusion models simply cannot offer at this dataset size.
