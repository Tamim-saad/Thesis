# থিসিস অগ্রগতি রিপোর্ট (আপডেট)

### বিষয়: ANSYS Bone Mesh ডেটায় Machine Learning প্রয়োগ

### তারিখ: ১৭ ফেব্রুয়ারি, ২০২৬

---

## ✅ প্রধান আপডেট: Mesh Generation সফল

আগে আমরা CDB ফাইল থেকে শুধু **point cloud** (বিচ্ছিন্ন বিন্দু) দেখাতে পারতাম। এখন সফলভাবে **প্রকৃত 3D mesh surface** তৈরি করতে পারছি।

---

### কিভাবে Mesh তৈরি হচ্ছে (Technical Process):

ANSYS CDB ফাইলে দুইটি গুরুত্বপূর্ণ সেকশন আছে:

1. **NBLOCK** — প্রতিটি node-এর (X, Y, Z) coordinate থাকে। এটি আগে থেকেই পার্স করতে পারতাম। এটা দিয়ে শুধু বিচ্ছিন্ন বিন্দু (point cloud) দেখানো যায়।

2. **EBLOCK (নতুন পার্স করা হয়েছে)** — এখানে element connectivity তথ্য থাকে, অর্থাৎ কোন কোন node মিলে একটি element (tetrahedral/ত্রিমাত্রিক ঘর) তৈরি করে। প্রতিটি tetrahedral element-এ **৪টি node** থাকে।

**Mesh তৈরির ধাপগুলো:**

```
CDB ফাইল
  │
  ├── NBLOCK পার্স → Node coordinates (x, y, z) — ১৮,০০০+ nodes প্রতি ফাইলে
  │
  ├── EBLOCK পার্স → Element connectivity — কোন ৪টি node মিলে ১টি tetrahedron
  │
  ├── Surface Triangle Extraction:
  │     প্রতি tetrahedron-এর ৪টি face আছে (প্রতিটি face = ১টি triangle)
  │     যে face শুধু ১টি element-এর সাথে সংযুক্ত, সেটি surface face
  │     → এই surface triangles মিলে mesh-এর বাইরের আবরণ তৈরি হয়
  │
  └── 3D Rendering → Plotly Mesh3d দিয়ে interactive visualization
```

**সহজ ভাষায়:** NBLOCK দিয়ে আমরা জানি বিন্দুগুলো কোথায়, আর EBLOCK দিয়ে জানি কোন বিন্দুগুলো মিলে ত্রিভুজাকার/চতুস্তলকীয় ঘর তৈরি করে। দুটো মিলিয়ে সম্পূর্ণ 3D mesh surface দেখানো সম্ভব হচ্ছে।

---

### বিদ্যমান কাজ থেকে কিভাবে আলাদা (Novelty):

| বিষয়               | বিদ্যমান গবেষণা                                                                 | আমাদের পদ্ধতি                                                                              |
| ------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **ডেটা সোর্স**      | বেশিরভাগ কাজে pre-processed mesh ব্যবহার হয় (STL, OBJ ফরম্যাট)                 | আমরা **সরাসরি ANSYS CDB ফাইল** পার্স করছি — যেটা FEA সিমুলেশনের raw output                 |
| **Mesh Generation** | সাধারণত third-party সফটওয়্যার (ANSYS Workbench, Hypermesh) দিয়ে mesh দেখা হয় | আমরা **Python-এ programmatically** mesh তৈরি ও বিশ্লেষণ করছি                               |
| **ML Input**        | অধিকাংশ ML কাজে point cloud বা voxelized data ব্যবহার হয়                       | আমরা **element connectivity (graph structure)** সংরক্ষণ করছি — যেটা GNN-এ ব্যবহার করা যাবে |
| **প্রয়োগ ক্ষেত্র** | সাধারণ mesh quality check rule-based (threshold) পদ্ধতিতে হয়                   | আমরা **ML দিয়ে automated quality prediction** করতে চাই                                    |
| **Bone-specific**   | Bone mesh-এ ML-based quality assessment এর কাজ খুবই সীমিত                       | আমাদের কাজ **bone FEA mesh-এ** focused — clinical relevance আছে                            |

**মূল গবেষণা অবদান:** Raw ANSYS CDB ফাইল থেকে সরাসরি mesh পার্স করে, element connectivity-কে graph structure হিসেবে ব্যবহার করে, GNN দিয়ে bone mesh quality automated assessment — এই সম্পূর্ণ pipeline টি নতুন।

### আগে (Point Cloud) vs এখন (Mesh Surface):

![Point Cloud vs Mesh Surface — BH034_left_bonemat.cdb ফাইলের জন্য Point Cloud (বামে) এবং Enhanced Mesh Surface (ডানে) পাশাপাশি তুলনা](/home/tamim/.gemini/antigravity/brain/c136c843-d713-43ed-bd37-4f961a915976/pointcloud_vs_mesh.jpg)

![Bone Mesh Alphahull Approximation — Alphahull পদ্ধতিতে তৈরি 3D bone mesh](/home/tamim/.gemini/antigravity/brain/c136c843-d713-43ed-bd37-4f961a915976/mesh_alphahull.jpg)

---

## সম্পন্ন কাজের সারসংক্ষেপ (v5)

### ১. ডেটা পার্সিং

| বিষয়                 | তথ্য                                 |
| --------------------- | ------------------------------------ |
| মোট CDB ফাইল          | **১৯৮ টি** (bone mesh)               |
| মোট nodes             | **৩৫,৯৫,৮৫১** টি                     |
| প্রতি ফাইলে গড় nodes | ~১৮,০০০                              |
| Node range            | ১৩,৬৪৭ – ৩০,৫২২                      |
| NBLOCK পার্সিং        | ✅ সম্পন্ন                           |
| EBLOCK পার্সিং (নতুন) | ✅ সম্পন্ন — mesh surface তৈরি সম্ভব |

### ২. ডেটা প্রি-প্রসেসিং

- ✅ Node coordinates clean ও validate
- ✅ Point clouds normalize (unit sphere)
- ✅ ১০২৪ points per mesh — Farthest Point Sampling (FPS)
- ✅ Data augmentation (rotation, scaling, jitter, translation)

### ৩. ভিজ্যুয়ালাইজেশন পদ্ধতি

| পদ্ধতি                    | বিবরণ                                    | অবস্থা                 |
| ------------------------- | ---------------------------------------- | ---------------------- |
| Point Cloud (Scatter3d)   | বিচ্ছিন্ন বিন্দু                         | ✅ আগে থেকে ছিল        |
| **Mesh Surface** (নতুন)   | EBLOCK connectivity দিয়ে প্রকৃত surface | ✅ **নতুন যোগ হয়েছে** |
| **Alphahull Mesh** (নতুন) | Point cloud থেকে approximated surface    | ✅ **নতুন যোগ হয়েছে** |
| Wireframe                 | Mesh-এর edge structure                   | ✅ **নতুন যোগ হয়েছে** |

### ৪. ML মডেল ও ট্রেনিং রেজাল্ট

**PointNet AutoEncoder:**
| প্যারামিটার | মান |
|---|---|
| Epochs | ৩৮/৫০ (Early Stopping) |
| Best Validation Loss | **1.463** |
| Test Loss | **1.874** |

**Reconstruction Quality:**
| মেট্রিক | গড় মান |
|---|---|
| Chamfer Distance | 0.365 ± 0.026 |
| Original Coverage | 61.8% ± 7.6% |
| F1 Score | 0.402 ± 0.051 |

> **বিশ্লেষণ:** মডেল মূল shape-এর ~62% ক্যাপচার করতে পারছে। Reconstruction accuracy আরো উন্নত করা দরকার।

---

## পরবর্তী পদক্ষেপ

1. **FEA Mesh Quality Metrics** — Aspect Ratio, Scaled Jacobian, Skewness ইত্যাদি বের করা (EBLOCK data ব্যবহার করে)
2. **GNN/GAT মডেল** — Graph Neural Network দিয়ে mesh quality classification (নতুন research contribution)
3. **GPU ট্রেনিং** — Colab Pro ব্যবহার করে দ্রুত ও ভালো রেজাল্ট
4. **IEEE Conference Paper** — Overleaf-এ IEEE template শুরু করা (Intro, Background, Research Gap)

---

## সামগ্রিক অবস্থা

| কম্পোনেন্ট                          | অবস্থা              |
| ----------------------------------- | ------------------- |
| ANSYS CDB Parser (NBLOCK)           | ✅ সম্পন্ন          |
| **EBLOCK Parser → Mesh Generation** | ✅ **নতুন সম্পন্ন** |
| Data Preprocessing & Augmentation   | ✅ সম্পন্ন          |
| **3D Mesh Visualization**           | ✅ **নতুন সম্পন্ন** |
| PointNet AutoEncoder                | ✅ সম্পন্ন          |
| PointNet Classifier                 | ✅ সম্পন্ন          |
| Training Framework                  | ✅ সম্পন্ন          |
| FEA Quality Metrics                 | 🔄 পরবর্তী ধাপ      |
| GNN Model                           | 🔄 পরবর্তী ধাপ      |
| IEEE Paper Writing                  | 🔄 শুরু করতে হবে    |
