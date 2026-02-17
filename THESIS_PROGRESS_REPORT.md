# থিসিস অগ্রগতি রিপোর্ট

### বিষয়: ANSYS Bone Mesh ডেটায় Machine Learning প্রয়োগ

### তারিখ: ১৭ ফেব্রুয়ারি, ২০২৬

---

## ১. কাজের সারসংক্ষেপ

আমার থিসিসে ANSYS CDB ফাইল থেকে প্রাপ্ত bone mesh ডেটায় মেশিন লার্নিং প্রয়োগ করা হচ্ছে। বর্তমান নোটবুকের সর্বশেষ ভার্সন হলো **`ANSYS_Mesh_v5.ipynb`**। এই নোটবুকে একটি সম্পূর্ণ পাইপলাইন তৈরি করা হয়েছে যেখানে ডেটা পার্সিং, প্রি-প্রসেসিং, ভিজ্যুয়ালাইজেশন, মডেল ট্রেনিং এবং রেজাল্ট ইভ্যালুয়েশন অন্তর্ভুক্ত।

---

## ২. ডেটা পার্সিং ও লোডিং

### ২.১ NBLOCK পার্সিং (Node Coordinates)

- **১৯৮ টি CDB ফাইল** সফলভাবে পার্স করা হয়েছে
- মোট **৩৫,৯৫,৮৫১ টি nodes** (X, Y, Z coordinates) এক্সট্রাক্ট করা হয়েছে
- প্রতি ফাইলে গড়ে প্রায় **১৮,১৬১ টি nodes** রয়েছে
- Node count range: **১৩,৬৪৭** থেকে **৩০,৫২২**
- FORTRAN fixed-width ফরম্যাট সাপোর্ট সহ robust parser তৈরি করা হয়েছে

### ২.২ EBLOCK পার্সিং (Element Connectivity) — **নতুন ফিচার**

- CDB ফাইলের EBLOCK সেকশন থেকে element connectivity (কোন nodes মিলে triangular face তৈরি করে) পার্স করার ক্ষমতা যোগ করা হয়েছে
- এটি mesh surface ভিজ্যুয়ালাইজেশনের জন্য অত্যন্ত গুরুত্বপূর্ণ
- `parse_eblock_section()` এবং `parse_cdb_complete()` মেথড তৈরি করা হয়েছে

---

## ৩. ডেটা প্রি-প্রসেসিং

সম্পন্ন হওয়া কাজসমূহ:

- ✅ Node coordinates **clean ও validate** করা হয়েছে
- ✅ Point clouds **normalize** করা হয়েছে (unit sphere-এ)
- ✅ **১০২৪ points** per mesh **sampling** করা হয়েছে (FPS - Farthest Point Sampling)
- ✅ **Data Augmentation** প্রয়োগ করা হয়েছে:
  - Random rotation
  - Random scaling
  - Random jitter
  - Random translation

---

## ৪. ভিজ্যুয়ালাইজেশন

### ৪.১ আগের পদ্ধতি (Point Cloud)

আগে শুধুমাত্র `go.Scatter3d` দিয়ে বিচ্ছিন্ন বিন্দু (dots) হিসেবে দেখানো হতো, যা দেখতে mesh-এর মতো ছিল না।

### ৪.২ বর্তমান উন্নত পদ্ধতি (Mesh Surface)

নতুন ভিজ্যুয়ালাইজেশন ফাংশনগুলো যোগ করা হয়েছে:

- **`visualize_bone_mesh_3d()`** — EBLOCK element connectivity ব্যবহার করে প্রকৃত mesh surface তৈরি করে
- **`visualize_bone_mesh_approximated()`** — Alphahull ব্যবহার করে point cloud থেকে surface আনুমানিকভাবে তৈরি করে
- **`compare_mesh_visualizations()`** — Point Cloud বনাম Mesh Surface পাশাপাশি তুলনা করে দেখায়
- **`extract_surface_triangles()`** — Tetrahedral elements থেকে surface triangles বের করে
- **`extract_mesh_edges()`** — Wireframe ভিজ্যুয়ালাইজেশনের জন্য edges বের করে

### ৪.৩ ভিজ্যুয়ালাইজেশনের আউটপুট

**Point Cloud বনাম Mesh Surface তুলনা:**

![Point Cloud vs Mesh Surface — BH034_left_bonemat.cdb ফাইলের জন্য Point Cloud (বামে) এবং Enhanced Mesh Surface (ডানে) পাশাপাশি তুলনা](/home/tamim/.gemini/antigravity/brain/c136c843-d713-43ed-bd37-4f961a915976/pointcloud_vs_mesh.jpg)

**Alphahull Approximation Mesh:**

![Bone Mesh Alphahull Approximation — Alphahull পদ্ধতিতে তৈরি 3D bone mesh, যেখানে হাড়ের গঠন স্পষ্টভাবে দেখা যাচ্ছে](/home/tamim/.gemini/antigravity/brain/c136c843-d713-43ed-bd37-4f961a915976/mesh_alphahull.jpg)

---

## ৫. মডেল আর্কিটেকচার

### ৫.১ PointNet AutoEncoder

- **Encoder**: Shared MLPs (`64 → 128 → 1024`) + Max Pooling দিয়ে global feature vector (1024-dim) তৈরি
- **Decoder**: FC layers (`1024 → 512 → 1024 → N×3`) দিয়ে point cloud reconstruct
- **Loss Function**: Chamfer Distance

### ৫.২ PointNet Classifier

- Left vs Right bone classification-এর জন্য তৈরি
- Shared MLPs + Global Feature + FC Layers

### ৫.৩ 3D CNN

- Voxelized point cloud-এর উপর 3D convolution প্রয়োগ

---

## ৬. ট্রেনিং রেজাল্ট

### ৬.১ PointNet AutoEncoder ট্রেনিং

| প্যারামিটার               | মান                              |
| ------------------------- | -------------------------------- |
| মোট Epochs                | ৩৮ (৫০ এর মধ্যে, Early Stopping) |
| Learning Rate             | শুরুতে ~0.001, শেষে 0.000343     |
| Batch Size                | 4                                |
| Patience (Early Stopping) | 10                               |
| **Best Validation Loss**  | **1.463238**                     |
| Final Train Loss          | 2.633126                         |
| Final Validation Loss     | 2.029132                         |
| **Test Loss**             | **1.874263**                     |

> **দ্রষ্টব্য:** Training Loss প্রথম epoch-এ **8.45** থেকে ধীরে ধীরে **~2.63** তে নেমে এসেছে। Validation Loss **3.57** থেকে **~1.46** তে নেমেছে।

### ৬.২ Reconstruction Quality Assessment

| মেট্রিক                   | গড় মান | Std Dev |
| ------------------------- | ------- | ------- |
| Chamfer Distance          | 0.3647  | ±0.0261 |
| Hausdorff Distance        | 1.9126  | ±0.1734 |
| Original Coverage         | 61.8%   | ±7.6%   |
| Reconstruction Accuracy   | 29.7%   | ±3.8%   |
| F1 Score                  | 0.4016  | ±0.0507 |
| Volume Preservation       | 419.1%  | ±44.5%  |
| Surface Area Preservation | 249.6%  | ±16.2%  |

> **বিশ্লেষণ:** বর্তমান reconstruction quality এখনো উন্নতির প্রয়োজন। Original Coverage 61.8% মানে মডেল মূল shape-এর প্রায় 62% ক্যাপচার করতে পারছে, কিন্তু Reconstruction Accuracy মাত্র 29.7%।

---

## ৭. ব্যবহৃত টেকনোলজি ও লাইব্রেরি

- **PyTorch** (v2.9.0) — Deep Learning Framework
- **NumPy, Pandas** — ডেটা প্রসেসিং
- **Plotly** — 3D interactive ভিজ্যুয়ালাইজেশন (Mesh3d, Scatter3d)
- **Matplotlib, Seaborn** — Statistical ভিজ্যুয়ালাইজেশন
- **Scikit-learn** — Preprocessing ও Metrics
- **Google Colab** — Execution Environment (CPU মোডে রান হয়েছে)

---

## ৮. বর্তমান চ্যালেঞ্জ

1. **Google Colab JavaScript Error**: বড় mesh ভিজ্যুয়ালাইজেশনে মাঝে মাঝে Colab-এ "Page Unresponsive" এবং "Could not load JavaScript files" এরর আসছে। তবে এটি কোডের সমস্যা নয়, ব্রাউজার/Colab-এর সীমাবদ্ধতা।
2. **GPU Access**: বর্তমানে CPU তে ট্রেনিং হয়েছে, GPU পেলে আরো ভালো ও দ্রুত রেজাল্ট সম্ভব।
3. **Reconstruction Quality**: F1 Score এবং Accuracy আরো উন্নত করতে হবে।

---

## ৯. পরবর্তী পদক্ষেপ (Next Steps)

1. **আরো উন্নত মডেল ব্যবহার**: PointNet++, Graph Neural Networks (GNN) ইত্যাদি
2. **EBLOCK connectivity** ব্যবহার করে সঠিক mesh surface reconstruction
3. **GPU accelerated training** — Colab Pro বা অন্য GPU environment ব্যবহার
4. **Cross-validation** — Clinical data দিয়ে proper validation
5. **Hyperparameter tuning** — Learning rate, batch size, network depth ইত্যাদি optimize করা
6. **Publication-ready figures** তৈরি — IEEE Conference Paper-এর জন্য
7. **Left vs Right bone classification** টাস্ক ইমপ্লিমেন্ট ও ইভ্যালুয়েট করা

---

## ১০. সামগ্রিক অবস্থা

| কম্পোনেন্ট                                 | অবস্থা     |
| ------------------------------------------ | ---------- |
| ANSYS CDB Parser (NBLOCK + EBLOCK)         | ✅ সম্পন্ন |
| Data Preprocessor                          | ✅ সম্পন্ন |
| Data Augmentation                          | ✅ সম্পন্ন |
| PyTorch Dataset                            | ✅ সম্পন্ন |
| PointNet Models (AutoEncoder + Classifier) | ✅ সম্পন্ন |
| 3D CNN Model                               | ✅ সম্পন্ন |
| Training Framework                         | ✅ সম্পন্ন |
| Mesh Visualization (Enhanced)              | ✅ সম্পন্ন |
| Reconstruction Quality Assessment          | ✅ সম্পন্ন |
| Model Optimization & Improvement           | 🔄 চলমান   |
| IEEE Paper Writing                         | 🔄 চলমান   |
