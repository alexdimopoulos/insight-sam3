# INSIGHT — Indoor Scene Intelligence from Geometric-Semantic Hierarchy Transfer

> Combining foundation model segmentation, metric 3D geometry, and interoperability standards to generate structured indoor spatial intelligence.

[![Demo Video](https://img.youtube.com/vi/0Rlql4ui6mA/maxresdefault.jpg)](https://www.youtube.com/watch?v=0Rlql4ui6mA)

---

## Problem Statement

Indoor environments—buildings, tunnels, underground structures—lack the spatial intelligence infrastructure available for outdoor spaces. Three practical barriers limit progress:

**Bandwidth.** Dense 3D geometry can reach tens of gigabytes per floor—impractical for transmission over degraded or portable networks during active incidents.

**Labeled data.** Training 3D scene understanding models requires large annotated datasets. Manual point cloud labeling is time-intensive—a single large building can take weeks of skilled effort—and existing indoor datasets have minimal coverage of safety-critical infrastructure (standpipes, AEDs, electrical shutoffs, fire alarm panels).

**Interoperability.** No common vocabulary exists for describing building features across agencies and regions.

---

## Motivation

Prior work on [Point Cloud City Open3D-ML](https://github.com/alexdimopoulos/PointCloudCity-Open3D-ML) evaluated 3D ML models for public safety point cloud labeling using NIST PSIAP datasets. That effort revealed two findings:

1. **Labeled indoor data is scarce.** Training robust 3D segmentation models requires substantial annotated data that public safety datasets lack.

2. **Point clouds and images have complementary strengths.** Point clouds represent geometry precisely but are difficult to label semantically. Images are well-suited to semantic tasks—foundation models trained on internet-scale image data can identify objects that no 3D model has seen.

INSIGHT bridges this gap: use foundation model understanding on 2D images, then transfer semantics to 3D geometry through registered depth data.

---

## Approach

INSIGHT implements a semantic lifting pipeline: 2D images with registered depth data are processed through foundation model segmentation to produce structured 3D building intelligence.

```
RGB Images + Depth Data
        ↓
   SAM3 (text-prompted 2D segmentation)
        ↓
   Geometric lifting (project masks into 3D via depth)
        ↓
   Instance fusion (merge detections across viewpoints)
        ↓
   ┌────────────────────────────────────────┐
   │                                        │
   ▼                                        ▼
Pointcept Training Data              ISO 19164 Scene Graphs
(full geometry + labels)             (compact, transmittable)
```

SAM3's text-prompted segmentation identifies objects in 2D images ("fire extinguisher," "electrical panel," "exit sign") without task-specific training. Registered depth data provides metric 3D coordinates per pixel. Projecting 2D masks through depth produces semantically-labeled 3D segments that combine foundation model understanding with calibrated sensor accuracy.

The system produces two outputs:

1. **Pointcept-compatible point clouds** with per-point semantic labels and instance IDs for training 3D models.

2. **Lightweight scene graphs** encoding building structure as a queryable hierarchy—compressing tens of gigabytes of source geometry to single-digit megabytes, small enough to transmit over constrained networks.

---

## Example Point Cloud Segmentations

[![Point Cloud Segmentation Examples](https://img.youtube.com/vi/2ktOWO9D2uA/maxresdefault.jpg)](https://www.youtube.com/watch?v=2ktOWO9D2uA)

---

## Instance Fusion

Objects appear in multiple camera views. INSIGHT maintains a global instance registry, merging detections when 3D centroids fall within a spatial threshold (default: 0.5 meters). Result: one instance per physical object, with geometry aggregated from all viewing angles.

---

## Standards Alignment

Scene graph output implements ISO 19164 (IndoorGML) concepts: buildings contain floors, floors contain features, features have geometric and semantic properties. The class taxonomy aligns with IFC (Industry Foundation Classes) naming conventions.

Standardized vocabularies enable:
- CAD/BIM export compatibility
- Cross-agency data sharing
- Aggregated training datasets
- Dispatch and field application integration

---

## Role-Based Filtering

Different disciplines require different information. Scene graphs support query-time filtering via `filter_classes_for_responder()`:

| Role | Priority Features |
|------|-------------------|
| **Firefighter** | Doors, stairs, exit signs, windows, fire extinguishers, standpipes, fire hose cabinets, sprinklers, fire alarm panels, fire alarm pulls, electrical panels, gas shutoffs |
| **Hazmat** | Gas shutoffs, water shutoffs, electrical panels, doors, windows |
| **Police/Tactical** | Doors, windows, stairs, elevators, columns, walls |
| **EMS** | Doors, stairs, elevators, ramps, AEDs |
| **Search & Rescue** | Doors, stairs, windows, columns, walls, ceilings, floors |

Exit signs, doors, and stairs are always detected regardless of responder role.

---

## Class Taxonomy

23 classes organized by operational function:

| Category | Classes |
|----------|---------|
| **Egress & Access** | door, window, stairs, elevator, ramp, exit_sign, railing |
| **Fire Suppression** | fire_extinguisher, standpipe, fire_hose_cabinet, sprinkler |
| **Fire Alarm** | fire_alarm_panel, fire_alarm_pull |
| **Utility Control** | electrical_panel, gas_shutoff, water_shutoff |
| **Medical** | aed |
| **Structural** | wall, floor, ceiling, column |
| **Obstacles** | furniture |

Each class includes metadata: ISO class name, superclass, relevance category (CRITICAL, EGRESS, CONTROL, CONTEXT, OBSTACLE), and responder priority level (CRITICAL, HIGH, MEDIUM, LOW).

---

## Output Formats

### Scene Graphs (GraphML)

```
Building
  └── Floor
        ├── floor_surface (CellSpaceBoundary)
        ├── wall_surface (CellSpaceBoundary)
        ├── ceiling_surface (CellSpaceBoundary)
        ├── {area}_instance_1 (door)
        ├── {area}_instance_2 (fire_extinguisher)
        ├── {area}_instance_3 (electrical_panel)
        └── ...
```

Each feature node stores: semantic class, ISO metadata, 3D oriented bounding box (center, extent, rotation matrix), detection confidence, and responder priority.

**Size:** Compression of approximately four orders of magnitude—e.g., 39.3 GB geometry database reduced to 2.2 MB scene graph.

### Pointcept Training Data (.pth)

```python
{
    "coord":    np.float32 (N, 3),   # XYZ in meters
    "color":    np.float32 (N, 3),   # RGB normalized [0, 1]
    "normal":   np.float32 (N, 3),   # Surface normals (estimated)
    "segment":  np.int64 (N,),       # Semantic class ID
    "instance": np.int64 (N,),       # Instance ID
}
```

Compatible with [Pointcept](https://github.com/Pointcept/Pointcept) training pipelines (Point Transformer, SparseUNet, etc.).

### Intermediate Outputs

| File | Description |
|------|-------------|
| `geometry_database.h5` | Compressed point cloud storage with per-instance datasets |
| `instance_mapping.json` | Maps instance string IDs to integer IDs and class names |
| `processing_checkpoint.json` | Tracks processed images for resume capability |
| `model_detection_log.csv` | Per-detection logging with confidence scores and status |

---

## Usage

### Basic Commands

```bash
# Process an area
python run.py /path/to/area_1

# Process with Pointcept export
python run.py /path/to/area_1 --export-pointcept

# Custom output directory
python run.py /path/to/area_1 --output-dir ./my_results

# Adjust confidence threshold
python run.py /path/to/area_1 --conf-threshold 0.4

# Downsample output point cloud
python run.py /path/to/area_1 --export-pointcept --voxel-size 0.02
```

### Resume & Recovery

Processing automatically checkpoints progress. If interrupted, simply re-run the same command to resume:

```bash
# Automatically resumes from last checkpoint
python run.py /path/to/area_1

# Verify checkpoint integrity (dry run)
python run.py /path/to/area_1 --verify-checkpoint --dry-run

# Verify and fix checkpoint, then reprocess missing
python run.py /path/to/area_1 --reprocess-missing
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `area_path` | (required) | Path to area directory containing `data/rgb/` |
| `--output-dir` | `output_results` | Output directory for all results |
| `--weights-dir` | `model_weights` | Directory containing SAM3 model weights |
| `--export-pointcept` | off | Export to Pointcept `.pth` format after processing |
| `--conf-threshold` | 0.3 | Minimum confidence score for detections |
| `--voxel-size` | none | Voxel size (meters) for downsampling; if unset, no downsampling |
| `--verify-checkpoint` | off | Verify checkpoint against actual outputs |
| `--dry-run` | off | With `--verify-checkpoint`, only report (don't modify) |
| `--reprocess-missing` | off | Verify checkpoint and reprocess any missing images |

---

## Input Requirements

Stanford 2D-3D-Semantics format:

```
area_1/
└── data/
    ├── rgb/
    │   └── {frame_id}_domain_rgb.png
    └── global_xyz/
        └── {frame_id}_domain_global_xyz.exr
```

RGB images provide visual input to SAM3. EXR files encode world-space XYZ coordinates per pixel.

---

## Output Directory Structure

```
output_results/
├── model_detection_log.csv              # Detection log across all areas
├── pointcept/
│   ├── class_mapping.json               # Class names, IDs, responder groups
│   └── {area_name}.pth                  # Pointcept training data (if exported)
└── {area_name}/
    ├── results/                         # Annotated RGB images with detections
    │   └── {frame_id}.png
    ├── geometry_database.h5             # Point cloud data by instance
    ├── {area_name}_scene_graph.graphml  # Scene graph output
    ├── instance_mapping.json            # Instance ID mappings
    └── processing_checkpoint.json       # Resume checkpoint
```

---

## Configuration

### Detection Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| Confidence threshold | 0.3 | Minimum score to accept a detection |
| Mask threshold | 0.5 | Threshold for binary mask generation |
| IoU threshold | 0.5 | IoU threshold for duplicate removal (NMS) |

### Instance Merging

| Parameter | Default | Description |
|-----------|---------|-------------|
| Merge distance | 0.5 m | Max centroid distance to merge as same instance |
| Min points | 10 | Minimum points required to create valid instance |

### Memory Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| GC interval | 25 frames | Run garbage collection every N frames |
| H5 flush interval | 50 frames | Flush H5 database every N frames |
| Checkpoint interval | 10 frames | Save checkpoint every N processed images |

---

## Installation

### Requirements

- Python 3.11+
- PyTorch with CUDA support
- NVIDIA GPU recommended (CPU processing supported but slow)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- transformers (Hugging Face) — SAM3 model
- Open3D — Point cloud processing
- OpenEXR — Depth data reading
- NetworkX — Scene graph storage
- h5py — Geometry database
- rich — Progress display

### Model Weights

SAM3 is a gated model on Hugging Face. To download it:

1. **Create a Hugging Face account** at https://huggingface.co/join
2. **Accept the license** at https://huggingface.co/facebook/sam3
3. **Get your access token** at https://huggingface.co/settings/tokens

For local installation, log in once:
```bash
huggingface-cli login
```

For Docker, pass your token as an environment variable (see Docker section below).

If local weights exist in `./model_weights/sam3/`, they will be used instead of downloading.

### Docker

Build and run with GPU support:

```bash
# Build the image
docker build -t insight .

# Show help
docker run --rm insight

# Process an area (requires HF_TOKEN for gated model)
docker run --rm --gpus all \
  -e HF_TOKEN=your_huggingface_token \
  -v /path/to/dataset:/datasets \
  -v /path/to/output:/usr/src/app/output_results \
  insight /datasets/area_1 --export-pointcept

# With cached model weights (avoids re-downloading)
docker run --rm --gpus all \
  -v /path/to/dataset:/datasets \
  -v /path/to/weights:/usr/src/app/model_weights \
  -v /path/to/output:/usr/src/app/output_results \
  insight /datasets/area_1 --export-pointcept

# Alternative: Mount local HuggingFace cache (if already logged in)
docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/dataset:/datasets \
  -v /path/to/output:/usr/src/app/output_results \
  insight /datasets/area_1 --export-pointcept
```

**Volume mounts:**

| Mount Point | Purpose |
|-------------|---------|
| `/datasets` | Input data (area directories) |
| `/usr/src/app/model_weights` | SAM3 weights cache |
| `/usr/src/app/output_results` | Processing outputs |
| `/root/.cache/huggingface` | HuggingFace cache (optional) |

**Note:** Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

---

## Applications

**Pre-incident planning.** Scene graphs load instantly on mobile devices, enabling review of egress paths, standpipe locations, and utility shutoffs before arrival.

**Training data generation.** Programmatic labeling bootstraps datasets for 3D semantic segmentation. Generated labels can be human-verified rather than created from scratch.

**Bandwidth-constrained operations.** Scene graphs compress tens of gigabytes of source geometry to single-digit megabytes—approximately four orders of magnitude—making building intelligence transmittable over portable mesh networks.

**Interoperability research.** ISO-aligned taxonomy provides a testbed for cross-agency data sharing.

---

## Research Context

INSIGHT addresses capability gaps identified by NIST Public Safety Communications Research (PSCR) Division's Location-Based Services portfolio, which focuses on indoor mapping, tracking, and navigation for the public safety community.

---

## Related Resources

- [Stanford 2D-3D-Semantics Dataset](https://sdss.redivis.com/datasets/f304-a3vhsvcaf)
- [Pointcept](https://github.com/Pointcept/Pointcept)
- [IndoorGML (ISO 19164)](https://www.iso.org/standard/83153.html) | [OGC IndoorGML](https://www.ogc.org/standards/indoorgml/)
- [NIST PSCR Location-Based Services](https://www.nist.gov/ctl/pscr/research-portfolios/location-based-services)
- [Point Cloud City Open3D-ML](https://github.com/alexdimopoulos/PointCloudCity-Open3D-ML) — Prior work on 3D ML evaluation for public safety

---

## License

MIT

---

## Citation

```bibtex
@software{insight_2026,
  title={INSIGHT: Indoor Scene Intelligence from Geometric-Semantic Hierarchy Transfer},
  author={Dimopoulos, Alex},
  year={2026},
  url={https://github.com/alexdimopoulos/insight}
}
```
