#!/usr/bin/env python3
"""
SAM3 3D Processor - Pointcept Compatible

A point cloud processing pipeline for public safety applications using SAM3
segmentation. Detects and labels safety-critical objects (exits, fire equipment,
utilities) in 3D scenes for use by first responders.

# README: INSTALLATION
# --------------------
# Requires Python 3.10+
# Install dependencies: pip install -r requirements.txt
# Download model weights to ./model_weights/sam3/ or use --weights-dir flag

# README: QUICK START
# -------------------
# Basic usage:
#   python run.py /path/to/area_directory
#
# With all options:
#   python run.py /path/to/area --output-dir ./results --export-pointcept
#
# Resume interrupted processing:
#   python run.py /path/to/area  # Automatically resumes from checkpoint

# README: INPUT FORMAT
# --------------------
# Expected directory structure:
#   area_directory/
#   ├── data/
#   │   ├── rgb/           # RGB images (*.png)
#   │   └── global_xyz/    # Corresponding EXR depth files (*_domain_global_xyz.exr)
#
# RGB images should be named: {frame_id}_domain_rgb.png
# Depth files should be named: {frame_id}_domain_global_xyz.exr

# README: OUTPUT FORMAT
# ---------------------
# Pointcept-compatible output structure:
#   coord: np.float32 (N, 3) - XYZ coordinates in meters
#   color: np.float32 (N, 3) - RGB normalized to [0, 1]
#   normal: np.float32 (N, 3) - Surface normal vectors
#   segment: np.int64 (N,) - Semantic class IDs (-1 for unlabeled)
#   instance: np.int64 (N,) - Instance IDs (-1 for unlabeled)
#
# Additional outputs:
#   - Scene graph (GraphML format) with spatial relationships
#   - Annotated RGB images with detection overlays
#   - Instance mapping JSON for ID lookups
#   - Class mapping JSON for Pointcept dataset config

# README: DETECTION CLASSES
# -------------------------
# 23 public safety classes organized by category:
#   - Egress & Access: door, window, stairs, elevator, ramp, exit_sign, railing
#   - Fire Suppression: fire_extinguisher, standpipe, fire_hose_cabinet, sprinkler
#   - Fire Alarm: fire_alarm_panel, fire_alarm_pull
#   - Utility Shutoffs: electrical_panel, gas_shutoff, water_shutoff
#   - Medical: aed
#   - Structural: wall, floor, ceiling, column
#   - Obstacles: furniture
"""

__author__ = "Alexander Nikitas Dimopoulos"
__license__ = "MIT"

import os
import gc
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import cv2
import networkx as nx
import csv
import json
import sys
import h5py
import logging
import contextlib
import threading
from datetime import datetime
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, NamedTuple

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table
from rich import box

import open3d as o3d
import OpenEXR
import Imath
from transformers import Sam3Model, Sam3Processor

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

console = Console(force_terminal=True)

# =============================================================================
# TYPE ALIASES & CONSTANTS
# =============================================================================

NDArray = np.ndarray

# README: MODEL CONFIGURATION
# These values are tuned for SAM3; adjust if using different models
CLIP_MAX_LENGTH = 32  # Maximum token length for text prompts
TEXT_QUERY_BATCH_SIZE = 23  # Batch size for text queries (matches class count)

# README: PROCESSING THRESHOLDS
# Adjust these to tune detection sensitivity vs. precision
DEFAULT_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence score to accept detection
DEFAULT_MASK_THRESHOLD = 0.5  # Threshold for binary mask generation
DEFAULT_IOU_THRESHOLD = 0.5  # IoU threshold for duplicate detection removal

# README: INSTANCE MERGING
# Controls how detections across frames are merged into instances
INSTANCE_MERGE_DISTANCE_M = 0.5  # Max distance (meters) to merge as same instance
MIN_POINTS_FOR_INSTANCE = 10  # Minimum points required to create valid instance

# README: MEMORY MANAGEMENT
# Adjust for your GPU memory constraints
GC_CLEANUP_INTERVAL = 25  # Run garbage collection every N frames
H5_FLUSH_INTERVAL = 50  # Flush H5 database every N frames
CHECKPOINT_SAVE_INTERVAL = 10  # Save checkpoint every N processed images

# README: DEFAULT PATHS
# Override via command line arguments
DEFAULT_OUTPUT_DIR = Path("output_results")
DEFAULT_WEIGHTS_DIR = Path("model_weights")
HUGGINGFACE_MODEL_ID = "facebook/sam3"  # Fallback if local weights not found


# =============================================================================
# CLASS REGISTRY - Public Safety Classes (23 classes)
# README: CLASS DEFINITIONS
# Each class includes metadata for:
#   - iso_class/iso_superclass: OGC IndoorGML classification
#   - relevance: CRITICAL/EGRESS/CONTROL/CONTEXT/OBSTACLE
#   - responder_priority: CRITICAL/HIGH/MEDIUM/LOW
# =============================================================================

CLASS_REGISTRY = {
    # Egress & Access
    "door":              {"iso_class": "Door",      "iso_superclass": "ConstructiveFeature", "relevance": "EGRESS",   "responder_priority": "HIGH"},
    "window":            {"iso_class": "Window",    "iso_superclass": "ConstructiveFeature", "relevance": "EGRESS",   "responder_priority": "HIGH"},
    "stairs":            {"iso_class": "Stair",     "iso_superclass": "Pathway",             "relevance": "EGRESS",   "responder_priority": "HIGH"},
    "elevator":          {"iso_class": "Elevator",  "iso_superclass": "Pathway",             "relevance": "EGRESS",   "responder_priority": "MEDIUM"},
    "ramp":              {"iso_class": "Ramp",      "iso_superclass": "Pathway",             "relevance": "EGRESS",   "responder_priority": "MEDIUM"},
    "exit_sign":         {"iso_class": "ExitSign",  "iso_superclass": "SafetyEquipment",     "relevance": "CRITICAL", "responder_priority": "HIGH"},
    "railing":           {"iso_class": "Railing",   "iso_superclass": "ConstructiveFeature", "relevance": "CONTEXT",  "responder_priority": "LOW"},
    
    # Fire Suppression
    "fire_extinguisher":   {"iso_class": "FireExtinguisher",   "iso_superclass": "SafetyEquipment", "relevance": "CRITICAL", "responder_priority": "HIGH"},
    "standpipe":           {"iso_class": "StandpipeConnection","iso_superclass": "SafetyEquipment", "relevance": "CRITICAL", "responder_priority": "CRITICAL"},
    "fire_hose_cabinet":   {"iso_class": "FireHoseCabinet",    "iso_superclass": "SafetyEquipment", "relevance": "CRITICAL", "responder_priority": "HIGH"},
    "sprinkler":           {"iso_class": "FireSprinkler",      "iso_superclass": "SafetyEquipment", "relevance": "CRITICAL", "responder_priority": "MEDIUM"},
    
    # Fire Alarm & Detection
    "fire_alarm_panel":    {"iso_class": "FireAlarmPanel",     "iso_superclass": "SafetyEquipment", "relevance": "CRITICAL", "responder_priority": "CRITICAL"},
    "fire_alarm_pull":     {"iso_class": "FireAlarmPull",      "iso_superclass": "SafetyEquipment", "relevance": "CRITICAL", "responder_priority": "HIGH"},
    
    # Utility Shutoffs
    "electrical_panel":    {"iso_class": "ElectricalPanel", "iso_superclass": "UtilityControl", "relevance": "CONTROL", "responder_priority": "CRITICAL"},
    "gas_shutoff":         {"iso_class": "GasShutoff",      "iso_superclass": "UtilityControl", "relevance": "CONTROL", "responder_priority": "CRITICAL"},
    "water_shutoff":       {"iso_class": "WaterShutoff",    "iso_superclass": "UtilityControl", "relevance": "CONTROL", "responder_priority": "HIGH"},
    
    # Medical
    "aed":                 {"iso_class": "AED", "iso_superclass": "MedicalEquipment", "relevance": "CRITICAL", "responder_priority": "CRITICAL"},
    
    # Structural
    "wall":              {"iso_class": "Wall",    "iso_superclass": "ConstructiveFeature", "relevance": "CONTEXT", "responder_priority": "LOW"},
    "floor":             {"iso_class": "Slab",    "iso_superclass": "ConstructiveFeature", "relevance": "CONTEXT", "responder_priority": "LOW"},
    "ceiling":           {"iso_class": "Ceiling", "iso_superclass": "ConstructiveFeature", "relevance": "CONTEXT", "responder_priority": "LOW"},
    "column":            {"iso_class": "Column",  "iso_superclass": "ConstructiveFeature", "relevance": "CONTEXT", "responder_priority": "MEDIUM"},
    
    # Obstacles
    "furniture":         {"iso_class": "Furniture", "iso_superclass": "Obstacle", "relevance": "OBSTACLE", "responder_priority": "LOW"},
}

DETECTION_CLASSES = list(CLASS_REGISTRY.keys())
SCENE_GRAPH_EXCLUDED_CLASSES = {'floor', 'wall', 'ceiling'}
SURFACE_CLASSES = {'floor', 'wall', 'ceiling'}

SURFACE_CLASS_INSTANCE_IDS = {
    'floor': 10000,
    'wall': 10001,
    'ceiling': 10002,
}


# =============================================================================
# SAM3 PROMPT CONFIGURATION
# README: PROMPT ENGINEERING
# Multiple prompts per class improve recall; primary prompt used for labeling
# =============================================================================

SAM3_PROMPTS = {
    "door":              ["door", "doorway", "entrance"],
    "window":            ["window"],
    "stairs":            ["stairs", "staircase", "stairwell"],
    "elevator":          ["elevator", "elevator door"],
    "ramp":              ["ramp", "wheelchair ramp", "incline"],
    "exit_sign":         ["exit sign", "emergency exit sign", "green exit sign", "lighted exit sign"],
    "railing":           ["railing", "handrail", "guardrail", "banister"],
    "fire_extinguisher": ["fire extinguisher", "red fire extinguisher", "extinguisher"],
    "standpipe":         ["standpipe", "fire department connection", "FDC", "hose connection", "fire hose valve"],
    "fire_hose_cabinet": ["fire hose cabinet", "fire hose", "hose cabinet", "fire cabinet"],
    "sprinkler":         ["sprinkler", "fire sprinkler", "sprinkler head"],
    "fire_alarm_panel":  ["fire alarm panel", "fire alarm control panel", "alarm panel", "fire panel"],
    "fire_alarm_pull":   ["fire alarm", "fire alarm pull station", "pull station", "fire pull", "alarm pull"],
    "electrical_panel":  ["electrical panel", "breaker box", "electrical box", "circuit breaker", "fuse box"],
    "gas_shutoff":       ["gas valve", "gas shutoff", "gas meter", "gas line"],
    "water_shutoff":     ["water valve", "water shutoff", "water meter", "main water valve"],
    "aed":               ["AED", "defibrillator", "automated external defibrillator", "heart defibrillator"],
    "wall":              ["wall"],
    "floor":             ["floor", "ground"],
    "ceiling":           ["ceiling"],
    "column":            ["column", "pillar", "support column", "beam", "structural beam"],
    "furniture":         ["furniture", "chair", "table", "desk", "sofa", "couch", "bed", "cabinet",
                          "shelving", "bookshelf", "wardrobe", "dresser", "bench"],
}

SAM3_PRIMARY_PROMPT = {
    "door": "door",
    "window": "window",
    "stairs": "stairs",
    "elevator": "elevator",
    "ramp": "ramp",
    "exit_sign": "exit sign",
    "railing": "railing",
    "fire_extinguisher": "fire extinguisher",
    "standpipe": "standpipe",
    "fire_hose_cabinet": "fire hose cabinet",
    "sprinkler": "sprinkler",
    "fire_alarm_panel": "fire alarm panel",
    "fire_alarm_pull": "fire alarm",
    "electrical_panel": "electrical panel",
    "gas_shutoff": "gas valve",
    "water_shutoff": "water valve",
    "aed": "AED",
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "column": "column",
    "furniture": "furniture",
}


# =============================================================================
# RESPONDER GROUPS
# README: RESPONDER-SPECIFIC FILTERING
# Use filter_classes_for_responder() to get relevant classes for each role
# =============================================================================

RESPONDER_GROUPS = {
    "FIREFIGHTER": [
        "door", "stairs", "exit_sign", "window",
        "fire_extinguisher", "standpipe", "fire_hose_cabinet", "sprinkler",
        "fire_alarm_panel", "fire_alarm_pull",
        "electrical_panel", "gas_shutoff",
    ],
    "HAZMAT": [
        "gas_shutoff", "water_shutoff", "electrical_panel",
        "door", "window",
    ],
    "POLICE_TACTICAL": [
        "door", "window", "stairs", "elevator",
        "column", "wall",
    ],
    "EMS": [
        "door", "stairs", "elevator", "ramp",
        "aed",
    ],
    "SEARCH_RESCUE": [
        "door", "stairs", "window",
        "column", "wall", "ceiling", "floor",
    ],
}

ALWAYS_DETECT = ["exit_sign", "door", "stairs"]


# =============================================================================
# LEGACY CLASS MAPPING
# README: BACKWARD COMPATIBILITY
# Maps older class names to current schema for dataset migration
# =============================================================================

LEGACY_CLASS_MAP = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "door": "door",
    "window": "window",
    "stairs": "stairs",
    "elevator": "elevator",
    "ramp": "ramp",
    "railing": "railing",
    "column": "column",
    "beam": "column",
    "escalator": "stairs",
    "fire extinguisher": "fire_extinguisher",
    "standpipe connection": "standpipe",
    "fire hose cabinet": "fire_hose_cabinet",
    "fire sprinkler": "sprinkler",
    "fire alarm pull station": "fire_alarm_pull",
    "fire alarm control panel": "fire_alarm_panel",
    "exit sign": "exit_sign",
    "emergency exit sign": "exit_sign",
    "electrical panel": "electrical_panel",
    "gas meter": "gas_shutoff",
    "gas shutoff valve": "gas_shutoff",
    "water shutoff valve": "water_shutoff",
    "AED": "aed",
    "chair": "furniture",
    "table": "furniture",
    "sofa": "furniture",
    "bed": "furniture",
    "cabinet": "furniture",
    "wardrobe": "furniture",
    "lamp": "furniture",
    "tv": "furniture",
    "laptop": "furniture",
    "potted plant": "furniture",
    "photo frame": "furniture",
    "sign": "exit_sign",
    "placard": "exit_sign",
    "fire": None,
    "person": None,
}


# =============================================================================
# CLASS MAPPING - Derived from registry
# =============================================================================

CLASS_TO_ID = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}
NUM_CLASSES = len(DETECTION_CLASSES)


# =============================================================================
# SEMANTIC COLOR MAP (visualization only)
# README: VISUALIZATION
# Colors chosen for visual distinction in annotated outputs
# =============================================================================

DEFAULT_SEMANTIC_COLOR = (0.5, 0.5, 0.5)

SEMANTIC_COLORS = {
    "door":              (0.10, 1.00, 0.30),  # Bright green - egress
    "window":            (0.20, 0.90, 0.50),
    "stairs":            (0.00, 0.95, 0.70),
    "elevator":          (0.20, 0.70, 0.80),
    "ramp":              (0.30, 0.85, 0.60),
    "exit_sign":         (0.00, 1.00, 0.50),  # Bright green - critical
    "railing":           (0.50, 0.70, 0.60),
    "fire_extinguisher": (1.00, 0.10, 0.10),  # Red - fire equipment
    "standpipe":         (1.00, 0.00, 0.30),
    "fire_hose_cabinet": (0.90, 0.15, 0.20),
    "sprinkler":         (1.00, 0.40, 0.40),
    "fire_alarm_panel":  (1.00, 0.30, 0.00),  # Orange - alarms
    "fire_alarm_pull":   (1.00, 0.20, 0.10),
    "electrical_panel":  (1.00, 0.80, 0.00),  # Yellow - utilities
    "gas_shutoff":       (1.00, 0.60, 0.00),
    "water_shutoff":     (0.30, 0.60, 1.00),  # Blue - water
    "aed":               (0.00, 0.50, 1.00),  # Blue - medical
    "wall":              (0.45, 0.45, 0.50),  # Gray - structural
    "floor":             (0.35, 0.35, 0.40),
    "ceiling":           (0.40, 0.40, 0.45),
    "column":            (0.55, 0.50, 0.45),
    "furniture":         (0.55, 0.45, 0.35),  # Brown - obstacles
}


# =============================================================================
# CHECKPOINT MANAGEMENT
# README: CHECKPOINTING
# Processing can be interrupted and resumed at any time.
# Checkpoint stores list of processed image stems in JSON format.
# =============================================================================

def load_checkpoint(checkpoint_path: Path) -> set:
    """
    Load set of already-processed image stems from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        
    Returns:
        Set of image stem strings that have been processed
    """
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_images', []))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()


def save_checkpoint(checkpoint_path: Path, processed_images: set) -> None:
    """
    Save checkpoint with all processed image stems.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        processed_images: Set of processed image stem strings
    """
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'processed_images': list(processed_images),
            'last_updated': datetime.now().isoformat(),
            'count': len(processed_images)
        }, f)


def append_to_checkpoint(checkpoint_path: Path, image_stem: str, processed_images: set) -> None:
    """
    Add a single processed image to checkpoint with periodic full saves.
    
    Uses incremental log file between full saves for crash recovery.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        image_stem: Stem of newly processed image
        processed_images: Set to update (modified in place)
    """
    processed_images.add(image_stem)
    if len(processed_images) % CHECKPOINT_SAVE_INTERVAL == 0:
        save_checkpoint(checkpoint_path, processed_images)
    else:
        log_path = checkpoint_path.with_suffix('.log')
        with open(log_path, 'a') as f:
            f.write(f"{image_stem}\n")


def consolidate_checkpoint(checkpoint_path: Path, processed_images: set) -> None:
    """
    Final save and cleanup of checkpoint files.
    
    Removes incremental log file after consolidating to main checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        processed_images: Final set of all processed images
    """
    save_checkpoint(checkpoint_path, processed_images)
    log_path = checkpoint_path.with_suffix('.log')
    if log_path.exists():
        log_path.unlink()


def recover_checkpoint(checkpoint_path: Path) -> set:
    """
    Recover from both JSON checkpoint and incremental log.
    
    Handles crash recovery by combining main checkpoint with any
    incremental entries logged since last full save.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        
    Returns:
        Complete set of processed image stems
    """
    processed = load_checkpoint(checkpoint_path)
    log_path = checkpoint_path.with_suffix('.log')
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    stem = line.strip()
                    if stem:
                        processed.add(stem)
        except IOError:
            pass
    return processed


def scan_output_directory_for_processed(results_dir: Path, input_image_stems: set) -> set:
    """
    Scan output directory to find images that have already been annotated.
    
    Provides secondary recovery mechanism by checking actual output files.
    
    Args:
        results_dir: Path to results output directory
        input_image_stems: Set of all input image stems for matching
        
    Returns:
        Set of image stems that have corresponding output files
    """
    already_annotated = set()
    
    if not results_dir.exists():
        return already_annotated
    
    annotated_files = list(results_dir.glob("*.png")) + list(results_dir.glob("*.PNG"))
    
    for annotated_path in annotated_files:
        annotated_stem = annotated_path.stem
        potential_input_stem = f"{annotated_stem}_domain_rgb"
        if potential_input_stem in input_image_stems:
            already_annotated.add(potential_input_stem)
            continue
        if annotated_stem in input_image_stems:
            already_annotated.add(annotated_stem)
    
    return already_annotated


def verify_checkpoint_against_outputs(
    checkpoint_path: Path,
    results_dir: Path,
    input_image_paths: list,
    dry_run: bool = False
) -> Tuple[set, set]:
    """
    Verify checkpoint entries have corresponding output files.
    
    Useful for detecting interrupted runs where checkpoint was saved
    but output file was not written.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        results_dir: Path to results output directory
        input_image_paths: List of input image paths
        dry_run: If True, only report without modifying checkpoint
        
    Returns:
        Tuple of (verified_processed, missing_outputs) sets
    """
    checkpoint_processed = recover_checkpoint(checkpoint_path)
    
    if not checkpoint_processed:
        console.print("[dim]No checkpoint found or checkpoint is empty[/dim]")
        return set(), set()
    
    input_stems = {p.stem for p in input_image_paths}
    actual_outputs = scan_output_directory_for_processed(results_dir, input_stems)
    missing_outputs = checkpoint_processed - actual_outputs
    verified_processed = checkpoint_processed & actual_outputs
    extra_outputs = actual_outputs - checkpoint_processed
    
    console.print()
    console.rule("[bold cyan]Checkpoint Verification Report[/bold cyan]")
    console.print(f"  Checkpoint entries:    {len(checkpoint_processed):>8,}")
    console.print(f"  Actual output files:   {len(actual_outputs):>8,}")
    console.print(f"  Verified (in both):    {len(verified_processed):>8,}")
    
    if missing_outputs:
        console.print()
        console.print(f"[bold yellow]Warning:[/bold yellow] {len(missing_outputs):,} checkpoint entries WITHOUT output files:")
        for stem in sorted(missing_outputs)[:15]:
            console.print(f"  [dim]- {stem}[/dim]")
        if len(missing_outputs) > 15:
            console.print(f"  [dim]... and {len(missing_outputs) - 15} more[/dim]")
        
        if not dry_run:
            save_checkpoint(checkpoint_path, verified_processed)
            console.print()
            console.print(f"[green]Removed {len(missing_outputs):,} unverified entries from checkpoint[/green]")
            console.print(f"[green]Checkpoint now has {len(verified_processed):,} verified entries[/green]")
        else:
            console.print()
            console.print(f"[cyan]Dry run:[/cyan] Would remove {len(missing_outputs):,} entries")
            console.print("[dim]Run without --dry-run to apply changes[/dim]")
    else:
        console.print()
        console.print(f"[green]All {len(checkpoint_processed):,} checkpoint entries have corresponding outputs[/green]")
    
    if extra_outputs:
        console.print(f"[dim]Found {len(extra_outputs):,} output files not in checkpoint (will be added)[/dim]")
    
    return verified_processed, missing_outputs


def recover_checkpoint_with_output_scan(
    checkpoint_path: Path,
    results_dir: Path,
    input_image_paths: list,
    verify_mode: bool = False
) -> set:
    """
    Comprehensive recovery checking both checkpoint files AND output directory.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        results_dir: Path to results output directory
        input_image_paths: List of input image paths
        verify_mode: If True, verify and clean checkpoint before recovery
        
    Returns:
        Complete set of processed image stems from all sources
    """
    if verify_mode:
        verified, removed = verify_checkpoint_against_outputs(
            checkpoint_path, results_dir, input_image_paths, dry_run=False
        )
        
        input_stems = {p.stem for p in input_image_paths}
        output_processed = scan_output_directory_for_processed(results_dir, input_stems)
        new_from_output = output_processed - verified
        
        if new_from_output:
            console.print(f"[dim]Adding {len(new_from_output)} outputs not in checkpoint[/dim]")
            verified.update(new_from_output)
            save_checkpoint(checkpoint_path, verified)
        
        return verified
    else:
        processed = recover_checkpoint(checkpoint_path)
        
        input_stems = {p.stem for p in input_image_paths}
        output_processed = scan_output_directory_for_processed(results_dir, input_stems)
        
        new_from_output = output_processed - processed
        processed.update(output_processed)
        
        if new_from_output:
            console.print(f"[dim]Found {len(new_from_output)} additional processed images from output directory scan[/dim]")
        
        return processed


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_gpu_memory() -> None:
    """Clear GPU memory cache and synchronize CUDA operations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def periodic_cleanup(frame_index: int, interval: int = GC_CLEANUP_INTERVAL) -> None:
    """
    Run garbage collection periodically to prevent memory buildup.
    
    Args:
        frame_index: Current frame number in processing loop
        interval: Number of frames between cleanup operations
    """
    if frame_index > 0 and frame_index % interval == 0:
        gc.collect()
        clear_gpu_memory()


# =============================================================================
# CLASS UTILITIES
# =============================================================================

def export_class_mapping(output_path: Path) -> None:
    """
    Export class names and IDs for Pointcept dataset config.
    
    Args:
        output_path: Path to output JSON file
    """
    mapping = {
        "classes": DETECTION_CLASSES,
        "num_classes": NUM_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "id_to_class": ID_TO_CLASS,
        "responder_groups": RESPONDER_GROUPS,
        "always_detect": ALWAYS_DETECT,
    }
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    console.print(f"[green]Class mapping exported:[/green] {output_path}")


def get_semantic_color(class_name: str) -> Tuple[float, float, float]:
    """
    Get RGB color tuple for visualization.
    
    Args:
        class_name: Name of detection class
        
    Returns:
        RGB tuple with values in [0, 1] range
    """
    key = class_name.lower().strip().replace(' ', '_')
    return SEMANTIC_COLORS.get(key, DEFAULT_SEMANTIC_COLOR)


def get_responder_priority(class_name: str) -> str:
    """
    Get responder priority level for a class.
    
    Args:
        class_name: Name of detection class
        
    Returns:
        Priority string: 'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'
    """
    if class_name in CLASS_REGISTRY:
        return CLASS_REGISTRY[class_name].get("responder_priority", "LOW")
    return "LOW"


def filter_classes_for_responder(responder_type: str) -> list:
    """
    Get relevant classes for a specific responder type.
    
    Args:
        responder_type: One of FIREFIGHTER, HAZMAT, POLICE_TACTICAL, EMS, SEARCH_RESCUE
        
    Returns:
        List of class names relevant to that responder type
    """
    return RESPONDER_GROUPS.get(responder_type.upper(), DETECTION_CLASSES)


# =============================================================================
# OUTPUT SUPPRESSION UTILITY
# =============================================================================

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# =============================================================================
# H5 DATABASE OPERATIONS
# =============================================================================

def update_instance_h5_pointcept(
    h5_handle: h5py.File,
    instance_id: str,
    new_points: NDArray,
    new_colors_rgb: NDArray,
    semantic_id: int,
    instance_int_id: int
) -> None:
    """
    Store points with Pointcept-compatible structure in H5 database.
    
    Appends to existing instance data if present, otherwise creates new dataset.
    
    Args:
        h5_handle: Open H5 file handle
        instance_id: String identifier for instance (used as H5 key)
        new_points: (N, 3) array of XYZ coordinates
        new_colors_rgb: (N, 3) array of RGB colors normalized to [0, 1]
        semantic_id: Integer class ID from CLASS_TO_ID
        instance_int_id: Integer instance ID for Pointcept format
    """
    n_points = new_points.shape[0]
    semantic_col = np.full((n_points, 1), semantic_id, dtype=np.float32)
    instance_col = np.full((n_points, 1), instance_int_id, dtype=np.float32)
    
    new_data = np.hstack((
        new_points.astype(np.float32),
        new_colors_rgb.astype(np.float32),
        semantic_col,
        instance_col
    ))
    
    if instance_id in h5_handle:
        dset = h5_handle[instance_id]
        current_shape = dset.shape
        new_shape = (current_shape[0] + new_data.shape[0], 8)
        dset.resize(new_shape)
        dset[current_shape[0]:] = new_data
    else:
        h5_handle.create_dataset(
            instance_id,
            data=new_data,
            maxshape=(None, 8),
            chunks=True,
            compression='gzip',
            compression_opts=4
        )


# =============================================================================
# POINTCEPT EXPORT
# README: POINTCEPT INTEGRATION
# Exports to .pth format compatible with Pointcept framework
# =============================================================================

def export_pointcept_scene(
    h5_path: Path,
    output_path: Path,
    estimate_normals: bool = True,
    voxel_size: Optional[float] = None,
    remove_outliers: bool = True
) -> None:
    """
    Convert merged H5 database to Pointcept .pth format.
    
    Args:
        h5_path: Path to H5 geometry database
        output_path: Path for output .pth file
        estimate_normals: If True, estimate surface normals
        voxel_size: If provided, downsample to this voxel size (meters)
        remove_outliers: If True, apply statistical outlier removal
    """
    all_data = []
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("[cyan]Loading point cloud data...", total=None)
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                data = f[key][:]
                if data.shape[0] > 0:
                    all_data.append(data)
        progress.update(task, completed=True)
    
    if not all_data:
        console.print(f"[yellow]Warning:[/yellow] No data found in {h5_path}")
        return
    
    data = np.vstack(all_data)
    console.print(f"[dim]Total points before processing: {data.shape[0]:>12,}[/dim]")
    
    coord = data[:, :3].astype(np.float32)
    color = data[:, 3:6].astype(np.float32)
    segment = data[:, 6].astype(np.int64)
    instance = data[:, 7].astype(np.int64)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)
    
    if remove_outliers:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("[cyan]Removing outliers...", total=None)
            pcd, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            inlier_idx = np.array(inlier_idx)
            coord = np.asarray(pcd.points).astype(np.float32)
            color = np.asarray(pcd.colors).astype(np.float32)
            segment = segment[inlier_idx]
            instance = instance[inlier_idx]
            progress.update(task, completed=True)
        console.print(f"[dim]Points after outlier removal:   {coord.shape[0]:>12,}[/dim]")
    
    if voxel_size is not None and voxel_size > 0:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task(f"[cyan]Voxel downsampling ({voxel_size}m)...", total=None)
            pcd_down, _, idx_map = pcd.voxel_down_sample_and_trace(
                voxel_size=voxel_size,
                min_bound=pcd.get_min_bound() - voxel_size * 0.5,
                max_bound=pcd.get_max_bound() + voxel_size * 0.5
            )
            
            new_segment = []
            new_instance = []
            for indices in idx_map:
                indices = np.array(indices)
                if len(indices) > 0:
                    seg_votes = segment[indices]
                    new_segment.append(np.bincount(seg_votes.astype(int)).argmax())
                    new_instance.append(instance[indices[0]])
            
            coord = np.asarray(pcd_down.points).astype(np.float32)
            color = np.asarray(pcd_down.colors).astype(np.float32)
            segment = np.array(new_segment, dtype=np.int64)
            instance = np.array(new_instance, dtype=np.int64)
            pcd = pcd_down
            progress.update(task, completed=True)
        console.print(f"[dim]Points after downsampling:      {coord.shape[0]:>12,}[/dim]")
    
    if estimate_normals:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("[cyan]Estimating normal vectors...", total=None)
            try:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                centroid = pcd.get_center()
                pcd.orient_normals_towards_camera_location(centroid + np.array([0, 0, 10]))
                normal = np.asarray(pcd.normals).astype(np.float32)
                progress.update(task, completed=True)
                console.print("[dim]Normal estimation complete[/dim]")
            except RuntimeError as e:
                progress.update(task, completed=True)
                console.print(f"[yellow]Warning:[/yellow] Normal estimation failed ({e}), using zeros")
                normal = np.zeros_like(coord, dtype=np.float32)
    else:
        normal = np.zeros_like(coord, dtype=np.float32)
    
    scene_dict = {
        "coord": coord,
        "color": color,
        "normal": normal,
        "segment": segment,
        "instance": instance,
    }
    
    torch.save(scene_dict, output_path)
    
    console.print(f"[bold green]Exported Pointcept scene:[/bold green] {output_path}")
    console.print(f"[dim]  Points:                  {coord.shape[0]:>12,}[/dim]")
    console.print(f"[dim]  Unique semantic classes: {len(np.unique(segment)):>12}[/dim]")
    console.print(f"[dim]  Unique instances:        {len(np.unique(instance)):>12}[/dim]")


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def read_exr_to_numpy(exr_path: Path) -> Optional[NDArray]:
    """
    Read EXR file to numpy array.
    
    Args:
        exr_path: Path to EXR depth/position file
        
    Returns:
        (H, W, 3) numpy array or None if read fails
    """
    try:
        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        r, g, b = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
        return np.dstack((r.reshape(size), g.reshape(size), b.reshape(size)))
    except OpenEXR.Exception as e:
        console.print(f"[red]Error reading EXR {exr_path}: {e}[/red]")
        return None


def async_save_image(path: str, img: NDArray) -> None:
    """
    Save image (used for threaded async saves).
    
    Args:
        path: Output file path
        img: Image array in BGR format
    """
    cv2.imwrite(path, img)


def find_corresponding_exr(image_path: Path) -> Optional[Path]:
    """
    Find the corresponding EXR depth file for an RGB image.
    
    Expects directory structure: data/rgb/*.png -> data/global_xyz/*.exr
    
    Args:
        image_path: Path to RGB image
        
    Returns:
        Path to corresponding EXR file or None if not found
    """
    try:
        parts = image_path.parts
        data_index = parts.index('data')
        base_path = Path(*parts[:data_index+1])
        relative_path_parts = parts[data_index+2:]
        new_name = image_path.name.replace('_domain_rgb.png', '_domain_global_xyz.exr')
        return base_path / 'global_xyz' / Path(*relative_path_parts).with_name(new_name)
    except ValueError:
        return None


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for visualization.
    
    Args:
        num_colors: Number of unique colors needed
        
    Returns:
        List of BGR color tuples
    """
    np.random.seed(0)
    hsv_colors = [((180 * i) // num_colors, 255, 255) for i in range(num_colors)]
    bgr_colors = [cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0] for hsv in hsv_colors]
    np.random.shuffle(bgr_colors)
    return [tuple(map(int, color)) for color in bgr_colors]


def apply_mask_overlay(
    image: NDArray,
    mask: NDArray,
    color: Tuple[int, int, int],
    alpha: float = 0.5
) -> NDArray:
    """
    Apply colored mask overlay to image.
    
    Args:
        image: Input BGR image
        mask: Boolean mask array
        color: BGR color tuple
        alpha: Overlay opacity (0-1)
        
    Returns:
        Image with mask overlay applied
    """
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def get_conf_color(score: float) -> str:
    """
    Get Rich color string based on confidence score.
    
    Args:
        score: Confidence score (0-1)
        
    Returns:
        Rich color name string
    """
    if score >= 0.8:
        return "bright_green"
    if score >= 0.5:
        return "yellow"
    return "red"


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def calculate_iou(boxA: NDArray, boxB: NDArray) -> float:
    """
    Calculate intersection over union between two bounding boxes.
    
    Args:
        boxA: First box as [x1, y1, x2, y2]
        boxB: Second box as [x1, y1, x2, y2]
        
    Returns:
        IoU value (0-1)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_sam3_model(
    weights_dir: Path,
    device: str
) -> Tuple[Sam3Model, Sam3Processor]:
    """
    Load SAM3 model and processor.
    
    Attempts to load from local weights first, falls back to HuggingFace Hub.
    
    Args:
        weights_dir: Directory containing local model weights
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, processor)
    """
    local_model_path = weights_dir / "sam3"
    
    with suppress_output():
        if local_model_path.exists():
            console.print(f"[dim]Loading model from local weights: {local_model_path}[/dim]")
            sam3_model = Sam3Model.from_pretrained(str(local_model_path)).to(device)
            sam3_processor = Sam3Processor.from_pretrained(str(local_model_path))
        else:
            console.print(f"[dim]Downloading model from HuggingFace: {HUGGINGFACE_MODEL_ID}[/dim]")
            sam3_model = Sam3Model.from_pretrained(HUGGINGFACE_MODEL_ID).to(device)
            sam3_processor = Sam3Processor.from_pretrained(HUGGINGFACE_MODEL_ID)
    
    sam3_model.eval()
    if device == "cuda":
        sam3_model = sam3_model.to(torch.float16)
    
    return sam3_model, sam3_processor


# =============================================================================
# SCENE GRAPH MANAGEMENT
# =============================================================================

def create_fresh_scene_graph(area_name: str, floor_node_id: str) -> nx.Graph:
    """
    Create a new scene graph with building and floor structure.
    
    Args:
        area_name: Name of the area (used as building node ID)
        floor_node_id: ID for the floor node
        
    Returns:
        NetworkX graph with initial structure
    """
    area_scene_graph = nx.Graph()
    area_scene_graph.add_node(area_name, iso_class="Building", name=area_name)
    area_scene_graph.add_node(floor_node_id, iso_class="Floor", floor_number=1)
    area_scene_graph.add_edge(area_name, floor_node_id, relationship="composedOf")
    
    # Add surface boundary nodes for structural elements
    for surface_type in SURFACE_CLASSES:
        surface_node_id = f"{floor_node_id}_{surface_type}_surface"
        area_scene_graph.add_node(
            surface_node_id,
            label=surface_type,
            iso_class="CellSpaceBoundary",
            iso_superclass="SpaceBoundary",
            relevance="CONTEXT",
            surface_type=surface_type
        )
        area_scene_graph.add_edge(floor_node_id, surface_node_id, relationship="boundedBy")
    
    return area_scene_graph


def load_existing_scene_graph(
    graph_path: Path,
    area_name: str,
    floor_node_id: str
) -> nx.Graph:
    """
    Load existing scene graph or create fresh one if loading fails.
    
    Args:
        graph_path: Path to GraphML file
        area_name: Name of the area
        floor_node_id: ID for the floor node
        
    Returns:
        NetworkX graph (either loaded or freshly created)
    """
    try:
        area_scene_graph = nx.read_graphml(str(graph_path))
        console.print(f"[green]  Loaded scene graph with {area_scene_graph.number_of_nodes()} nodes[/green]")
        return area_scene_graph
    except (nx.NetworkXError, FileNotFoundError) as e:
        console.print(f"[red]  Failed to load scene graph: {e}[/red]")
        console.print("[yellow]  Creating fresh scene graph[/yellow]")
        return create_fresh_scene_graph(area_name, floor_node_id)


def load_instance_mapping(
    mapping_path: Path,
    h5_path: Path,
    scene_graph: nx.Graph
) -> Tuple[Dict[str, Dict], int, Dict[str, int]]:
    """
    Load instance mapping from JSON and recover centroids from H5.
    
    Args:
        mapping_path: Path to instance mapping JSON
        h5_path: Path to H5 geometry database
        scene_graph: Scene graph for checking node membership
        
    Returns:
        Tuple of (instance_manager, max_instance_id, string_to_int_mapping)
    """
    try:
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        instance_string_to_int = mapping_data.get('string_to_int', {})
        instance_info = mapping_data.get('instance_info', {})
        
        global_instance_manager = {}
        
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as h5_read:
                for instance_id, info in instance_info.items():
                    centroid = np.array([0.0, 0.0, 0.0])
                    
                    if instance_id in h5_read:
                        try:
                            data = h5_read[instance_id][:]
                            if data.shape[0] > 0:
                                points = data[:, :3]
                                centroid = np.mean(points, axis=0)
                        except (KeyError, ValueError):
                            pass
                    
                    global_instance_manager[instance_id] = {
                        'class_name': info.get('class', 'unknown'),
                        'global_id': instance_id,
                        'centroid_3d': centroid,
                        'exclude_from_graph': instance_id not in scene_graph.nodes()
                    }
            console.print(f"[green]  Loaded centroids from H5 for {len(global_instance_manager)} instances[/green]")
        else:
            for instance_id, info in instance_info.items():
                global_instance_manager[instance_id] = {
                    'class_name': info.get('class', 'unknown'),
                    'global_id': instance_id,
                    'centroid_3d': np.array([0.0, 0.0, 0.0]),
                    'exclude_from_graph': instance_id not in scene_graph.nodes()
                }
            console.print(f"[yellow]  Warning: H5 not found, using placeholder centroids[/yellow]")
        
        max_int_id = max(instance_string_to_int.values()) if instance_string_to_int else 0
        
        console.print(f"[green]  Loaded {len(instance_string_to_int)} instance mappings[/green]")
        console.print(f"[green]  Instance counter will continue from {max_int_id}[/green]")
        
        return global_instance_manager, max_int_id, instance_string_to_int
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        console.print(f"[red]  Failed to load instance mapping: {e}[/red]")
        console.print("[yellow]  Starting with fresh instance tracking[/yellow]")
        return {}, 0, {}


def refine_scene_graph_geometry(
    scene_graph: nx.Graph,
    h5_path: Path,
    building_node_id: str,
    floor_node_id: str
) -> int:
    """
    Update scene graph node geometry from accumulated H5 point data.
    
    Args:
        scene_graph: Scene graph to update (modified in place)
        h5_path: Path to H5 geometry database
        building_node_id: ID of building node (to skip)
        floor_node_id: ID of floor node (to skip)
        
    Returns:
        Number of nodes successfully refined
    """
    refined_count = 0
    nodes_to_process = [n for n in scene_graph.nodes() if n not in [building_node_id, floor_node_id]]
    
    with h5py.File(h5_path, 'r') as h5_handle:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Refining geometry...", total=len(nodes_to_process))
            
            for node_id in nodes_to_process:
                progress.update(task, advance=1)
                
                if node_id not in h5_handle:
                    continue
                    
                data = h5_handle[node_id][:]
                points = data[:, :3]
                
                if points.shape[0] < MIN_POINTS_FOR_INSTANCE:
                    continue
                
                try:
                    centroid = np.mean(points, axis=0)
                    
                    scene_graph.nodes[node_id]['cx'] = float(centroid[0])
                    scene_graph.nodes[node_id]['cy'] = float(centroid[1])
                    scene_graph.nodes[node_id]['cz'] = float(centroid[2])
                    
                    refined_count += 1
                    
                except RuntimeError as e:
                    console.print(f"[yellow]Warning: Failed to refine {node_id}: {e}[/yellow]")
    
    return refined_count


# =============================================================================
# CORE PROCESSING
# =============================================================================

class Detection(NamedTuple):
    """Single object detection result."""
    box: NDArray
    label: str
    score: float
    mask: NDArray
    source: str


def process_single_frame(
    image_path: Path,
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    device: str,
    detection_classes: List[str],
    prompt_to_class: Dict[str, str],
    vis_color_map: Dict[str, Tuple[int, int, int]],
    o3d_color_map: Dict[str, Tuple[float, float, float]],
    area_name: str,
    log_writer: csv.writer,
    frame_index: int = 0,
    output_dir: Optional[Path] = None,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD
) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[NDArray], List[Dict[str, Any]], 
           Optional[List[Dict[str, Any]]], Optional[NDArray], Optional[NDArray], Dict[str, int]]:
    """
    Process a single frame for object detection and 3D extraction.
    
    Args:
        image_path: Path to input RGB image
        sam3_model: Loaded SAM3 model
        sam3_processor: SAM3 processor for tokenization
        device: Target device ('cuda' or 'cpu')
        detection_classes: List of class names to detect
        prompt_to_class: Mapping from prompt text to class name
        vis_color_map: BGR colors for visualization
        o3d_color_map: RGB colors for Open3D
        area_name: Name of the area being processed
        log_writer: CSV writer for detection logging
        frame_index: Index of current frame (for memory management)
        output_dir: Optional output directory (unused, kept for API compatibility)
        conf_threshold: Minimum confidence for detections
        mask_threshold: Threshold for mask binarization
        
    Returns:
        Tuple of (point_cloud, annotated_image, frame_objects, serializable_detections,
                  masks, xyz_data, frame_stats)
    """
    frame_stats = {
        'classes_queried': len(detection_classes),
        'sam3_detections': 0,
        'final_detections': 0
    }

    exr_path = find_corresponding_exr(image_path)
    if not exr_path or not exr_path.exists():
        return None, None, [], None, None, None, frame_stats

    xyz_data = read_exr_to_numpy(exr_path)
    image_bgr = cv2.imread(str(image_path))
    
    if xyz_data is None or image_bgr is None:
        return None, None, [], None, None, None, frame_stats
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    h, w, _ = image_bgr.shape
    
    def log_csv(detection: Detection, status: str):
        log_writer.writerow([
            image_path.name, detection.source, detection.label,
            f"{detection.score:.4f}", detection.box.tolist(), status
        ])

    all_detections: List[Detection] = []
    prompts_list = list(prompt_to_class.keys())
    
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(device=='cuda')):
        try:
            base_image_inputs = sam3_processor.image_processor(images=pil_image, return_tensors="pt")
            original_sizes = base_image_inputs.get("original_sizes")
            
            for i in range(0, len(prompts_list), TEXT_QUERY_BATCH_SIZE):
                prompt_chunk = prompts_list[i : i + TEXT_QUERY_BATCH_SIZE]
                current_batch_len = len(prompt_chunk)

                text_inputs = sam3_processor.tokenizer(
                    prompt_chunk, padding="max_length", max_length=CLIP_MAX_LENGTH,
                    truncation=True, return_tensors="pt"
                )

                current_image_inputs = {}
                for k, v in base_image_inputs.items():
                    if isinstance(v, torch.Tensor):
                        current_image_inputs[k] = v.repeat(current_batch_len, *([1] * (v.ndim - 1)))
                    else:
                        current_image_inputs[k] = v

                if isinstance(original_sizes, torch.Tensor):
                    batch_original_sizes = original_sizes.repeat(current_batch_len, 1)
                else:
                    batch_original_sizes = [original_sizes[0]] * current_batch_len

                inputs = {**current_image_inputs, **text_inputs}
                model_inputs = {
                    k: v.to(device) for k, v in inputs.items()
                    if k in ['pixel_values', 'input_ids', 'attention_mask', 'pixel_mask']
                }
                
                outputs = sam3_model(**model_inputs)
                
                results = sam3_processor.post_process_instance_segmentation(
                    outputs, threshold=conf_threshold, mask_threshold=mask_threshold,
                    target_sizes=batch_original_sizes.tolist() if hasattr(batch_original_sizes, 'tolist') else batch_original_sizes
                )
                
                for batch_idx, result_dict in enumerate(results):
                    prompt_text = prompt_chunk[batch_idx]
                    class_label = prompt_to_class.get(prompt_text, prompt_text)
                    
                    masks = result_dict.get('masks', None)
                    scores = result_dict.get('scores', None)
                    boxes = result_dict.get('boxes', None)

                    if masks is not None and len(masks) > 0:
                        for det_i in range(len(masks)):
                            score = float(scores[det_i].float().cpu())
                            if score < conf_threshold:
                                continue
                            
                            mask_torch = masks[det_i]
                            box = boxes[det_i].float().cpu().numpy()
                            
                            if mask_torch.shape[0] != h or mask_torch.shape[1] != w:
                                mask = mask_torch.float().cpu().numpy()
                                mask = cv2.resize(
                                    mask.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                            else:
                                mask = mask_torch.cpu().numpy().astype(bool)
                            
                            box_xyxy = np.array([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                            det = Detection(box_xyxy, class_label, score, mask, 'sam3')
                            all_detections.append(det)
                            log_csv(det, "DETECTED_RAW")
                
                del inputs, model_inputs, outputs, results
                
                if device == 'cuda':
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            console.print(f"[bold red]Inference Error: {e}[/bold red]")
            return None, None, [], None, None, None, frame_stats

    clear_gpu_memory()
    frame_stats['sam3_detections'] = len(all_detections)
    
    if not all_detections:
        return None, image_bgr, [], None, None, xyz_data, frame_stats

    # Non-maximum suppression by IoU
    final_detections: List[Detection] = []
    final_boxes: List[NDArray] = []
    all_detections.sort(key=lambda x: x.score, reverse=True)
    
    for det in all_detections:
        is_duplicate = False
        for final_box in final_boxes:
            if calculate_iou(det.box, final_box) > DEFAULT_IOU_THRESHOLD:
                is_duplicate = True
                log_csv(det, "DISCARDED_DUPLICATE")
                break
        
        if not is_duplicate:
            final_detections.append(det)
            final_boxes.append(det.box)
            log_csv(det, "KEPT_FINAL")

    frame_stats['final_detections'] = len(final_detections)
    
    if not final_detections:
        return None, image_bgr, [], None, None, xyz_data, frame_stats

    annotated_image = image_bgr.copy()
    frame_objects_data = []
    serializable_detections = []
    all_masks = []
    
    for detection in final_detections:
        label = detection.label
        bbox = detection.box
        mask = detection.mask
        all_masks.append(mask)
        
        # Annotate image
        color = vis_color_map.get(label, (0, 0, 255))
        annotated_image = apply_mask_overlay(annotated_image, mask, color)
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(
            annotated_image, f"{label} {detection.score:.2f}",
            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        # Extract 3D points
        points_3d = xyz_data[mask]
        valid_mask = np.any(points_3d != 0, axis=1)
        valid_points = points_3d[valid_mask]

        original_rgb = image_rgb[mask]
        original_rgb_valid = original_rgb[valid_mask]
        colors_rgb_normalized = original_rgb_valid.astype(np.float32) / 255.0
        
        semantic_id = CLASS_TO_ID.get(label, -1)

        serializable_detections.append({
            "box": detection.box.tolist(),
            "label": detection.label,
            "score": float(detection.score),
            "source": detection.source
        })

        if valid_points.shape[0] > MIN_POINTS_FOR_INSTANCE:
            o3d_color = o3d_color_map.get(label, (1.0, 0.0, 0.0))
            colors_semantic = np.tile(o3d_color, (valid_points.shape[0], 1))
            
            try:
                centroid = np.mean(valid_points, axis=0)
                box_center = str(centroid.tolist())
            except RuntimeError:
                box_center = "[0,0,0]"

            class_info = CLASS_REGISTRY.get(label, {
                "iso_class": "Unknown",
                "iso_superclass": "Unknown",
                "relevance": "CONTEXT",
                "responder_priority": "LOW"
            })
            
            obj_data = {
                "class_name": label,
                "semantic_id": semantic_id,
                "source_model": "sam3",
                "exclude_from_graph": label in SCENE_GRAPH_EXCLUDED_CLASSES,
                "confidence": float(detection.score),
                "iso_class": class_info["iso_class"],
                "iso_superclass": class_info["iso_superclass"],
                "public_safety_relevance": class_info["relevance"],
                "responder_priority": class_info.get("responder_priority", "LOW"),
                "building": area_name,
                "source_file": str(image_path),
                "box_center": box_center,
                "points_3d": valid_points,
                "colors_rgb": colors_rgb_normalized,
                "colors_semantic": colors_semantic,
            }
            frame_objects_data.append(obj_data)
    
    sam_masks = np.array(all_masks) if all_masks else np.array([], dtype=bool)
    
    return None, annotated_image, frame_objects_data, serializable_detections, sam_masks, xyz_data, frame_stats


# =============================================================================
# ARGUMENT PARSING
# README: COMMAND LINE OPTIONS
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="SAM3 3D Processor - Public Safety Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python run.py /datasets/area_1
  
  # Verify checkpoint integrity
  python run.py /datasets/area_1 --verify-checkpoint --dry-run
  
  # Fix checkpoint and reprocess missing images
  python run.py /datasets/area_1 --reprocess-missing
  
  # Export to Pointcept format
  python run.py /datasets/area_1 --export-pointcept
  
  # Custom output directory
  python run.py /datasets/area_1 --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        "area_path",
        type=Path,
        help="Path to the area directory containing data/rgb/ images"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help=f"Directory containing model weights (default: {DEFAULT_WEIGHTS_DIR})"
    )
    
    parser.add_argument(
        "--verify-checkpoint",
        action="store_true",
        help="Verify checkpoint against actual outputs and remove entries for missing outputs"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --verify-checkpoint, only show what would be removed"
    )
    
    parser.add_argument(
        "--reprocess-missing",
        action="store_true",
        help="Verify checkpoint, remove bad entries, and reprocess missing images"
    )
    
    parser.add_argument(
        "--export-pointcept",
        action="store_true",
        help="Export to Pointcept .pth format after processing"
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for detections (default: {DEFAULT_CONFIDENCE_THRESHOLD})"
    )
    
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size for downsampling (meters). If not set, no downsampling is performed."
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_area(
    area_path: Path,
    output_dir: Path,
    weights_dir: Path,
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    device: str,
    verify_mode: bool = False,
    export_pointcept: bool = False,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    voxel_size: Optional[float] = None
) -> None:
    """
    Process all images in an area directory.
    
    Args:
        area_path: Path to area directory
        output_dir: Base output directory
        weights_dir: Model weights directory
        sam3_model: Loaded SAM3 model
        sam3_processor: SAM3 processor
        device: Target device
        verify_mode: If True, verify and clean checkpoint first
        export_pointcept: If True, export to Pointcept format after processing
        conf_threshold: Confidence threshold for detections
        voxel_size: Optional voxel size for downsampling
    """
    area_name = area_path.name
    paths_in_area = sorted(area_path.glob("data/rgb/*.[pP][nN][gG]"))
    
    if not paths_in_area:
        console.print(f"[red]No images found in {area_path}/data/rgb/[/red]")
        return
    
    console.print(f"\n[bold]Found {len(paths_in_area):,} input images in {area_name}[/bold]")
    
    # Setup output directories
    area_output_dir = output_dir / area_name
    area_output_dir.mkdir(exist_ok=True)
    results_dir = area_output_dir / "results"
    checkpoint_path = area_output_dir / "processing_checkpoint.json"
    
    # Recover checkpoint
    if verify_mode:
        console.print("[bold yellow]Verify mode:[/bold yellow] checking checkpoint against outputs...")
    
    processed_images = recover_checkpoint_with_output_scan(
        checkpoint_path, results_dir, paths_in_area, verify_mode=verify_mode
    )
    
    if processed_images:
        console.print(f"[bold yellow]Resuming:[/bold yellow] Found {len(processed_images):,} previously processed images")
    
    images_to_process = [p for p in paths_in_area if p.stem not in processed_images]
    
    if not images_to_process and (area_output_dir / f"{area_name}_scene_graph.graphml").exists():
        console.print("[green]All images already processed.[/green]")
        return
    
    console.print(f"[bold]{len(images_to_process):,} images to process[/bold]")
    console.print(f"[dim]({len(processed_images):,} already done, {len(images_to_process):,} remaining)[/dim]")
    
    # Setup color maps
    prompt_to_class = {prompt: cls for cls, prompt in SAM3_PRIMARY_PROMPT.items()}
    o3d_color_map = {label: get_semantic_color(label) for label in DETECTION_CLASSES}
    vis_color_map = {
        label: (int(c[2]*255), int(c[1]*255), int(c[0]*255))
        for label, c in o3d_color_map.items()
    }
    
    # Setup paths
    h5_database_path = area_output_dir / "geometry_database.h5"
    graph_output_path = area_output_dir / f"{area_name}_scene_graph.graphml"
    instance_mapping_path = area_output_dir / "instance_mapping.json"
    detection_log_path = output_dir / "model_detection_log.csv"
    
    building_node_id = area_name
    floor_node_id = f"{area_name}_floor_1"
    
    # Clear H5 if starting fresh
    if not processed_images and h5_database_path.exists():
        os.remove(h5_database_path)
    
    # Load or create scene graph and instance tracking
    if processed_images and graph_output_path.exists() and instance_mapping_path.exists():
        console.print("[cyan]Loading existing scene graph and instance data...[/cyan]")
        area_scene_graph = load_existing_scene_graph(graph_output_path, area_name, floor_node_id)
        global_instance_manager, global_instance_counter, instance_string_to_int = load_instance_mapping(
            instance_mapping_path, h5_database_path, area_scene_graph
        )
    else:
        console.print("[dim]Creating fresh scene graph and instance tracking...[/dim]")
        area_scene_graph = create_fresh_scene_graph(area_name, floor_node_id)
        global_instance_manager = {}
        global_instance_counter = 0
        instance_string_to_int = {}
    
    image_saver_threads = []
    
    with open(detection_log_path, 'w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["image_path", "source_model", "detected_label", "confidence", "box", "status"])
        
        console.rule(f"[bold magenta]AREA: {area_name}[/bold magenta]")
        
        with h5py.File(h5_database_path, 'a') as h5_handle:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None, complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=10
            ) as progress:
                
                skipped_count = len(paths_in_area) - len(images_to_process)
                
                if skipped_count > 0:
                    console.print(f"[dim]Skipping {skipped_count:,} already-processed images[/dim]")
                
                main_task = progress.add_task(
                    f"Processing...", total=len(paths_in_area), completed=skipped_count
                )
                
                for i, image_path in enumerate(paths_in_area):
                    image_stem = image_path.stem
                    if image_stem in processed_images:
                        continue
                    
                    # Clean up completed threads
                    image_saver_threads = [t for t in image_saver_threads if t.is_alive()]

                    frame_pcd, annotated_image, frame_objects, \
                    frame_detections_2d, frame_masks, xyz_data, frame_stats = process_single_frame(
                        image_path, sam3_model, sam3_processor, device,
                        DETECTION_CLASSES, prompt_to_class, vis_color_map, o3d_color_map,
                        area_name, log_writer, frame_index=i, output_dir=output_dir,
                        conf_threshold=conf_threshold
                    )
                    
                    progress.update(main_task, advance=1)
                    
                    if frame_objects:
                        results_base_dir = area_output_dir / "results"
                        frame_output_name = image_path.stem.replace('_domain_rgb', '')
                        img_out_path = results_base_dir / f"{frame_output_name}.png"

                        for obj in frame_objects:
                            obj['annotated_file_path'] = str(img_out_path)

                        for new_detection in frame_objects:
                            try:
                                new_centroid = np.array(json.loads(new_detection['box_center']))
                            except (json.JSONDecodeError, KeyError):
                                continue
                            
                            new_pts = new_detection['points_3d']
                            new_colors_rgb = new_detection['colors_rgb']
                            semantic_id = new_detection['semantic_id']
                            
                            # Handle surface classes specially
                            if new_detection['class_name'] in SURFACE_CLASSES:
                                surface_node_id = f"{floor_node_id}_{new_detection['class_name']}_surface"
                                update_instance_h5_pointcept(
                                    h5_handle, surface_node_id, new_pts, new_colors_rgb,
                                    semantic_id, SURFACE_CLASS_INSTANCE_IDS[new_detection['class_name']]
                                )
                                continue
                            
                            # Find matching existing instance
                            best_match_id = None
                            min_distance = float('inf')
                            
                            for global_id, existing_instance in global_instance_manager.items():
                                if existing_instance['class_name'] != new_detection['class_name']:
                                    continue
                                distance = np.linalg.norm(new_centroid - existing_instance['centroid_3d'])
                                if distance < min_distance and distance < INSTANCE_MERGE_DISTANCE_M:
                                    min_distance = distance
                                    best_match_id = global_id

                            if best_match_id is not None:
                                # Merge with existing instance
                                instance_int_id = instance_string_to_int[best_match_id]
                                update_instance_h5_pointcept(
                                    h5_handle, best_match_id, new_pts, new_colors_rgb,
                                    semantic_id, instance_int_id
                                )
                            else:
                                # Create new instance
                                global_instance_counter += 1
                                new_global_id = f"{area_name}_instance_{global_instance_counter}"
                                
                                instance_string_to_int[new_global_id] = global_instance_counter
                                
                                global_instance_manager[new_global_id] = {
                                    'class_name': new_detection['class_name'],
                                    'global_id': new_global_id,
                                    'centroid_3d': new_centroid,
                                    'exclude_from_graph': new_detection.get('exclude_from_graph', False),
                                }
                                
                                update_instance_h5_pointcept(
                                    h5_handle, new_global_id, new_pts, new_colors_rgb,
                                    semantic_id, global_instance_counter
                                )
                                
                                # Add to scene graph
                                if not new_detection.get('exclude_from_graph', False):
                                    graph_attrs = {
                                        "label": new_detection['class_name'],
                                        "iso_class": new_detection['iso_class'],
                                        "iso_superclass": new_detection['iso_superclass'],
                                        "relevance": new_detection['public_safety_relevance'],
                                        "responder_priority": new_detection.get('responder_priority', 'LOW'),
                                        "confidence": float(new_detection['confidence']),
                                        "cx": float(new_centroid[0]),
                                        "cy": float(new_centroid[1]),
                                        "cz": float(new_centroid[2]),
                                        "source": new_detection['source_model']
                                    }
                                    area_scene_graph.add_node(new_global_id, **graph_attrs)
                                    area_scene_graph.add_edge(
                                        floor_node_id, new_global_id, relationship="containsFeature"
                                    )

                        # Save annotated image asynchronously
                        if annotated_image is not None:
                            results_base_dir.mkdir(exist_ok=True)
                            t = threading.Thread(
                                target=async_save_image, args=(str(img_out_path), annotated_image)
                            )
                            t.start()
                            image_saver_threads.append(t)

                    del frame_pcd, annotated_image, frame_objects, frame_detections_2d, frame_masks, xyz_data
                    
                    append_to_checkpoint(checkpoint_path, image_stem, processed_images)
                    periodic_cleanup(i)
                    
                    if i > 0 and i % H5_FLUSH_INTERVAL == 0:
                        h5_handle.flush()

    # Wait for all image saves to complete
    for t in image_saver_threads:
        t.join()
    
    consolidate_checkpoint(checkpoint_path, processed_images)

    console.print()
    console.print("[bold green]Processing Complete[/bold green]")

    # Refine geometry
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[yellow]Running Geometry Refinement...", total=None)
        refined_count = refine_scene_graph_geometry(area_scene_graph, h5_database_path, building_node_id, floor_node_id)
        progress.update(task, completed=True)
    console.print(f"[green]Refined geometry for {refined_count:,} instances[/green]")

    # Save scene graph
    nx.write_graphml(area_scene_graph, str(graph_output_path))
    console.print(f"[green]Scene Graph Saved:[/green] {graph_output_path}")
    
    # Export to Pointcept if requested
    if export_pointcept:
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Exporting to Pointcept format...", total=None)
            pointcept_dir = output_dir / "pointcept"
            pointcept_dir.mkdir(exist_ok=True)
            pointcept_output_path = pointcept_dir / f"{area_name}.pth"
            export_pointcept_scene(
                h5_path=h5_database_path,
                output_path=pointcept_output_path,
                estimate_normals=True,
                voxel_size=voxel_size,
                remove_outliers=True
            )
            progress.update(task, completed=True)
        console.print(f"[green]Pointcept export complete:[/green] {pointcept_output_path}")
    else:
        console.print("[yellow]Pointcept export skipped[/yellow] [dim](use --export-pointcept to enable)[/dim]")
        console.print(f"[yellow]H5 database saved:[/yellow] {h5_database_path}")
    
    # Save instance mapping
    with open(instance_mapping_path, 'w') as f:
        json.dump({
            "string_to_int": instance_string_to_int,
            "instance_info": {k: {"class": v["class_name"]} for k, v in global_instance_manager.items()}
        }, f, indent=2)
    console.print(f"[green]Instance mapping saved:[/green] {instance_mapping_path}")
    
    # Print summary table
    console.print()
    console.rule("[bold cyan]Processing Summary[/bold cyan]")
    summary_table = Table(box=box.SIMPLE)
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="green", justify="right")
    
    summary_table.add_row("Area", area_name)
    summary_table.add_row("Images Processed", f"{len(processed_images):,}")
    summary_table.add_row("Scene Graph Nodes", f"{area_scene_graph.number_of_nodes():,}")
    summary_table.add_row("Unique Instances", f"{len(global_instance_manager):,}")
    summary_table.add_row("Geometry Refined", f"{refined_count:,}")
    
    console.print(summary_table)
    console.print()
    console.rule("[bold green]Complete[/bold green]")


SCRIPT_VERSION = "4.3.2-PUBLIC-SAFETY-RESUME-FIX"


def main() -> None:
    """Main entry point for SAM3 3D Processor."""
    args = parse_args()
    
    # Setup directories
    output_dir = args.output_dir
    weights_dir = args.weights_dir
    output_dir.mkdir(exist_ok=True)
    
    pointcept_dir = output_dir / "pointcept"
    pointcept_dir.mkdir(exist_ok=True)
    
    # Print header
    console.print()
    console.rule("[bold blue]SAM3 INFERENCE - PUBLIC SAFETY EDITION[/bold blue]")
    console.print(f"[bold cyan]Script Version:[/bold cyan] {SCRIPT_VERSION}")
    console.print(f"[bold cyan]Detection Classes:[/bold cyan] {NUM_CLASSES} (person detection DISABLED)")
    
    # Export class mapping
    export_class_mapping(pointcept_dir / "class_mapping.json")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[bold cyan]Device:[/bold cyan] {device.upper()}")
        
        area_path = args.area_path
        
        # Handle verify-checkpoint mode
        if args.verify_checkpoint:
            paths_in_area = sorted(area_path.glob("data/rgb/*.[pP][nN][gG]"))
            area_output_dir = output_dir / area_path.name
            results_dir = area_output_dir / "results"
            checkpoint_path = area_output_dir / "processing_checkpoint.json"
            
            verified, missing = verify_checkpoint_against_outputs(
                checkpoint_path, results_dir, paths_in_area, dry_run=args.dry_run
            )
            
            if args.dry_run:
                console.print("\n[cyan]To apply changes, run without --dry-run[/cyan]")
            else:
                images_to_process = [p for p in paths_in_area if p.stem not in verified]
                if images_to_process:
                    console.print(f"\n[yellow]To reprocess the {len(images_to_process)} missing images, run:[/yellow]")
                    console.print(f"[dim]  python run.py {area_path} --reprocess-missing[/dim]")
            return
        
        # Load model with spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task1 = progress.add_task("[cyan]Loading SAM3 model...", total=None)
            sam3_model, sam3_processor = load_sam3_model(weights_dir, device)
            progress.update(task1, completed=True)
        console.print("[green]Model loaded successfully[/green]")
        
        # Process area
        process_area(
            area_path=area_path,
            output_dir=output_dir,
            weights_dir=weights_dir,
            sam3_model=sam3_model,
            sam3_processor=sam3_processor,
            device=device,
            verify_mode=args.reprocess_missing,
            export_pointcept=args.export_pointcept,
            conf_threshold=args.conf_threshold,
            voxel_size=args.voxel_size
        )

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Processing interrupted by user[/yellow]")
        console.print("[dim]Progress has been checkpointed. Run again to resume.[/dim]")
    except Exception as e:
        console.print(f"[bold red]FATAL ERROR:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()