#!/usr/bin/env python3
"""
Mask Editor Application - YTVIS-19 Format
GUI tool for editing segmentation masks with multi-video support
Enhanced with new annotation creation and AI assist features
"""

import argparse
import json
import os
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pycocotools import mask as maskUtils
from typing import Dict, List, Optional, Tuple
import glob
import torch
import tempfile
from sam2.build_sam import build_sam2_video_predictor


def mask_to_bbox(mask):
    """
    mask: (H, W) binary mask (0 or 1)
    returns: (x_min, y_min, x_max, y_max)
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None  # empty mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return int(x_min), int(y_min), int(x_max), int(y_max)

def predict_masks_with_sam2(
    frames: List[np.ndarray],
    mask: np.ndarray,
    current_frame_idx: int,
    checkpoint: str,
    model_cfg: str,
    device: str = "cuda"
) -> Dict:
    """
    Use SAM2 video predictor to track a single instance across all frames
    Args:
        frames: List of numpy arrays (H, W, 3) representing video frames
        mask: Initial mask (H, W) for the instance in the first frame
        checkpoint: Path to SAM2 checkpoint
        model_cfg: Path to SAM2 model config
        device: Device to run on
    Returns:
        Dictionary with structure:
        {
            'masks': {frame_idx: mask_array},
            'boxes': {frame_idx: box},
            'image_shape': (height, width)
        }
    """
    print(f"Loading SAM2 model from {checkpoint}...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    
    # Get dimensions from first frame
    h, w = frames[0].shape[:2]
    
    results = {
        'masks': {},
        'boxes': {},
        'image_shape': (h, w)
    }
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save frames to temporary directory (SAM2 requires file paths)
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"{current_frame_idx+i:05d}.jpg")
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)
        
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            # Initialize state
            inference_state = predictor.init_state(video_path=temp_dir)
            
            print(f"Adding mask prompt for instance")
            
            # Add mask prompt to SAM2
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=current_frame_idx,
                obj_id=0,
                mask=mask
            )
            
            # Propagate through video
            print("\nPropagating masks through video...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask_logit = out_mask_logits[0].squeeze(0)
                mask = (mask_logit > 0).cpu().numpy().astype(np.uint8)
                
                results['masks'][out_frame_idx] = mask
                results['boxes'][out_frame_idx] = mask_to_bbox(mask)
    
    return results


class MaskEditorApp:
    """GUI application for editing masks with zoom/pan support and YTVIS-19 format"""
    
    def __init__(self, root, root_dir: str, annotations_path: str, model_cfg=None, model_chkpt=None):
        self.root = root
        self.root.title("Mask Editor - YTVIS-19 Annotation Tool")
        self.root_dir = root_dir
        self.annotations_path = annotations_path
        self.model_cfg = model_cfg  # SAM2 model config for AI assist
        self.model_chkpt = model_chkpt  # SAM2 model checkpoint for AI assist
        
        # Load YTVIS annotations
        self.load_ytvis_annotations()
        
        # Discover all videos in root_dir
        self.discover_videos()
        
        # Track which videos have been modified (need saving)
        self.modified_videos = set()
        
        # Current state
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_instance_id = None
        
        # Initialize current video
        self.set_current_video(0)
        
        # Drawing state
        self.drawing = False
        self.brush_size = 20
        self.mode = 'add'
        self.last_draw_pos = None
        self.tool = 'brush'
        self.polygon_points = []
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Image buffering
        self.image_cache = {}
        self.overlay_buffer = None
        self.display_buffer = None
        self.last_display_params = None
        self.base_image = None
        self.update_pending = False
        
        # Image adjustment parameters
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.show_image = True
        self.show_mask = True
        
        # Create UI
        self.create_ui()
        self.update_display()
    
    def discover_videos(self):
        """Discover all video directories in root_dir"""
        self.all_video_dirs = []
        
        # Find all subdirectories in root_dir that contain image files
        for item in sorted(os.listdir(self.root_dir)):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(glob.glob(os.path.join(item_path, ext)))
                
                if image_files:
                    self.all_video_dirs.append(item)
        
        # Merge with existing videos in annotations
        self.all_videos = []
        self.video_name_to_idx = {}
        
        # Add existing annotated videos
        for idx, video in enumerate(self.videos):
            video_name = video['file_names'][0].split('/')[0]
            self.all_videos.append({
                'name': video_name,
                'is_annotated': True,
                'video_data': video,
                'original_idx': idx
            })
            self.video_name_to_idx[video_name] = len(self.all_videos) - 1
        
        # Add discovered videos that aren't yet annotated
        for video_dir in self.all_video_dirs:
            if video_dir not in self.video_name_to_idx:
                self.all_videos.append({
                    'name': video_dir,
                    'is_annotated': False,
                    'video_data': None,
                    'original_idx': None
                })
                self.video_name_to_idx[video_dir] = len(self.all_videos) - 1
        
    def load_ytvis_annotations(self):
        """Load annotations from YTVIS-19 format JSON"""
        with open(self.annotations_path, 'r') as f:
            self.ytvis_data = json.load(f)
        
        # Store categories
        self.categories = self.ytvis_data['categories']
        
        # Store videos
        self.videos = self.ytvis_data['videos']
        
        # Store annotations
        self.annotations = self.ytvis_data['annotations']
        
        # Build video ID to index mapping
        self.video_id_to_idx = {video['id']: idx for idx, video in enumerate(self.videos)}
        
        # Build category ID to index mapping
        self.category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        
        # Track next IDs for new annotations
        self.next_video_id = max([v['id'] for v in self.videos], default=0) + 1
        self.next_annotation_id = max([a['id'] for a in self.annotations], default=0) + 1
        
    def create_new_video_annotation(self, video_name: str):
        """Create new annotation structure for an unannotated video"""
        video_dir = os.path.join(self.root_dir, video_name)
        
        # Get all image files in the directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(video_dir, ext)))
        
        image_files = sorted(image_files)
        
        if not image_files:
            messagebox.showerror("Error", f"No image files found in {video_dir}")
            return False
        
        # Ask user for start and end frames
        dialog = FrameRangeDialog(self.root, len(image_files))
        if not dialog.result:
            return False
        
        start_frame, end_frame = dialog.result
        
        # Validate range
        if start_frame < 0 or end_frame >= len(image_files) or start_frame > end_frame:
            messagebox.showerror("Error", f"Invalid frame range: {start_frame} to {end_frame}")
            return False
        
        # Get image dimensions from first frame
        first_img = Image.open(image_files[start_frame])
        width, height = first_img.size
        
        # Create file_names list
        file_names = [os.path.join(video_name, os.path.basename(f)) for f in image_files[start_frame:end_frame+1]]
        
        # Create new video entry
        video_id = self.next_video_id
        self.next_video_id += 1
        
        video_data = {
            'id': video_id,
            'width': width,
            'height': height,
            'length': end_frame - start_frame + 1,
            'file_names': file_names
        }
        
        # Add to videos list
        self.videos.append(video_data)
        self.video_id_to_idx[video_id] = len(self.videos) - 1
        
        # Update all_videos entry
        for vid in self.all_videos:
            if vid['name'] == video_name:
                vid['is_annotated'] = True
                vid['video_data'] = video_data
                vid['original_idx'] = len(self.videos) - 1
                break
        
        # Mark this video as modified
        self.modified_videos.add(video_id)
        
        messagebox.showinfo("Success", f"Created annotation structure for {video_name}\nFrames: {start_frame} to {end_frame}")
        return True
        
    def set_current_video(self, video_idx):
        """Set the current video and load its data"""
        if video_idx >= len(self.all_videos):
            return
        
        video_info = self.all_videos[video_idx]
        
        # Check if video is annotated
        if not video_info['is_annotated']:
            response = messagebox.askyesno(
                "Create Annotations",
                f"Video '{video_info['name']}' has no annotations.\nWould you like to create annotations for it?"
            )
            if response:
                if self.create_new_video_annotation(video_info['name']):
                    # Reload the video_info after creation
                    video_info = self.all_videos[video_idx]
                else:
                    return
            else:
                return
        
        self.current_video_idx = video_idx
        current_video = video_info['video_data']
        self.current_video_id = current_video['id']
        self.video_name = current_video['file_names'][0].split('/')[0]
        
        # Get video dimensions (from first frame if needed)
        self.video_width = current_video['width']
        self.video_height = current_video['height']
        self.image_shape = (self.video_height, self.video_width, 3)
        
        # Build frame paths
        frame_dir = os.path.join(self.root_dir, self.video_name)
        self.frame_paths = current_video['file_names']
        
        # Get annotations for this video
        self.video_annotations = [ann for ann in self.annotations if ann['video_id'] == self.current_video_id]
        
        # Build masks and boxes structure for this video
        self.masks = {}
        self.boxes = {}
        self.instance_metadata = {}
        
        for ann in self.video_annotations:
            inst_id = ann['id']
            category_id = ann['category_id']
            
            # Find category name
            cat_name = next((cat['name'] for cat in self.categories if cat['id'] == category_id), 'unknown')
            
            self.instance_metadata[str(inst_id)] = {
                'category_id': category_id,
                'category_name': cat_name
            }
            
            self.masks[inst_id] = {}
            self.boxes[str(inst_id)] = {}
            
            # Decode segmentations
            segmentations = ann['segmentations']
            bboxes = ann['bboxes']
            
            for frame_idx, (seg, bbox) in enumerate(zip(segmentations, bboxes)):
                if seg is not None:
                    # Decode RLE
                    if isinstance(seg, dict):
                        rle = seg.copy()
                        if isinstance(rle['counts'], str):
                            rle['counts'] = rle['counts'].encode('utf-8')
                        mask = maskUtils.decode(rle)
                        self.masks[inst_id][frame_idx] = mask.astype(bool)
                    else:
                        # Polygon format - convert to mask
                        from PIL import Image, ImageDraw
                        mask_img = Image.new('L', (self.video_width, self.video_height), 0)
                        draw = ImageDraw.Draw(mask_img)
                        for polygon in seg:
                            points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                            draw.polygon(points, outline=1, fill=1)
                        self.masks[inst_id][frame_idx] = np.array(mask_img, dtype=bool)
                else:
                    self.masks[inst_id][frame_idx] = None
                
                # Store bbox
                if bbox is not None:
                    # YTVIS format: [x, y, width, height]
                    # Convert to [x_min, y_min, x_max, y_max]
                    x, y, w, h = bbox
                    self.boxes[str(inst_id)][str(frame_idx)] = [int(x), int(y), int(x + w), int(y + h)]
                else:
                    self.boxes[str(inst_id)][str(frame_idx)] = None
        
        # Get instance IDs
        self.instance_ids = sorted([ann['id'] for ann in self.video_annotations])
        if self.instance_ids:
            self.current_instance_id = self.instance_ids[0]
        else:
            self.current_instance_id = None
        
        # Reset frame index
        self.current_frame_idx = 0
        
        # Clear caches
        self.image_cache = {}
        self.invalidate_buffer()
        
    def create_ui(self):
        """Create the user interface"""
        # Top frame with controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Video selection
        ttk.Label(control_frame, text="Video:").grid(row=0, column=0, padx=5)
        self.video_var = tk.StringVar()
        video_names = []
        for vid in self.all_videos:
            if vid['is_annotated']:
                video_names.append(f"{vid['name']} (ID: {vid['video_data']['id']})")
            else:
                video_names.append(f"{vid['name']} (Not annotated)")
        
        self.video_combo = ttk.Combobox(
            control_frame,
            textvariable=self.video_var,
            values=video_names,
            state="readonly",
            width=40
        )
        self.video_combo.grid(row=0, column=1, columnspan=2, padx=5)
        self.video_combo.bind('<<ComboboxSelected>>', self.on_video_change)
        if self.all_videos:
            # Find first annotated video
            for i, vid in enumerate(self.all_videos):
                if vid['is_annotated']:
                    self.video_combo.current(i)
                    break
        
        # Frame navigation
        ttk.Label(control_frame, text="Frame:").grid(row=0, column=3, padx=5)
        self.frame_var = tk.IntVar(value=0)
        self.frame_spinbox = ttk.Spinbox(
            control_frame,
            from_=0,
            to=len(self.frame_paths) - 1 if self.frame_paths else 0,
            textvariable=self.frame_var,
            width=10,
            command=self.on_frame_change
        )
        self.frame_spinbox.grid(row=0, column=4, padx=5)
        self.frame_spinbox.bind('<Return>', lambda e: self.on_frame_change())
        
        self.frame_label = ttk.Label(control_frame, text=f"/ {len(self.frame_paths) - 1 if self.frame_paths else 0}")
        self.frame_label.grid(row=0, column=5, padx=5)
        
        ttk.Button(control_frame, text="◀ Prev", command=self.prev_frame).grid(row=0, column=6, padx=5)
        ttk.Button(control_frame, text="Next ▶", command=self.next_frame).grid(row=0, column=7, padx=5)
        
        # Instance selection and management
        ttk.Label(control_frame, text="Instance:").grid(row=1, column=0, padx=5)
        self.instance_var = tk.StringVar()
        self.instance_combo = ttk.Combobox(
            control_frame,
            textvariable=self.instance_var,
            values=[f"Instance {i}" for i in self.instance_ids],
            state="readonly",
            width=15
        )
        self.instance_combo.grid(row=1, column=1, padx=5)
        self.instance_combo.bind('<<ComboboxSelected>>', self.on_instance_change)
        if self.instance_ids:
            self.instance_combo.current(0)
        
        # Add new instance button
        ttk.Button(control_frame, text="New Instance", command=self.create_new_instance).grid(row=1, column=2, padx=5)
        
        # Class label
        ttk.Label(control_frame, text="Class:").grid(row=1, column=3, padx=5)
        self.class_var = tk.StringVar()
        class_names = [cat['name'] for cat in self.categories]
        self.class_combo = ttk.Combobox(
            control_frame,
            textvariable=self.class_var,
            values=class_names,
            state="readonly",
            width=15
        )
        self.class_combo.grid(row=1, column=4, padx=5)
        self.class_combo.bind('<<ComboboxSelected>>', self.on_class_change)

        # Initialize class dropdown
        if self.current_instance_id is not None:
            inst_key = str(self.current_instance_id)
            if inst_key in self.instance_metadata:
                cat_name = self.instance_metadata[inst_key]['category_name']
                self.class_var.set(cat_name)
                for i, cat in enumerate(self.categories):
                    if cat['name'] == cat_name:
                        self.class_combo.current(i)
                        break
        
        # Drawing controls
        drawing_frame = ttk.Frame(control_frame)
        drawing_frame.grid(row=2, column=0, columnspan=8, pady=10)
        
        # Tool selection
        ttk.Label(drawing_frame, text="Tool:").pack(side=tk.LEFT, padx=5)
        self.tool_var = tk.StringVar(value="brush")
        ttk.Radiobutton(
            drawing_frame,
            text="Brush",
            variable=self.tool_var,
            value="brush",
            command=self.on_tool_change
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            drawing_frame,
            text="Polygon",
            variable=self.tool_var,
            value="polygon",
            command=self.on_tool_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(drawing_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(drawing_frame, text="Brush:").pack(side=tk.LEFT, padx=5)
        self.brush_var = tk.IntVar(value=self.brush_size)
        brush_scale = ttk.Scale(
            drawing_frame,
            from_=5,
            to=100,
            variable=self.brush_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_brush_change
        )
        brush_scale.pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="add")
        ttk.Radiobutton(
            drawing_frame,
            text="Add",
            variable=self.mode_var,
            value="add",
            command=self.on_mode_change
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            drawing_frame,
            text="Erase",
            variable=self.mode_var,
            value="erase",
            command=self.on_mode_change
        ).pack(side=tk.LEFT, padx=5)
        
        # Zoom controls
        ttk.Separator(drawing_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(drawing_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        ttk.Button(drawing_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(drawing_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(drawing_frame, text="Reset", command=self.reset_view, width=6).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(drawing_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(drawing_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(drawing_frame, text="Clear Frame", command=self.clear_current_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(drawing_frame, text="Propagate Forward", command=self.propagate_mask).pack(side=tk.LEFT, padx=5)
        ttk.Button(drawing_frame, text="AI Assist", command=self.ai_assist_propagate).pack(side=tk.LEFT, padx=5)
        
        # Image adjustment controls
        adjust_frame = ttk.Frame(control_frame)
        adjust_frame.grid(row=3, column=0, columnspan=8, pady=10)

        # Show/Hide image toggle
        self.show_image_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            adjust_frame,
            text="Show Image",
            variable=self.show_image_var,
            command=self.on_show_image_change
        ).pack(side=tk.LEFT, padx=5)
        
        # Show/Hide mask toggle
        self.show_mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            adjust_frame,
            text="Show Mask",
            variable=self.show_mask_var,
            command=self.on_show_mask_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(adjust_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Brightness
        ttk.Label(adjust_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(
            adjust_frame,
            from_=0.3,
            to=10.0,
            variable=self.brightness_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_brightness_change
        )
        brightness_scale.pack(side=tk.LEFT, padx=5)
        
        # Contrast
        ttk.Label(adjust_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(
            adjust_frame,
            from_=0.3,
            to=10.0,
            variable=self.contrast_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_contrast_change
        )
        contrast_scale.pack(side=tk.LEFT, padx=5)
        
        # Saturation
        ttk.Label(adjust_frame, text="Saturation:").pack(side=tk.LEFT, padx=5)
        self.saturation_var = tk.DoubleVar(value=1.0)
        saturation_scale = ttk.Scale(
            adjust_frame,
            from_=0.0,
            to=10.0,
            variable=self.saturation_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_saturation_change
        )
        saturation_scale.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        ttk.Button(adjust_frame, text="Reset Adjustments", command=self.reset_adjustments).pack(side=tk.LEFT, padx=5)
        
        # Main canvas for image and mask display
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        
        # Mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_mousewheel_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self.on_mousewheel_linux(e, -1))
        
        # Keyboard shortcuts
        self.root.bind("<space>", lambda e: self.toggle_mode())
        self.root.bind("=", lambda e: self.zoom_in())
        self.root.bind("-", lambda e: self.zoom_out())
        self.root.bind("0", lambda e: self.reset_view())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Escape>", lambda e: self.cancel_polygon())
        self.root.bind("<Return>", lambda e: self.complete_polygon())
        self.root.bind("m", lambda e: self.toggle_mask_visibility())
        self.root.bind("M", lambda e: self.toggle_mask_visibility())
        self.root.bind("i", lambda e: self.toggle_image_visibility())
        self.root.bind("I", lambda e: self.toggle_image_visibility())
        
        # Bottom frame
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)
        
        ttk.Label(bottom_frame, text="[Space: Toggle Mode | M: Toggle Mask | Mouse Wheel: Zoom | Middle/Right-click+Drag: Pan | ←/→: Navigate]").pack(side=tk.LEFT)
        
        ttk.Button(
            bottom_frame,
            text="Save YTVIS Annotations",
            command=self.save_ytvis_annotations
        ).pack(side=tk.RIGHT, padx=5)

    def create_new_instance(self):
        """Create a new instance for the current video"""
        if not hasattr(self, 'current_video_id'):
            messagebox.showerror("Error", "No video selected")
            return
        
        # Ask for category
        category_dialog = CategoryDialog(self.root, self.categories)
        if not category_dialog.result:
            return
        
        category_id = category_dialog.result
        cat_name = next((cat['name'] for cat in self.categories if cat['id'] == category_id), 'unknown')
        
        # Create new annotation
        inst_id = self.next_annotation_id
        self.next_annotation_id += 1
        
        num_frames = len(self.frame_paths)
        
        new_annotation = {
            'id': inst_id,
            'video_id': self.current_video_id,
            'category_id': category_id,
            'segmentations': [None] * num_frames,
            'bboxes': [None] * num_frames,
            'areas': [None] * num_frames
        }
        
        self.annotations.append(new_annotation)
        self.video_annotations.append(new_annotation)
        
        # Add to instance tracking
        self.instance_ids.append(inst_id)
        self.instance_ids.sort()
        
        self.instance_metadata[str(inst_id)] = {
            'category_id': category_id,
            'category_name': cat_name
        }
        
        self.masks[inst_id] = {}
        self.boxes[str(inst_id)] = {}
        
        # Update UI
        self.instance_combo.config(values=[f"Instance {i}" for i in self.instance_ids])
        
        # Select the new instance
        new_idx = self.instance_ids.index(inst_id)
        self.instance_combo.current(new_idx)
        self.current_instance_id = inst_id
        
        # Update class combo
        self.class_var.set(cat_name)
        for i, cat in enumerate(self.categories):
            if cat['id'] == category_id:
                self.class_combo.current(i)
                break
        
        # Mark video as modified
        self.modified_videos.add(self.current_video_id)
        
        messagebox.showinfo("Success", f"Created new instance {inst_id} with category '{cat_name}'")
        self.update_display()

    def ai_assist_propagate(self):
        """Use VOS model to propagate mask from current frame"""
        if self.model_cfg is None:
            messagebox.showerror("Error", "No VOS model provided. Pass a model to the MaskEditorApp constructor.")
            return
        
        if self.model_chkpt is None:
            messagebox.showerror("Error", "No VOS model checkpoint provided. Pass a model to the MaskEditorApp constructor.")
            return
        
        if self.current_instance_id is None:
            messagebox.showwarning("Warning", "No instance selected")
            return
        
        if self.current_instance_id not in self.masks:
            messagebox.showwarning("Warning", "No mask data for current instance")
            return
        
        if self.current_frame_idx not in self.masks[self.current_instance_id]:
            messagebox.showwarning("Warning", "No mask for current frame")
            return
        
        current_mask = self.masks[self.current_instance_id][self.current_frame_idx]
        if current_mask is None or not np.any(current_mask):
            messagebox.showwarning("Warning", "Current frame has no mask to propagate")
            return
        
        # Ask user for propagation direction
        dialog = PropagationDialog(self.root, self.current_frame_idx, len(self.frame_paths))
        if not dialog.result:
            return
        
        start_frame, end_frame = dialog.result
        
        try:
            # Show progress
            progress_window = ProgressWindow(self.root, "Running AI Assist...")
            self.root.update()
            
            # Load all frames for the video
            frames = []
            for i in range(len(self.frame_paths)):
                frame_path = os.path.join(self.root_dir, self.frame_paths[i])
                img = Image.open(frame_path).convert("RGB")
                frames.append(np.array(img))
            
            frames = np.stack(frames, axis=0)  # Shape: (T, H, W, 3)
            
            # Run VOS model
            progress_window.update_text("Running VOS model...")
            self.root.update()
            
            # predicted_masks = self.model(frames, current_mask, self.current_frame_idx)
            results = predict_masks_with_sam2(
                frames,
                current_mask,
                self.current_frame_idx,
                self.model_chkpt,
                self.model_cfg
            )
            
            progress_window.close()
            
            # Update masks from start_frame to end_frame
            num_updated = 0
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx == self.current_frame_idx:
                    continue  # Skip the initial frame
                
                if frame_idx < len(results['masks']):
                    pred_mask = results['masks'][frame_idx]
                    
                    # Convert to boolean if needed
                    if pred_mask.dtype != bool:
                        pred_mask = pred_mask > 0.5
                    
                    # Update mask
                    self.masks[self.current_instance_id][frame_idx] = pred_mask.astype(bool)
                    self.update_bbox_from_mask(self.current_instance_id, frame_idx)
                    num_updated += 1
            
            # Mark video as modified
            self.modified_videos.add(self.current_video_id)
            
            self.invalidate_buffer()
            self.update_display()
            
            messagebox.showinfo("Success", f"AI Assist completed!\nUpdated {num_updated} frames from {start_frame} to {end_frame}")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.close()
            messagebox.showerror("Error", f"AI Assist failed:\n{str(e)}")

    def on_video_change(self, event=None):
        """Handle video selection change"""
        selection = self.video_combo.current()
        if selection >= 0 and selection != self.current_video_idx:
            self.set_current_video(selection)
            
            # Update UI
            self.frame_spinbox.config(to=len(self.frame_paths) - 1 if self.frame_paths else 0)
            self.frame_label.config(text=f"/ {len(self.frame_paths) - 1 if self.frame_paths else 0}")
            self.frame_var.set(0)
            
            # Update instance combo
            self.instance_combo.config(values=[f"Instance {i}" for i in self.instance_ids])
            if self.instance_ids:
                self.instance_combo.current(0)
                self.on_instance_change()
            else:
                self.instance_var.set("")
            
            self.update_display()

    def on_show_image_change(self):
        """Handle show image toggle"""
        self.show_image = self.show_image_var.get()
        self.invalidate_buffer()
        self.update_display()

    def toggle_image_visibility(self):
        self.show_image = not self.show_image
        self.show_image_var.set(self.show_image)
        self.invalidate_buffer()
        self.update_display()
    
    def on_show_mask_change(self):
        """Handle show mask toggle"""
        self.show_mask = self.show_mask_var.get()
        self.invalidate_buffer()
        self.update_display()
    
    def toggle_mask_visibility(self):
        """Toggle mask visibility with keyboard shortcut"""
        self.show_mask = not self.show_mask
        self.show_mask_var.set(self.show_mask)
        self.invalidate_buffer()
        self.update_display()
    
    def on_brightness_change(self, value):
        """Handle brightness change"""
        self.brightness = float(value)
        self.image_cache = {}
        self.invalidate_buffer()
        self.update_display()
    
    def on_contrast_change(self, value):
        """Handle contrast change"""
        self.contrast = float(value)
        self.image_cache = {}
        self.invalidate_buffer()
        self.update_display()
    
    def on_saturation_change(self, value):
        """Handle saturation change"""
        self.saturation = float(value)
        self.image_cache = {}
        self.invalidate_buffer()
        self.update_display()
    
    def reset_adjustments(self):
        """Reset all image adjustments to default"""
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.image_cache = {}
        self.invalidate_buffer()
        self.update_display()
    
    def on_tool_change(self):
        """Handle tool change"""
        self.tool = self.tool_var.get()
        if self.tool == 'polygon':
            self.polygon_points = []
            self.canvas.config(cursor="tcross")
        else:
            self.update_cursor()
        self.update_display()
    
    def toggle_mode(self):
        """Toggle between add and erase modes"""
        if self.mode == 'add':
            self.mode = 'erase'
            self.mode_var.set('erase')
        else:
            self.mode = 'add'
            self.mode_var.set('add')
        self.update_cursor()
    
    def update_cursor(self):
        """Update cursor based on mode and tool"""
        if self.tool == 'polygon':
            self.canvas.config(cursor="tcross")
        elif self.mode == 'add':
            self.canvas.config(cursor="tcross")
        else:
            self.canvas.config(cursor="circle")
    
    def zoom_in(self):
        """Zoom in"""
        self.zoom_level *= 1.2
        self.zoom_level = min(self.zoom_level, 10.0)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.invalidate_buffer()
        self.update_display()
    
    def zoom_out(self):
        """Zoom out"""
        self.zoom_level /= 1.2
        self.zoom_level = max(self.zoom_level, 0.1)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.invalidate_buffer()
        self.update_display()
    
    def reset_view(self):
        """Reset zoom and pan"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.zoom_label.config(text="100%")
        self.invalidate_buffer()
        self.update_display()
    
    def on_mousewheel(self, event):
        """Handle mouse wheel zoom"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def on_mousewheel_linux(self, event, direction):
        """Handle mouse wheel zoom on Linux"""
        if direction > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def on_right_click(self, event):
        """Handle right click"""
        if self.tool == 'polygon' and self.polygon_points:
            self.complete_polygon()
        else:
            self.on_pan_start(event)
    
    def on_pan_start(self, event):
        """Start panning"""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")
    
    def on_pan_drag(self, event):
        """Handle pan dragging"""
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.schedule_update()
    
    def on_pan_end(self, event):
        """End panning"""
        self.panning = False
        self.update_cursor()
        
    def on_frame_change(self):
        """Handle frame change"""
        self.current_frame_idx = self.frame_var.get()
        self.invalidate_buffer()
        self.update_display()
        
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_var.set(self.current_frame_idx)
            self.invalidate_buffer()
            self.update_display()
            
    def next_frame(self):
        """Go to next frame"""
        if self.current_frame_idx < len(self.frame_paths) - 1:
            self.current_frame_idx += 1
            self.frame_var.set(self.current_frame_idx)
            self.invalidate_buffer()
            self.update_display()
            
    def on_instance_change(self, event=None):
        """Handle instance selection change"""
        selection = self.instance_combo.current()
        if selection >= 0:
            self.current_instance_id = self.instance_ids[selection]
            inst_key = str(self.current_instance_id)
            if inst_key in self.instance_metadata:
                cat_name = self.instance_metadata[inst_key]['category_name']
                self.class_var.set(cat_name)
                for i, cat in enumerate(self.categories):
                    if cat['name'] == cat_name:
                        self.class_combo.current(i)
                        break
            self.invalidate_buffer()
            self.update_display()
            
    def on_class_change(self, event=None):
        """Handle class label change"""
        if self.current_instance_id is not None:
            cat_name = self.class_var.get()
            cat = next(cat for cat in self.categories if cat['name'] == cat_name)
            inst_key = str(self.current_instance_id)
            self.instance_metadata[inst_key] = {
                'category_id': cat['id'],
                'category_name': cat['name']
            }
            # Mark video as modified
            self.modified_videos.add(self.current_video_id)
            
    def on_brush_change(self, value):
        """Handle brush size change"""
        self.brush_size = int(float(value))
        
    def on_mode_change(self):
        """Handle drawing mode change"""
        self.mode = self.mode_var.get()
        self.update_cursor()
        
    def clear_current_frame(self):
        """Clear mask for current frame and instance"""
        if self.current_instance_id is not None:
            if self.current_instance_id in self.masks:
                self.masks[self.current_instance_id][self.current_frame_idx] = None
            self.update_bbox_from_mask(self.current_instance_id, self.current_frame_idx)
            self.modified_videos.add(self.current_video_id)
            self.invalidate_buffer()
            self.update_display()
            
    def propagate_mask(self):
        """Propagate current mask forward to all subsequent frames"""
        if self.current_instance_id is None:
            return
        if self.current_instance_id not in self.masks:
            return
        if self.current_frame_idx not in self.masks[self.current_instance_id]:
            return
            
        mask = self.masks[self.current_instance_id][self.current_frame_idx]
        if mask is None:
            return
        
        num_forward_frames = len(self.frame_paths) - self.current_frame_idx - 1
        if num_forward_frames <= 0:
            messagebox.showinfo("Propagate Mask", "Already at the last frame!")
            return
            
        response = messagebox.askyesno(
            "Propagate Mask Forward",
            f"Copy mask from frame {self.current_frame_idx} to the next {num_forward_frames} frame(s)?"
        )
        if response:
            for frame_idx in range(self.current_frame_idx + 1, len(self.frame_paths)):
                self.masks[self.current_instance_id][frame_idx] = mask.copy()
                self.update_bbox_from_mask(self.current_instance_id, frame_idx)
            self.modified_videos.add(self.current_video_id)
            self.invalidate_buffer()
            self.update_display()
            messagebox.showinfo("Success", f"Mask propagated to {num_forward_frames} forward frame(s)!")
    
    def invalidate_buffer(self):
        """Invalidate the overlay buffer to force regeneration"""
        self.overlay_buffer = None
        self.display_buffer = None
        self.last_display_params = None
            
    def on_mouse_press(self, event):
        """Handle mouse press for drawing"""
        if self.tool == 'polygon':
            self.add_polygon_point(event.x, event.y)
        elif not self.panning:
            self.drawing = True
            self.last_draw_pos = (event.x, event.y)
            self.draw_at_position(event.x, event.y)
        
    def on_mouse_drag(self, event):
        """Handle mouse drag for drawing"""
        if self.tool == 'brush' and self.drawing and not self.panning:
            if self.last_draw_pos is not None:
                self.draw_line(self.last_draw_pos[0], self.last_draw_pos[1], event.x, event.y)
            else:
                self.draw_at_position(event.x, event.y)
            self.last_draw_pos = (event.x, event.y)
            
    def on_mouse_release(self, event):
        """Handle mouse release"""
        if self.tool == 'brush':
            self.drawing = False
            self.last_draw_pos = None
        
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        if not hasattr(self, 'display_offset_x'):
            return None, None
        
        canvas_x -= self.pan_x
        canvas_y -= self.pan_y
        canvas_x -= self.display_offset_x
        canvas_y -= self.display_offset_y
        
        img_x = int(canvas_x / (self.scale_factor * self.zoom_level))
        img_y = int(canvas_y / (self.scale_factor * self.zoom_level))
        
        return img_x, img_y
    
    def mask_to_bbox(self, mask):
        """Convert mask to bounding box in YTVIS format [x, y, width, height]"""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    
    def update_bbox_from_mask(self, obj_id, frame_idx):
        """Update bounding box for a mask"""
        if obj_id not in self.masks or frame_idx not in self.masks[obj_id]:
            return
        
        mask = self.masks[obj_id][frame_idx]
        if mask is None:
            if str(obj_id) in self.boxes and str(frame_idx) in self.boxes[str(obj_id)]:
                self.boxes[str(obj_id)][str(frame_idx)] = None
            return
        
        bbox = self.mask_to_bbox(mask)
        
        if str(obj_id) not in self.boxes:
            self.boxes[str(obj_id)] = {}
        
        # Convert to [x_min, y_min, x_max, y_max] for internal storage
        if bbox:
            self.boxes[str(obj_id)][str(frame_idx)] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        else:
            self.boxes[str(obj_id)][str(frame_idx)] = None
        
    def add_polygon_point(self, canvas_x, canvas_y):
        """Add a point to the current polygon - allows points outside image bounds"""
        img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        if img_x is None or img_y is None:
            return
        
        # Don't restrict polygon points to image bounds - allow clicking outside
        # Points will be clipped when drawing the actual mask
        self.polygon_points.append((img_x, img_y))
        self.update_display()
    
    def cancel_polygon(self):
        """Cancel the current polygon"""
        if self.polygon_points:
            self.polygon_points = []
            self.update_display()
    
    def complete_polygon(self):
        """Complete and fill the current polygon"""
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Polygon Tool", "Need at least 3 points to create a polygon!")
            return
        
        if self.current_instance_id is None:
            return
        
        if self.current_instance_id not in self.masks:
            self.masks[self.current_instance_id] = {}
        if self.current_frame_idx not in self.masks[self.current_instance_id]:
            self.masks[self.current_instance_id][self.current_frame_idx] = np.zeros(
                self.image_shape[:2], dtype=bool
            )
        
        mask = self.masks[self.current_instance_id][self.current_frame_idx]
        if mask is None:
            mask = np.zeros(self.image_shape[:2], dtype=bool)
            self.masks[self.current_instance_id][self.current_frame_idx] = mask
        
        # Create polygon mask using PIL
        # Expand canvas to handle points outside image bounds, then crop
        # Find the bounding box of all polygon points
        xs = [p[0] for p in self.polygon_points]
        ys = [p[1] for p in self.polygon_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Determine if we need an expanded canvas
        needs_expansion = (min_x < 0 or min_y < 0 or 
                          max_x >= self.image_shape[1] or max_y >= self.image_shape[0])
        
        if needs_expansion:
            # Create expanded canvas
            offset_x = max(0, -min_x)
            offset_y = max(0, -min_y)
            expanded_width = max(self.image_shape[1], max_x + 1) + offset_x
            expanded_height = max(self.image_shape[0], max_y + 1) + offset_y
            
            poly_img = Image.new('L', (expanded_width, expanded_height), 0)
            draw = ImageDraw.Draw(poly_img)
            
            # Translate polygon points by offset
            translated_points = [(x + offset_x, y + offset_y) for x, y in self.polygon_points]
            draw.polygon(translated_points, outline=1, fill=1)
            
            # Extract the region corresponding to the actual image
            poly_mask = np.array(poly_img, dtype=bool)
            poly_mask = poly_mask[offset_y:offset_y + self.image_shape[0], 
                                  offset_x:offset_x + self.image_shape[1]]
        else:
            # Standard case - all points within bounds
            poly_img = Image.new('L', (self.image_shape[1], self.image_shape[0]), 0)
            draw = ImageDraw.Draw(poly_img)
            draw.polygon(self.polygon_points, outline=1, fill=1)
            poly_mask = np.array(poly_img, dtype=bool)
        
        # Apply polygon mask
        if self.mode == 'add':
            mask[poly_mask] = True
        else:
            mask[poly_mask] = False
        
        self.update_bbox_from_mask(self.current_instance_id, self.current_frame_idx)
        self.polygon_points = []
        
        self.modified_videos.add(self.current_video_id)
        self.invalidate_buffer()
        self.update_display()
        
    def draw_line(self, x1, y1, x2, y2):
        """Draw a line between two points"""
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        num_steps = max(int(distance / 2), 1)
        
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            x = x1 + t * dx
            y = y1 + t * dy
            self.draw_at_position(x, y, update_display=False)
        
        self.update_bbox_from_mask(self.current_instance_id, self.current_frame_idx)
        self.modified_videos.add(self.current_video_id)
        self.invalidate_buffer()
        self.schedule_update()
        
    def draw_at_position(self, canvas_x, canvas_y, update_display=True):
        """Draw mask at canvas position"""
        if self.current_instance_id is None:
            return
        
        img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        if img_x is None or img_y is None:
            return
            
        if img_x < 0 or img_x >= self.image_shape[1] or img_y < 0 or img_y >= self.image_shape[0]:
            return
            
        if self.current_instance_id not in self.masks:
            self.masks[self.current_instance_id] = {}
        if self.current_frame_idx not in self.masks[self.current_instance_id]:
            self.masks[self.current_instance_id][self.current_frame_idx] = np.zeros(
                self.image_shape[:2], dtype=bool
            )
        
        mask = self.masks[self.current_instance_id][self.current_frame_idx]
        if mask is None:
            mask = np.zeros(self.image_shape[:2], dtype=bool)
            self.masks[self.current_instance_id][self.current_frame_idx] = mask
        
        y, x = np.ogrid[:self.image_shape[0], :self.image_shape[1]]
        dist_sq = (x - img_x)**2 + (y - img_y)**2
        circle_mask = dist_sq <= (self.brush_size ** 2)
        
        if self.mode == 'add':
            mask[circle_mask] = True
        else:
            mask[circle_mask] = False
        
        if update_display:
            self.update_bbox_from_mask(self.current_instance_id, self.current_frame_idx)
            self.modified_videos.add(self.current_video_id)
            self.invalidate_buffer()
            self.schedule_update()
    
    def get_base_image(self):
        """Get base image with caching and adjustments"""
        frame_path = os.path.join(self.root_dir, self.frame_paths[self.current_frame_idx])
        
        cache_key = (self.current_frame_idx, self.brightness, self.contrast, self.saturation, self.current_video_idx)
        
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        img = Image.open(frame_path).convert("RGB")
        
        if self.brightness != 1.0 or self.contrast != 1.0 or self.saturation != 1.0:
            if self.brightness != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(self.brightness)
            
            if self.contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.contrast)
            
            if self.saturation != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(self.saturation)
        
        img_array = np.array(img)
        
        if len(self.image_cache) > 10:
            keys = sorted([k for k in self.image_cache.keys()], key=lambda x: x[0])
            for k in keys[:-5]:
                del self.image_cache[k]
        
        self.image_cache[cache_key] = img_array
        return img_array
    
    def generate_overlay(self):
        """Generate overlay image with mask"""
        if self.overlay_buffer is not None:
            return self.overlay_buffer

        if self.show_image:
            img_array = self.get_base_image()
            overlay = img_array.copy()
        else:
            overlay = np.zeros(
                (self.image_shape[0], self.image_shape[1], 3),
                dtype=np.uint8
            )
        
        if self.show_mask and self.current_instance_id is not None:
            if (self.current_instance_id in self.masks and 
                self.current_frame_idx in self.masks[self.current_instance_id]):
                mask = self.masks[self.current_instance_id][self.current_frame_idx]
                if mask is not None and np.any(mask):
                    color = np.array([0, 255, 0], dtype=np.uint8)
                    mask_indices = np.where(mask)
                    overlay[mask_indices] = (overlay[mask_indices] * 0.5 + color * 0.5).astype(np.uint8)
        
        self.overlay_buffer = overlay
        return overlay
    
    def schedule_update(self):
        """Schedule a display update with debouncing"""
        if not self.update_pending:
            self.update_pending = True
            self.root.after(8, self._do_update)
    
    def _do_update(self):
        """Actually perform the display update"""
        self.update_pending = False
        self.update_display()
        
    def update_display(self):
        """Update the canvas display"""
        if not self.frame_paths:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.update_display)
            return
        
        display_params = (
            canvas_width, canvas_height, self.zoom_level, self.pan_x, self.pan_y,
            self.current_frame_idx, self.current_instance_id, len(self.polygon_points),
            self.brightness, self.contrast, self.saturation,
            self.show_mask, self.show_image, self.current_video_idx
        )
        
        if self.last_display_params == display_params and self.display_buffer is not None and not self.polygon_points:
            self.canvas.delete("all")
            self.canvas.create_image(
                self.cached_display_x,
                self.cached_display_y,
                image=self.display_img,
                anchor=tk.NW
            )
            return
        
        overlay = self.generate_overlay()
        img_pil = Image.fromarray(overlay)
        img_width, img_height = img_pil.size
        
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        self.scale_factor = min(scale_w, scale_h)
        
        final_scale = self.scale_factor * self.zoom_level
        
        new_width = int(img_width * final_scale)
        new_height = int(img_height * final_scale)
        
        if final_scale < 0.5:
            resample = Image.Resampling.BILINEAR
        else:
            resample = Image.Resampling.LANCZOS
            
        img_pil = img_pil.resize((new_width, new_height), resample)
        
        self.display_offset_x = (canvas_width - new_width) // 2
        self.display_offset_y = (canvas_height - new_height) // 2
        
        display_x = self.display_offset_x + self.pan_x
        display_y = self.display_offset_y + self.pan_y
        
        self.display_img = ImageTk.PhotoImage(img_pil)
        
        self.last_display_params = display_params
        self.display_buffer = self.display_img
        self.cached_display_x = display_x
        self.cached_display_y = display_y
        
        self.canvas.delete("all")
        self.canvas.create_image(
            display_x,
            display_y,
            image=self.display_img,
            anchor=tk.NW
        )
        
        if self.tool == 'polygon' and self.polygon_points:
            self.draw_polygon_preview()
    
    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates"""
        canvas_x = img_x * self.scale_factor * self.zoom_level + self.display_offset_x + self.pan_x
        canvas_y = img_y * self.scale_factor * self.zoom_level + self.display_offset_y + self.pan_y
        return canvas_x, canvas_y
    
    def draw_polygon_preview(self):
        """Draw the current polygon points and edges"""
        if len(self.polygon_points) < 1:
            return
        
        canvas_points = []
        for img_x, img_y in self.polygon_points:
            canvas_x, canvas_y = self.image_to_canvas_coords(img_x, img_y)
            canvas_points.append((canvas_x, canvas_y))
        
        for i in range(len(canvas_points)):
            x1, y1 = canvas_points[i]
            
            r = 4
            self.canvas.create_oval(x1-r, y1-r, x1+r, y1+r, fill='red', outline='white', width=2)
            
            if i < len(canvas_points) - 1:
                x2, y2 = canvas_points[i + 1]
                self.canvas.create_line(x1, y1, x2, y2, fill='yellow', width=2)
        
        if len(canvas_points) >= 3:
            x1, y1 = canvas_points[-1]
            x2, y2 = canvas_points[0]
            self.canvas.create_line(x1, y1, x2, y2, fill='yellow', width=2, dash=(5, 5))
            
    def save_ytvis_annotations(self):
        """Save annotations back to YTVIS-19 format"""
        # Only update annotations for modified videos
        for video_id in self.modified_videos:
            # Find video index
            video_idx = self.video_id_to_idx.get(video_id)
            if video_idx is None:
                continue
            
            # Update annotations for this video
            for ann in self.annotations:
                if ann['video_id'] != video_id:
                    continue
                    
                inst_id = ann['id']
                
                # Update category if changed
                if str(inst_id) in self.instance_metadata:
                    ann['category_id'] = self.instance_metadata[str(inst_id)]['category_id']
                
                # Rebuild segmentations and bboxes
                segmentations = []
                bboxes = []
                
                num_frames = self.videos[video_idx]['length']
                
                for frame_idx in range(num_frames):
                    if inst_id in self.masks and frame_idx in self.masks[inst_id]:
                        mask = self.masks[inst_id][frame_idx]
                        if mask is not None and np.any(mask):
                            # Encode mask to RLE
                            mask_uint8 = mask.astype(np.uint8)
                            rle = maskUtils.encode(np.asfortranarray(mask_uint8))
                            rle['counts'] = rle['counts'].decode('utf-8')
                            segmentations.append(rle)
                            
                            # Get bbox
                            bbox = self.mask_to_bbox(mask)
                            bboxes.append(bbox)
                        else:
                            segmentations.append(None)
                            bboxes.append(None)
                    else:
                        segmentations.append(None)
                        bboxes.append(None)
                
                ann['segmentations'] = segmentations
                ann['bboxes'] = bboxes
                
                # Update areas
                areas = []
                for seg in segmentations:
                    if seg is not None:
                        if isinstance(seg, dict):
                            area = int(maskUtils.area(seg))
                        else:
                            area = 0
                        areas.append(area)
                    else:
                        areas.append(None)
                ann['areas'] = areas
        
        # Write back to file
        output_data = {
            'info': self.ytvis_data.get('info', {}),
            'categories': self.categories,
            'videos': self.videos,
            'annotations': self.annotations
        }
        
        with open(self.annotations_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Clear modified set
        self.modified_videos.clear()
        
        messagebox.showinfo("Saved", f"YTVIS annotations saved successfully to:\n{self.annotations_path}")


class FrameRangeDialog:
    """Dialog for selecting frame range"""
    def __init__(self, parent, num_frames):
        self.result = None
        
        dialog = tk.Toplevel(parent)
        dialog.title("Select Frame Range")
        dialog.geometry("400x150")
        dialog.transient(parent)
        dialog.grab_set()
        
        # Center the dialog on parent window
        dialog.update_idletasks()
        
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Total frames available: {num_frames}").pack(pady=5)
        
        # Start frame
        start_frame = ttk.Frame(frame)
        start_frame.pack(fill=tk.X, pady=5)
        ttk.Label(start_frame, text="Start frame:", width=15).pack(side=tk.LEFT)
        self.start_var = tk.IntVar(value=0)
        ttk.Spinbox(start_frame, from_=0, to=num_frames-1, textvariable=self.start_var, width=10).pack(side=tk.LEFT)
        
        # End frame
        end_frame = ttk.Frame(frame)
        end_frame.pack(fill=tk.X, pady=5)
        ttk.Label(end_frame, text="End frame:", width=15).pack(side=tk.LEFT)
        self.end_var = tk.IntVar(value=num_frames-1)
        ttk.Spinbox(end_frame, from_=0, to=num_frames-1, textvariable=self.end_var, width=10).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=lambda: self.on_ok(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
    
    def on_ok(self, dialog):
        self.result = (self.start_var.get(), self.end_var.get())
        dialog.destroy()


class CategoryDialog:
    """Dialog for selecting category"""
    def __init__(self, parent, categories):
        self.result = None
        
        dialog = tk.Toplevel(parent)
        dialog.title("Select Category")
        dialog.geometry("300x150")
        dialog.transient(parent)
        dialog.grab_set()
        
        # Center the dialog on parent window
        dialog.update_idletasks()
        
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Select category for new instance:").pack(pady=10)
        
        self.cat_var = tk.StringVar()
        cat_combo = ttk.Combobox(
            frame,
            textvariable=self.cat_var,
            values=[cat['name'] for cat in categories],
            state="readonly",
            width=25
        )
        cat_combo.pack(pady=5)
        if categories:
            cat_combo.current(0)
        
        self.categories = categories
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=lambda: self.on_ok(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
    
    def on_ok(self, dialog):
        cat_name = self.cat_var.get()
        cat = next((c for c in self.categories if c['name'] == cat_name), None)
        if cat:
            self.result = cat['id']
        dialog.destroy()


class PropagationDialog:
    """Dialog for selecting propagation range"""
    def __init__(self, parent, current_frame, num_frames):
        self.result = None
        
        dialog = tk.Toplevel(parent)
        dialog.title("AI Assist - Select Range")
        dialog.geometry("400x250")
        dialog.transient(parent)
        dialog.grab_set()
        
        # Center the dialog on parent window
        dialog.update_idletasks()
        
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Current frame: {current_frame}").pack(pady=5)
        ttk.Label(frame, text=f"Total frames: {num_frames}").pack(pady=5)
        ttk.Label(frame, text="Select range for mask propagation:").pack(pady=10)
        
        # Start frame
        start_frame_widget = ttk.Frame(frame)
        start_frame_widget.pack(fill=tk.X, pady=5)
        ttk.Label(start_frame_widget, text="Start frame:", width=15).pack(side=tk.LEFT)
        self.start_var = tk.IntVar(value=current_frame)
        ttk.Spinbox(start_frame_widget, from_=0, to=num_frames-1, textvariable=self.start_var, width=10).pack(side=tk.LEFT)
        
        # End frame
        end_frame_widget = ttk.Frame(frame)
        end_frame_widget.pack(fill=tk.X, pady=5)
        ttk.Label(end_frame_widget, text="End frame:", width=15).pack(side=tk.LEFT)
        self.end_var = tk.IntVar(value=num_frames-1)
        ttk.Spinbox(end_frame_widget, from_=0, to=num_frames-1, textvariable=self.end_var, width=10).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Run AI Assist", command=lambda: self.on_ok(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
    
    def on_ok(self, dialog):
        self.result = (self.start_var.get(), self.end_var.get())
        dialog.destroy()


class ProgressWindow:
    """Simple progress window"""
    def __init__(self, parent, text="Processing..."):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Please Wait")
        self.dialog.geometry("300x100")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog on the PARENT window, not the screen
        self.dialog.update_idletasks()
        
        # Get parent window position and size
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position relative to parent
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.label = ttk.Label(frame, text=text)
        self.label.pack(pady=20)
        
        # Disable close button
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    
    def update_text(self, text):
        self.label.config(text=text)
        self.dialog.update()
    
    def close(self):
        self.dialog.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Mask Editor GUI for YTVIS-19 format annotations with AI assist"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root directory for resolving video frame paths"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to YTVIS-19 format JSON annotations file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/sam2_hiera_large.pt",
        help="Path to SAM2 checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./sam2_configs/sam2_hiera_l.yaml",
        help="SAM2 model config file name (will look in sam2_configs/ directory or parent directory)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        return
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"SAM2 model config file not found: {args.config}\n"
            f"Please ensure the config file is in the sam2_configs/ directory or the current directory."
        )
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {args.checkpoint}\n"
            f"Please download checkpoints using: cd checkpoints && ./download_ckpts.sh"
        )
    
    root = tk.Tk()
    app = MaskEditorApp(root, args.root_dir, args.annotations, model_cfg=args.config, model_chkpt=args.checkpoint)
    root.mainloop()


if __name__ == "__main__":
    main()