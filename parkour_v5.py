import cv2
import numpy as np
import torch
import time
from pathlib import Path
from collections import deque
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import subprocess
import sys
import os
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Install required packages if missing (simpler alternatives)
required_packages = [
    'ultralytics', 'torch', 'opencv-python', 'numpy', 
    'pandas', 'matplotlib', 'reportlab', 'scipy'
]

def install_packages():
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Now import all required libraries
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.signal import savgol_filter
from collections import deque
import json
from datetime import datetime
from pathlib import Path
import os

class LightweightPoseEstimator:
    """Lightweight pose estimator using YOLO pose or simple body proportions"""
    
    def __init__(self, use_yolo_pose=True):
        """
        Args:
            use_yolo_pose: Use YOLO pose model if available, otherwise use body proportion estimation
        """
        self.use_yolo_pose = use_yolo_pose
        
        if use_yolo_pose:
            try:
                # Try to load YOLO pose model
                self.pose_model = YOLO('yolo11n-pose.pt')  # YOLOv11 pose model
                print("Loaded YOLOv11 Pose model")
                self.has_pose_model = True
            except:
                print("YOLO pose model not available, using body proportion estimation")
                self.has_pose_model = False
                self.pose_model = None
        else:
            self.has_pose_model = False
            self.pose_model = None
        
        # Body proportion constants (based on average human proportions)
        self.body_proportions = {
            'head_to_neck': 0.1,        # Head is 10% of body height
            'neck_to_hip': 0.3,         # Torso is 30% of body height
            'hip_to_knee': 0.25,        # Thigh is 25% of body height
            'knee_to_ankle': 0.25,      # Lower leg is 25% of body height
            'shoulder_width': 0.2,      # Shoulders are 20% of body height
            'hip_width': 0.15           # Hips are 15% of body height
        }
        
        # Keypoint indices (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections (for visualization)
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
    
    def estimate_from_bbox(self, bbox, frame_height):
        """Estimate pose keypoints from bounding box using body proportions"""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Estimate body height (assuming person is roughly standing)
        estimated_height = bbox_height * 1.2  # Add 20% for head above bbox
        
        # Generate synthetic keypoints based on body proportions
        keypoints = np.zeros((17, 3))  # 17 keypoints, 3 values (x, y, confidence)
        
        # Head (0)
        keypoints[0] = [center_x, y1 - estimated_height * 0.05, 0.8]
        
        # Eyes (1, 2) and ears (3, 4)
        eye_y = y1 - estimated_height * 0.02
        ear_y = y1 - estimated_height * 0.04
        keypoints[1] = [center_x - estimated_height * 0.02, eye_y, 0.7]
        keypoints[2] = [center_x + estimated_height * 0.02, eye_y, 0.7]
        keypoints[3] = [center_x - estimated_height * 0.03, ear_y, 0.6]
        keypoints[4] = [center_x + estimated_height * 0.03, ear_y, 0.6]
        
        # Shoulders (5, 6)
        shoulder_y = y1 + estimated_height * 0.1
        shoulder_width = estimated_height * self.body_proportions['shoulder_width']
        keypoints[5] = [center_x - shoulder_width/2, shoulder_y, 0.9]
        keypoints[6] = [center_x + shoulder_width/2, shoulder_y, 0.9]
        
        # Elbows (7, 8)
        elbow_y = shoulder_y + estimated_height * 0.15
        keypoints[7] = [center_x - shoulder_width/2 * 0.8, elbow_y, 0.7]
        keypoints[8] = [center_x + shoulder_width/2 * 0.8, elbow_y, 0.7]
        
        # Wrists (9, 10)
        wrist_y = elbow_y + estimated_height * 0.15
        keypoints[9] = [center_x - shoulder_width/2 * 0.6, wrist_y, 0.6]
        keypoints[10] = [center_x + shoulder_width/2 * 0.6, wrist_y, 0.6]
        
        # Hips (11, 12)
        hip_y = shoulder_y + estimated_height * 0.3
        hip_width = estimated_height * self.body_proportions['hip_width']
        keypoints[11] = [center_x - hip_width/2, hip_y, 0.9]
        keypoints[12] = [center_x + hip_width/2, hip_y, 0.9]
        
        # Knees (13, 14)
        knee_y = hip_y + estimated_height * 0.25
        keypoints[13] = [center_x - hip_width/2 * 0.8, knee_y, 0.8]
        keypoints[14] = [center_x + hip_width/2 * 0.8, knee_y, 0.8]
        
        # Ankles (15, 16)
        ankle_y = knee_y + estimated_height * 0.25
        keypoints[15] = [center_x - hip_width/2 * 0.6, ankle_y, 0.7]
        keypoints[16] = [center_x + hip_width/2 * 0.6, ankle_y, 0.7]
        
        return keypoints
    
    def estimate_pose(self, frame, bbox=None):
        """Estimate pose using available method"""
        if self.has_pose_model and self.pose_model is not None:
            try:
                # Use YOLO pose model
                results = self.pose_model(frame, verbose=False)[0]
                
                if results.keypoints is not None and len(results.keypoints) > 0:
                    # Get the first (most confident) person
                    keypoints = results.keypoints[0].data.cpu().numpy()[0]
                    return keypoints
            except:
                pass
        
        # Fallback to body proportion estimation
        if bbox is not None:
            return self.estimate_from_bbox(bbox, frame.shape[0])
        
        return None

class ParkourAthlete:
    """Represents a parkour athlete with personal data and progress tracking"""
    
    def __init__(self, athlete_id, name, age, weight, height, skill_level):
        self.id = athlete_id
        self.name = name
        self.age = age
        self.weight = weight  # kg
        self.height = height  # cm
        self.skill_level = skill_level  # beginner, intermediate, advanced, pro
        self.attempts = []
        self.progress_data = {
            'agility_scores': [],
            'safety_scores': [],
            'accuracy_scores': [],
            'completion_times': [],
            'dates': []
        }
    
    def add_attempt(self, attempt_data):
        """Add a new attempt to athlete's history"""
        self.attempts.append(attempt_data)
        self.progress_data['agility_scores'].append(attempt_data['agility_score'])
        self.progress_data['safety_scores'].append(attempt_data['safety_score'])
        self.progress_data['accuracy_scores'].append(attempt_data['accuracy_score'])
        self.progress_data['completion_times'].append(attempt_data['completion_time'])
        self.progress_data['dates'].append(attempt_data['timestamp'])
    
    def get_progress_chart_data(self):
        """Prepare data for progress charts"""
        return {
            'dates': self.progress_data['dates'],
            'agility': self.progress_data['agility_scores'],
            'safety': self.progress_data['safety_scores'],
            'accuracy': self.progress_data['accuracy_scores'],
            'times': self.progress_data['completion_times']
        }

class ParkourObstacle:
    """Defines parkour obstacles and target zones"""
    
    def __init__(self, name, obstacle_type, difficulty, target_zones):
        """
        Args:
            name: Name of obstacle (e.g., "Kong Vault Box", "Precision Jump")
            obstacle_type: Type of movement required
            difficulty: 1-10 scale
            target_zones: List of target zones [(x1,y1,x2,y2), ...]
        """
        self.name = name
        self.type = obstacle_type
        self.difficulty = difficulty
        self.target_zones = target_zones
        self.optimal_joint_angles = self._get_optimal_angles()
    
    def _get_optimal_angles(self):
        """Returns optimal joint angles for this obstacle type"""
        if "vault" in self.type.lower():
            return {
                'knee_angle_min': 100,  # degrees
                'knee_angle_max': 140,
                'hip_angle_min': 80,
                'hip_angle_max': 120,
                'landing_knee_angle': 140,  # For shock absorption
                'arm_extension_min': 150
            }
        elif "precision" in self.type.lower():
            return {
                'knee_angle_min': 120,
                'knee_angle_max': 160,
                'landing_depth_max': 30,  # cm
                'balance_tolerance': 15  # degrees from vertical
            }
        elif "wall_run" in self.type.lower():
            return {
                'foot_placement_width': 40,  # cm
                'body_lean_angle': 70,
                'push_angle': 45
            }
        return {}

class ParkourAnalyzer:
    """Main class for parkour analysis with multi-camera support"""
    
    def __init__(self, camera_configs=None):
        """
        Args:
            camera_configs: List of camera configurations for multi-camera setup
        """
        self.camera_configs = camera_configs or [
            {'id': 0, 'name': 'Front', 'fov': 60, 'resolution': (3840, 2160)},
            {'id': 1, 'name': 'Side', 'fov': 60, 'resolution': (3840, 2160)},
            {'id': 2, 'name': 'Top', 'fov': 90, 'resolution': (3840, 2160)}
        ]
        
        # Initialize models
        self.yolo_model = YOLO('yolo11n.pt')  # YOLOv11 for detection
        self.pose_estimator = LightweightPoseEstimator(use_yolo_pose=False)  # Use simple method
        
        # Tracking
        self.track_history = {}
        self.next_track_id = 0
        self.max_history = 100
        
        # Analysis parameters
        self.analysis_results = {
            'agility_metrics': {},
            'safety_metrics': {},
            'accuracy_metrics': {},
            'technical_errors': [],
            'slow_motion_triggers': []
        }
        
        # Reference data (pro athlete movements)
        self.pro_reference = self._load_pro_reference()
        
        # Visualization
        self.viz_scale = 0.5  # For 4K downscaling
        self.colors = {
            'athlete': (0, 255, 0),
            'target_zone': (255, 255, 0),
            'landing_zone': (255, 0, 0),
            'error': (0, 0, 255),
            'pro_reference': (0, 165, 255, 100)  # Orange with transparency
        }
        
        # Performance metrics
        self.metrics_history = deque(maxlen=1000)
        
        print("Parkour Analysis Framework Initialized")
        print(f"Cameras configured: {len(self.camera_configs)}")
    
    def _load_pro_reference(self):
        """Load professional athlete reference data"""
        # Simplified reference data for Kong Vault
        return {
            'keypoints_sequence': self._generate_pro_kong_vault_sequence(),
            'timing': {
                'approach_time': 1.2,  # seconds
                'vault_time': 0.8,
                'landing_time': 0.5
            },
            'optimal_angles': {
                'takeoff_knee': 125,
                'vault_hip': 95,
                'landing_knee': 140
            },
            'speed_profile': [0.8, 1.2, 0.9, 0.6, 0.3]  # Normalized speed
        }
    
    def _generate_pro_kong_vault_sequence(self):
        """Generate synthetic pro athlete keypoints for Kong Vault"""
        sequence = []
        frames = 60  # 2 seconds at 30fps
        
        # Base keypoints for standing position
        base_keypoints = np.array([
            [0.5, 0.1, 0.9],    # nose
            [0.48, 0.12, 0.8],  # left_eye
            [0.52, 0.12, 0.8],  # right_eye
            [0.46, 0.13, 0.7],  # left_ear
            [0.54, 0.13, 0.7],  # right_ear
            [0.4, 0.25, 0.9],   # left_shoulder
            [0.6, 0.25, 0.9],   # right_shoulder
            [0.35, 0.4, 0.8],   # left_elbow
            [0.65, 0.4, 0.8],   # right_elbow
            [0.3, 0.55, 0.7],   # left_wrist
            [0.7, 0.55, 0.7],   # right_wrist
            [0.45, 0.55, 0.9],  # left_hip
            [0.55, 0.55, 0.9],  # right_hip
            [0.45, 0.75, 0.8],  # left_knee
            [0.55, 0.75, 0.8],  # right_knee
            [0.45, 0.95, 0.7],  # left_ankle
            [0.55, 0.95, 0.7],  # right_ankle
        ])
        
        for i in range(frames):
            # Animate for kong vault motion
            progress = i / frames
            
            # Phase 1: Approach (0-0.3)
            if progress < 0.3:
                # Slight forward lean
                lean_factor = progress * 2
                frame_keypoints = base_keypoints.copy()
                frame_keypoints[:, 0] += 0.1 * lean_factor  # Shift forward
                frame_keypoints[5:, 1] -= 0.05 * lean_factor  # Lower center of mass
            
            # Phase 2: Takeoff (0.3-0.5)
            elif progress < 0.5:
                # Crouch and prepare for vault
                crouch_factor = (progress - 0.3) / 0.2
                frame_keypoints = base_keypoints.copy()
                frame_keypoints[13:, 1] += 0.2 * crouch_factor  # Bend knees
                frame_keypoints[7:11, 1] -= 0.1 * crouch_factor  # Raise arms
            
            # Phase 3: Vault (0.5-0.8)
            elif progress < 0.8:
                # Vaulting motion
                vault_factor = (progress - 0.5) / 0.3
                frame_keypoints = base_keypoints.copy()
                frame_keypoints[:, 1] -= 0.3 * vault_factor  # Body goes up
                frame_keypoints[7:11, 0] -= 0.2 * vault_factor  # Arms forward
                frame_keypoints[13:, 0] += 0.3 * vault_factor  # Legs tuck
            
            # Phase 4: Landing (0.8-1.0)
            else:
                # Landing preparation
                land_factor = (progress - 0.8) / 0.2
                frame_keypoints = base_keypoints.copy()
                frame_keypoints[:, 1] += 0.2 * land_factor  # Body comes down
                frame_keypoints[13:, 1] += 0.1 * land_factor  # Prepare for impact
            
            sequence.append({
                'frame': i,
                'keypoints': frame_keypoints,
                'confidence': np.ones(17) * 0.9
            })
        
        return sequence
    
    def process_video(self, video_path, athlete, obstacle):
        """
        Main processing function for parkour video analysis
        
        Args:
            video_path: Path to video file or camera index
            athlete: ParkourAthlete object
            obstacle: ParkourObstacle object
        """
        print(f"Processing video for {athlete.name} on {obstacle.name}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default assumption
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            total_frames = 1000  # For webcam
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {fps} FPS, {total_frames} frames, {frame_width}x{frame_height}")
        
        # Prepare output video
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (frame_width, frame_height))
        
        # Analysis variables
        frame_count = 0
        start_time = time.time()
        athlete_track_id = None
        motion_data = []
        slow_motion_frames = []
        
        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                # For webcam, break only if explicitly stopped
                if isinstance(video_path, int) or video_path == 0:
                    continue
                else:
                    break
            
            # Process frame
            processed_frame, frame_data = self.process_frame(
                frame, frame_count, fps, obstacle, athlete_track_id
            )
            
            # Update athlete tracking ID
            if athlete_track_id is None and frame_data.get('primary_track_id'):
                athlete_track_id = frame_data['primary_track_id']
            
            # Store motion data
            if frame_data.get('keypoints') is not None:
                motion_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'keypoints': frame_data['keypoints'],
                    'bbox': frame_data.get('bbox'),
                    'speed': frame_data.get('speed', 0),
                    'errors': frame_data.get('errors', [])
                })
                
                # Check for slow-motion triggers
                if frame_data.get('errors'):
                    slow_motion_frames.append(frame_count)
            
            # Add visualization overlays
            processed_frame = self.add_visualization_overlays(
                processed_frame, frame_data, obstacle, frame_count
            )
            
            # Add pro reference overlay (ghost)
            if frame_count % 5 == 0 and frame_count < len(self.pro_reference['keypoints_sequence']):
                processed_frame = self.add_pro_reference_overlay(
                    processed_frame, frame_count, fps, frame_width, frame_height
                )
            
            # Write to output
            out.write(processed_frame)
            
            # Display progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({fps_processed:.1f} fps)")
            
            # Show live preview (scaled down)
            preview = cv2.resize(processed_frame, (1280, 720))
            cv2.imshow('Parkour Analysis', preview)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # For webcam, limit to 1000 frames for demo
            if isinstance(video_path, int) and frame_count > 1000:
                print("Demo limit reached (1000 frames)")
                break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Perform comprehensive analysis
        analysis_results = self.analyze_performance(
            motion_data, obstacle, athlete, fps
        )
        
        # Generate slow-motion replay
        if slow_motion_frames:
            slowmo_path = self.generate_slow_motion_replay(
                video_path, slow_motion_frames, output_path, fps
            )
            if slowmo_path:
                analysis_results['slow_motion_path'] = slowmo_path
        
        # Generate feedback report
        report_path = self.generate_feedback_report(
            analysis_results, athlete, obstacle, output_path
        )
        
        # Store attempt data
        attempt_data = {
            'timestamp': datetime.now(),
            'video_path': output_path,
            'report_path': report_path,
            'agility_score': analysis_results['agility_score'],
            'safety_score': analysis_results['safety_score'],
            'accuracy_score': analysis_results['accuracy_score'],
            'completion_time': analysis_results['completion_time'],
            'technical_errors': analysis_results['technical_errors']
        }
        
        if 'slow_motion_path' in analysis_results:
            attempt_data['slow_motion_path'] = analysis_results['slow_motion_path']
        
        athlete.add_attempt(attempt_data)
        
        print(f"Analysis complete! Results saved to {output_path}")
        if report_path:
            print(f"Report generated: {report_path}")
        
        return analysis_results
    
    def process_frame(self, frame, frame_id, fps, obstacle, primary_track_id=None):
        """Process a single frame for parkour analysis"""
        frame_data = {
            'frame_id': frame_id,
            'timestamp': frame_id / fps,
            'detections': [],
            'keypoints': None,
            'errors': [],
            'metrics': {}
        }
        
        # Downscale for processing (for speed with 4K)
        orig_h, orig_w = frame.shape[:2]
        proc_frame = cv2.resize(frame, (orig_w // 2, orig_h // 2))
        
        # Detect athletes using YOLOv11
        try:
            detections = self.yolo_model(proc_frame, classes=0, verbose=False)[0]  # class 0 = person
        except:
            # Fallback if YOLO fails
            return frame, frame_data
        
        if detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                if conf > 0.3:  # Lower confidence threshold for demo
                    # Scale back to original size
                    x1, y1, x2, y2 = box * 2
                    bbox = (x1, y1, x2, y2)
                    
                    # Assign or update track ID
                    track_id = self.update_tracking(bbox, frame_id)
                    
                    # Estimate pose using lightweight method
                    keypoints = self.pose_estimator.estimate_pose(frame, bbox)
                    
                    if keypoints is not None:
                        frame_data['keypoints'] = keypoints
                        
                        # Analyze pose for parkour-specific metrics
                        pose_analysis = self.analyze_pose(
                            keypoints, obstacle, frame_id
                        )
                        
                        # Calculate speed and agility metrics
                        if track_id in self.track_history:
                            speed = self.calculate_speed(
                                track_id, frame_id, fps
                            )
                            frame_data['speed'] = speed
                            frame_data['metrics']['speed'] = speed
                        
                        # Detect technical errors
                        errors = self.detect_technical_errors(
                            pose_analysis, obstacle
                        )
                        frame_data['errors'].extend(errors)
                        
                        # Check jump accuracy against target zones
                        if pose_analysis.get('feet_positions'):
                            accuracy = self.check_jump_accuracy(
                                pose_analysis['feet_positions'],
                                obstacle.target_zones
                            )
                            frame_data['metrics']['jump_accuracy'] = accuracy
                        
                        frame_data['detections'].append({
                            'track_id': track_id,
                            'bbox': bbox,
                            'confidence': conf,
                            'keypoints': keypoints,
                            'pose_analysis': pose_analysis
                        })
                        
                        if primary_track_id is None:
                            primary_track_id = track_id
                            frame_data['primary_track_id'] = track_id
                            frame_data['bbox'] = bbox
        
        # Draw visualizations on original frame
        visualized_frame = self.visualize_frame(frame, frame_data, obstacle)
        
        return visualized_frame, frame_data
    
    def update_tracking(self, bbox, frame_id):
        """Simple tracking using IoU and history"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Find best match among existing tracks
        best_match_id = None
        best_iou = 0.3  # IoU threshold
        
        for track_id, history in self.track_history.items():
            if len(history) > 0:
                last_bbox = history[-1]['bbox']
                iou = self.calculate_iou(bbox, last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
        
        if best_match_id is not None:
            # Update existing track
            self.track_history[best_match_id].append({
                'frame_id': frame_id,
                'bbox': bbox,
                'center': center,
                'timestamp': time.time()
            })
            return best_match_id
        else:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            self.track_history[track_id] = deque(maxlen=self.max_history)
            self.track_history[track_id].append({
                'frame_id': frame_id,
                'bbox': bbox,
                'center': center,
                'timestamp': time.time()
            })
            return track_id
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return iou
    
    def analyze_pose(self, keypoints, obstacle, frame_id):
        """Analyze pose for parkour-specific metrics"""
        analysis = {
            'joint_angles': {},
            'body_alignment': 0,
            'landing_depth': 0,
            'feet_positions': [],
            'hesitation_score': 0
        }
        
        # Calculate joint angles
        if keypoints is not None and len(keypoints) >= 17:
            # Knee angles (left and right) - indices: hip(11/12), knee(13/14), ankle(15/16)
            left_knee_angle = self.calculate_angle(
                keypoints[11], keypoints[13], keypoints[15]
            )
            right_knee_angle = self.calculate_angle(
                keypoints[12], keypoints[14], keypoints[16]
            )
            
            analysis['joint_angles']['left_knee'] = left_knee_angle
            analysis['joint_angles']['right_knee'] = right_knee_angle
            
            # Hip angles - indices: shoulder(5/6), hip(11/12), knee(13/14)
            left_hip_angle = self.calculate_angle(
                keypoints[5], keypoints[11], keypoints[13]
            )
            right_hip_angle = self.calculate_angle(
                keypoints[6], keypoints[12], keypoints[14]
            )
            
            analysis['joint_angles']['left_hip'] = left_hip_angle
            analysis['joint_angles']['right_hip'] = right_hip_angle
            
            # Body alignment (vertical alignment score)
            shoulder_center = (keypoints[5] + keypoints[6]) / 2
            hip_center = (keypoints[11] + keypoints[12]) / 2
            ankle_center = (keypoints[15] + keypoints[16]) / 2
            
            # Calculate deviation from vertical line
            vertical_deviation = abs(shoulder_center[0] - ankle_center[0])
            analysis['body_alignment'] = max(0, 100 - vertical_deviation * 50)
            
            # Feet positions (ankles)
            analysis['feet_positions'] = [keypoints[15][:2], keypoints[16][:2]]
            
            # Track height for landing depth
            if not hasattr(self, 'min_height'):
                self.min_height = float('inf')
            if not hasattr(self, 'max_height_frame'):
                self.max_height_frame = -1
            
            current_height = min(keypoints[15][1], keypoints[16][1])
            if current_height < self.min_height:
                self.min_height = current_height
                self.max_height_frame = frame_id
            
            if frame_id > self.max_height_frame:
                analysis['landing_depth'] = current_height - self.min_height
        
        return analysis
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (in 2D)"""
        # Ensure we have valid points
        if len(a) < 2 or len(b) < 2 or len(c) < 2:
            return 90  # Default angle
        
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])
        
        ba = a - b
        bc = c - b
        
        # Check for zero vectors
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm == 0 or bc_norm == 0:
            return 90
        
        cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def calculate_speed(self, track_id, frame_id, fps):
        """Calculate athlete speed based on tracking history"""
        if track_id in self.track_history and len(self.track_history[track_id]) > 1:
            history = list(self.track_history[track_id])
            
            # Get recent positions
            recent_frames = min(5, len(history))
            if recent_frames > 1:
                start_frame = history[-recent_frames]
                end_frame = history[-1]
                
                # Calculate pixel distance
                start_center = start_frame['center']
                end_center = end_frame['center']
                
                pixel_distance = np.sqrt(
                    (end_center[0] - start_center[0])**2 + 
                    (end_center[1] - start_center[1])**2
                )
                
                # Convert to meters (rough approximation)
                # Assuming athlete height ~1.7m = ~200 pixels
                meters_per_pixel = 1.7 / 200
                distance_meters = pixel_distance * meters_per_pixel
                
                # Calculate time
                time_seconds = recent_frames / fps
                
                # Speed in m/s
                speed = distance_meters / time_seconds if time_seconds > 0 else 0
                
                return speed
        
        return 0
    
    def detect_technical_errors(self, pose_analysis, obstacle):
        """Detect technical errors in parkour technique"""
        errors = []
        optimal_angles = obstacle.optimal_joint_angles
        
        # Check knee angles for safety
        left_knee = pose_analysis['joint_angles'].get('left_knee', 90)
        right_knee = pose_analysis['joint_angles'].get('right_knee', 90)
        
        if 'knee_angle_min' in optimal_angles:
            if left_knee < optimal_angles['knee_angle_min']:
                errors.append(f"Left knee angle too acute: {left_knee:.1f}° (min {optimal_angles['knee_angle_min']}°)")
            if right_knee < optimal_angles['knee_angle_min']:
                errors.append(f"Right knee angle too acute: {right_knee:.1f}° (min {optimal_angles['knee_angle_min']}°)")
        
        if 'knee_angle_max' in optimal_angles:
            if left_knee > optimal_angles['knee_angle_max']:
                errors.append(f"Left knee hyperextended: {left_knee:.1f}° (max {optimal_angles['knee_angle_max']}°)")
            if right_knee > optimal_angles['knee_angle_max']:
                errors.append(f"Right knee hyperextended: {right_knee:.1f}° (max {optimal_angles['knee_angle_max']}°)")
        
        # Check body alignment
        if pose_analysis['body_alignment'] < 70:
            errors.append(f"Poor body alignment: {pose_analysis['body_alignment']:.1f}%")
        
        # Check landing depth
        if 'landing_depth_max' in optimal_angles:
            if pose_analysis['landing_depth'] > optimal_angles['landing_depth_max'] * 10:  # Convert to pixels
                errors.append(f"Deep landing: {pose_analysis['landing_depth']:.1f}px")
        
        return errors
    
    def check_jump_accuracy(self, feet_positions, target_zones):
        """Check accuracy of landing relative to target zones"""
        if not feet_positions or not target_zones:
            return 0
        
        left_foot, right_foot = feet_positions
        
        # Find closest target zone
        min_distance = float('inf')
        for zone in target_zones:
            zone_center = ((zone[0] + zone[2]) / 2, (zone[1] + zone[3]) / 2)
            
            # Calculate distance from each foot to zone center
            left_dist = np.sqrt(
                (left_foot[0] - zone_center[0])**2 + 
                (left_foot[1] - zone_center[1])**2
            )
            right_dist = np.sqrt(
                (right_foot[0] - zone_center[0])**2 + 
                (right_foot[1] - zone_center[1])**2
            )
            
            avg_dist = (left_dist + right_dist) / 2
            min_distance = min(min_distance, avg_dist)
        
        # Convert distance to accuracy score (0-100)
        # Assuming 50 pixels = good accuracy threshold
        accuracy = max(0, 100 - (min_distance * 0.5))
        
        return accuracy
    
    def visualize_frame(self, frame, frame_data, obstacle):
        """Add visualizations to frame"""
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        
        # Draw target zones
        for zone in obstacle.target_zones:
            cv2.rectangle(viz_frame, 
                         (int(zone[0]), int(zone[1])), 
                         (int(zone[2]), int(zone[3])), 
                         self.colors['target_zone'], 2)
            cv2.putText(viz_frame, "Target Zone", 
                       (int(zone[0]), int(zone[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['target_zone'], 2)
        
        # Draw detections and keypoints
        for detection in frame_data['detections']:
            bbox = detection['bbox']
            track_id = detection['track_id']
            
            # Draw bounding box
            cv2.rectangle(viz_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         self.colors['athlete'], 2)
            
            # Draw track ID
            cv2.putText(viz_frame, f"Athlete {track_id}", 
                       (int(bbox[0]), int(bbox[1]) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['athlete'], 2)
            
            # Draw keypoints if available
            if detection.get('keypoints') is not None:
                keypoints = detection['keypoints']
                for i, kp in enumerate(keypoints):
                    if len(kp) >= 3 and kp[2] > 0.3:  # Confidence threshold
                        cv2.circle(viz_frame, 
                                 (int(kp[0]), int(kp[1])), 
                                 4, self.colors['athlete'], -1)
                
                # Draw skeleton
                for start, end in self.pose_estimator.skeleton:
                    if (len(keypoints) > max(start, end) and 
                        len(keypoints[start]) >= 3 and len(keypoints[end]) >= 3 and
                        keypoints[start][2] > 0.3 and keypoints[end][2] > 0.3):
                        cv2.line(viz_frame,
                               (int(keypoints[start][0]), int(keypoints[start][1])),
                               (int(keypoints[end][0]), int(keypoints[end][1])),
                               self.colors['athlete'], 2)
        
        # Display metrics
        y_offset = 50
        for metric_name, metric_value in frame_data.get('metrics', {}).items():
            cv2.putText(viz_frame, f"{metric_name}: {metric_value:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Display errors
        if frame_data['errors']:
            error_y = h - 100
            cv2.putText(viz_frame, "Errors detected:", 
                       (10, error_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['error'], 2)
            for i, error in enumerate(frame_data['errors'][:3]):  # Show max 3 errors
                cv2.putText(viz_frame, f"- {error[:40]}", 
                           (10, error_y + 25 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['error'], 1)
        
        return viz_frame
    
    def add_visualization_overlays(self, frame, frame_data, obstacle, frame_id):
        """Add additional visualization overlays"""
        overlay_frame = frame.copy()
        
        # Add speed indicator
        if 'speed' in frame_data.get('metrics', {}):
            speed = frame_data['metrics']['speed']
            
            # Create speed gauge
            gauge_center = (frame.shape[1] - 100, 100)
            gauge_radius = 40
            
            # Draw gauge background
            cv2.circle(overlay_frame, gauge_center, gauge_radius, (50, 50, 50), -1)
            
            # Draw speed needle
            max_speed = 10  # m/s
            angle = (speed / max_speed) * 270  # 0-270 degrees
            
            if angle > 270:
                angle = 270
            
            end_x = gauge_center[0] + int(gauge_radius * 0.8 * np.cos(np.radians(angle - 90)))
            end_y = gauge_center[1] + int(gauge_radius * 0.8 * np.sin(np.radians(angle - 90)))
            
            cv2.line(overlay_frame, gauge_center, (end_x, end_y), 
                    (0, 255, 0) if speed < 8 else (0, 165, 255) if speed < 9 else (0, 0, 255), 
                    3)
            
            # Add speed text
            cv2.putText(overlay_frame, f"{speed:.1f} m/s", 
                       (gauge_center[0] - 30, gauge_center[1] + gauge_radius + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(overlay_frame, f"Frame: {frame_id}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay_frame
    
    def add_pro_reference_overlay(self, frame, frame_id, fps, frame_width, frame_height):
        """Add semi-transparent pro athlete reference overlay"""
        overlay = frame.copy()
        
        # Get pro reference pose for this frame
        ref_idx = min(frame_id // 2, len(self.pro_reference['keypoints_sequence']) - 1)
        ref_pose = self.pro_reference['keypoints_sequence'][ref_idx]
        
        # Draw pro reference as ghost (semi-transparent)
        alpha = 0.3  # Transparency
        
        if 'keypoints' in ref_pose:
            keypoints = ref_pose['keypoints']
            
            # Scale keypoints from normalized to frame coordinates
            scaled_keypoints = keypoints.copy()
            scaled_keypoints[:, 0] *= frame_width
            scaled_keypoints[:, 1] *= frame_height
            
            # Draw skeleton for pro reference
            for start, end in self.pose_estimator.skeleton:
                if (len(scaled_keypoints) > max(start, end) and 
                    scaled_keypoints[start][2] > 0.3 and scaled_keypoints[end][2] > 0.3):
                    start_pt = (int(scaled_keypoints[start][0]), 
                               int(scaled_keypoints[start][1]))
                    end_pt = (int(scaled_keypoints[end][0]), 
                             int(scaled_keypoints[end][1]))
                    
                    # Draw semi-transparent line
                    overlay_with_line = overlay.copy()
                    cv2.line(overlay_with_line, start_pt, end_pt, 
                            self.colors['pro_reference'][:3], 2, cv2.LINE_AA)
                    
                    # Blend
                    cv2.addWeighted(overlay_with_line, alpha, overlay, 1 - alpha, 0, overlay)
        
        # Add label
        cv2.putText(overlay, "Pro Reference (Dom Tomato)", 
                   (frame_width - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['pro_reference'][:3], 2)
        
        return overlay
    
    def analyze_performance(self, motion_data, obstacle, athlete, fps):
        """Comprehensive performance analysis"""
        print("Analyzing performance...")
        
        if not motion_data:
            return {
                'agility_score': 0,
                'safety_score': 0,
                'accuracy_score': 0,
                'completion_time': 0,
                'technical_errors': [],
                'detailed_metrics': {}
            }
        
        # Extract metrics
        speeds = [m.get('speed', 0) for m in motion_data]
        errors = [err for m in motion_data for err in m.get('errors', [])]
        accuracy_scores = [m.get('metrics', {}).get('jump_accuracy', 0) 
                          for m in motion_data if 'metrics' in m]
        
        # Calculate agility score (based on speed consistency and flow)
        avg_speed = np.mean(speeds) if speeds else 0
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        
        # Agility: high average speed with low variability = good flow
        agility_score = max(0, min(100, avg_speed * 10 - speed_std * 5))
        
        # Calculate safety score (based on technical errors)
        error_penalty = min(30, len(set(errors)) * 5)  # Max 30 point penalty
        safety_score = max(0, 100 - error_penalty)
        
        # Calculate accuracy score
        accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 0
        
        # Calculate completion time
        completion_time = len(motion_data) / fps if fps > 0 else 0
        
        # Detailed metrics
        detailed_metrics = {
            'avg_speed_mps': avg_speed,
            'speed_consistency': 100 - min(100, speed_std * 10),
            'max_speed': max(speeds) if speeds else 0,
            'min_speed': min(speeds) if speeds else 0,
            'total_errors': len(errors),
            'unique_errors': len(set(errors)),
            'peak_accuracy': max(accuracy_scores) if accuracy_scores else 0,
            'flow_consistency': self.calculate_flow_consistency(motion_data)
        }
        
        # Compare with pro reference
        pro_comparison = self.compare_with_pro_reference(motion_data, fps)
        
        results = {
            'agility_score': agility_score,
            'safety_score': safety_score,
            'accuracy_score': accuracy_score,
            'completion_time': completion_time,
            'technical_errors': list(set(errors)),  # Unique errors
            'detailed_metrics': detailed_metrics,
            'pro_comparison': pro_comparison,
            'motion_data_summary': {
                'total_frames': len(motion_data),
                'has_landing': len(accuracy_scores) > 0,
                'performance_level': self.assess_performance_level(
                    agility_score, safety_score, accuracy_score
                )
            }
        }
        
        return results
    
    def calculate_flow_consistency(self, motion_data):
        """Calculate flow consistency (how smooth the movement is)"""
        if len(motion_data) < 5:
            return 0
        
        # Extract speed changes
        speeds = [m.get('speed', 0) for m in motion_data]
        
        # Calculate acceleration (derivative of speed)
        if len(speeds) > 1:
            acceleration = np.diff(speeds)
            # Smooth acceleration indicates good flow
            acceleration_std = np.std(acceleration) if len(acceleration) > 1 else 0
            flow_score = max(0, 100 - acceleration_std * 20)
            return flow_score
        
        return 0
    
    def compare_with_pro_reference(self, motion_data, fps):
        """Compare athlete performance with pro reference"""
        comparison = {
            'speed_ratio': 0,
            'timing_difference': 0,
            'technique_similarity': 0,
            'overall_match': 0
        }
        
        if not motion_data:
            return comparison
        
        # Compare timing
        athlete_duration = len(motion_data) / fps
        pro_duration = sum(self.pro_reference['timing'].values())
        
        comparison['timing_difference'] = athlete_duration - pro_duration
        
        # Compare speed profile
        athlete_speeds = [m.get('speed', 0) for m in motion_data]
        if athlete_speeds and max(athlete_speeds) > 0:
            # Normalize speeds
            athlete_norm = np.array(athlete_speeds) / max(athlete_speeds)
            pro_norm = np.array(self.pro_reference['speed_profile'])
            
            # Pad or truncate to same length
            min_len = min(len(athlete_norm), len(pro_norm))
            if min_len > 0:
                similarity = 1 - np.mean(np.abs(
                    athlete_norm[:min_len] - pro_norm[:min_len]
                ))
                comparison['technique_similarity'] = similarity * 100
        
        # Overall match score
        comparison['overall_match'] = (
            comparison['technique_similarity'] * 0.6 +
            max(0, 100 - abs(comparison['timing_difference']) * 10) * 0.4
        )
        
        return comparison
    
    def assess_performance_level(self, agility, safety, accuracy):
        """Assess overall performance level"""
        avg_score = (agility + safety + accuracy) / 3
        
        if avg_score >= 85:
            return "Professional"
        elif avg_score >= 70:
            return "Advanced"
        elif avg_score >= 50:
            return "Intermediate"
        else:
            return "Beginner"
    
    def generate_slow_motion_replay(self, input_path, trigger_frames, output_path, fps):
        """Generate slow-motion replay of detected mistakes"""
        print("Generating slow-motion replay...")
        
        # Create slow motion segments
        slow_motion_path = output_path.replace('.mp4', '_slowmo.mp4')
        
        try:
            # Try to create a simple slow-motion clip
            if isinstance(input_path, int):  # Webcam
                print("Cannot create slow-motion from webcam in demo")
                return None
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print("Cannot open input video for slow-motion")
                return None
            
            # Get frame dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(slow_motion_path, fourcc, fps // 4, 
                                (frame_width, frame_height))
            
            # Process trigger frames
            processed_frames = set()
            for frame_num in trigger_frames[:10]:  # Limit to first 10 triggers
                if frame_num in processed_frames:
                    continue
                
                # Get frames around the error (±15 frames)
                start_frame = max(0, frame_num - 15)
                end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_num + 15)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for i in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Add slow-motion indicator
                    cv2.putText(frame, "SLOW MOTION REPLAY", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, (0, 0, 255), 3)
                    
                    # Write multiple times for slow motion (4x slower)
                    for _ in range(4):  # 0.25x speed
                        out.write(frame)
                    
                    processed_frames.add(i)
            
            cap.release()
            out.release()
            
            if len(processed_frames) > 0:
                print(f"Slow-motion replay saved: {slow_motion_path}")
                return slow_motion_path
            else:
                os.remove(slow_motion_path)
                return None
            
        except Exception as e:
            print(f"Error creating slow-motion: {e}")
            return None
    
    def generate_feedback_report(self, analysis_results, athlete, obstacle, video_path):
        """Generate PDF feedback report"""
        print("Generating feedback report...")
        
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, 
                                  f"report_{athlete.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        
        # Create PDF document
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"Parkour Performance Report - {athlete.name}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Athlete Info
        athlete_info = [
            ["Athlete Information", ""],
            ["Name:", athlete.name],
            ["Age:", str(athlete.age)],
            ["Height:", f"{athlete.height} cm"],
            ["Weight:", f"{athlete.weight} kg"],
            ["Skill Level:", athlete.skill_level],
            ["Obstacle:", obstacle.name],
            ["Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        athlete_table = Table(athlete_info, colWidths=[150, 200])
        athlete_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(athlete_table)
        story.append(Spacer(1, 20))
        
        # Performance Scores
        scores = [
            ["Performance Metrics", "Score", "Grade"],
            ["Agility Score", f"{analysis_results['agility_score']:.1f}/100", 
             self._get_grade(analysis_results['agility_score'])],
            ["Safety Score", f"{analysis_results['safety_score']:.1f}/100", 
             self._get_grade(analysis_results['safety_score'])],
            ["Accuracy Score", f"{analysis_results['accuracy_score']:.1f}/100", 
             self._get_grade(analysis_results['accuracy_score'])],
            ["Completion Time", f"{analysis_results['completion_time']:.2f}s", ""],
            ["Overall Level", analysis_results['motion_data_summary']['performance_level'], ""]
        ]
        
        scores_table = Table(scores, colWidths=[180, 80, 80])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(scores_table)
        story.append(Spacer(1, 20))
        
        # Detailed Metrics
        if 'detailed_metrics' in analysis_results:
            details = Paragraph("Detailed Performance Metrics:", styles['Heading2'])
            story.append(details)
            
            detail_items = []
            for key, value in analysis_results['detailed_metrics'].items():
                if isinstance(value, float):
                    detail_items.append([key.replace('_', ' ').title(), f"{value:.2f}"])
                else:
                    detail_items.append([key.replace('_', ' ').title(), str(value)])
            
            if detail_items:
                detail_table = Table(detail_items, colWidths=[200, 100])
                detail_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                story.append(detail_table)
                story.append(Spacer(1, 20))
        
        # Technical Errors
        if analysis_results['technical_errors']:
            errors_title = Paragraph("Technical Errors Detected:", styles['Heading2'])
            story.append(errors_title)
            
            for i, error in enumerate(analysis_results['technical_errors'][:10], 1):
                error_text = Paragraph(f"{i}. {error}", styles['Normal'])
                story.append(error_text)
            
            story.append(Spacer(1, 20))
        
        # Personalized Feedback
        feedback = self._generate_personalized_feedback(analysis_results, athlete)
        feedback_title = Paragraph("Personalized Feedback:", styles['Heading2'])
        story.append(feedback_title)
        
        for item in feedback:
            story.append(Paragraph(f"• {item}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"Report generated: {report_path}")
            return report_path
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            # Create a simple text report as fallback
            text_report_path = report_path.replace('.pdf', '.txt')
            self._create_text_report(text_report_path, analysis_results, athlete, obstacle)
            return text_report_path
    
    def _create_text_report(self, filepath, analysis_results, athlete, obstacle):
        """Create a simple text report as fallback"""
        with open(filepath, 'w') as f:
            f.write(f"Parkour Performance Report - {athlete.name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Athlete: {athlete.name}\n")
            f.write(f"Age: {athlete.age}\n")
            f.write(f"Height: {athlete.height} cm\n")
            f.write(f"Weight: {athlete.weight} kg\n")
            f.write(f"Skill Level: {athlete.skill_level}\n")
            f.write(f"Obstacle: {obstacle.name}\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            f.write("Performance Scores:\n")
            f.write(f"  Agility: {analysis_results['agility_score']:.1f}/100\n")
            f.write(f"  Safety: {analysis_results['safety_score']:.1f}/100\n")
            f.write(f"  Accuracy: {analysis_results['accuracy_score']:.1f}/100\n")
            f.write(f"  Completion Time: {analysis_results['completion_time']:.2f}s\n")
            f.write(f"  Overall Level: {analysis_results['motion_data_summary']['performance_level']}\n\n")
            
            if analysis_results['technical_errors']:
                f.write("Technical Errors:\n")
                for i, error in enumerate(analysis_results['technical_errors'], 1):
                    f.write(f"  {i}. {error}\n")
                f.write("\n")
            
            f.write("Feedback:\n")
            feedback = self._generate_personalized_feedback(analysis_results, athlete)
            for item in feedback:
                f.write(f"  • {item}\n")
        
        print(f"Text report generated: {filepath}")
        return filepath
    
    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 50:
            return "C"
        else:
            return "Needs Work"
    
    def _generate_personalized_feedback(self, analysis_results, athlete):
        """Generate personalized, athlete-friendly feedback"""
        feedback = []
        
        # Agility feedback
        agility = analysis_results['agility_score']
        if agility >= 80:
            feedback.append("Excellent agility and flow! Your movement is smooth and efficient.")
        elif agility >= 60:
            feedback.append("Good agility. Focus on maintaining consistent speed through the entire run.")
        else:
            feedback.append("Work on improving your flow. Try to reduce hesitation and maintain momentum.")
        
        # Safety feedback
        safety = analysis_results['safety_score']
        errors = analysis_results['technical_errors']
        
        if safety >= 90:
            feedback.append("Great technique with excellent safety awareness!")
        elif safety >= 70:
            feedback.append("Good technique overall. Pay attention to the following areas:")
            for error in errors[:2]:
                feedback.append(f"  - {error}")
        else:
            feedback.append("Focus on safety fundamentals:")
            for error in errors[:3]:
                feedback.append(f"  - {error}")
            feedback.append("Consider practicing at a slower pace to perfect form before adding speed.")
        
        # Accuracy feedback
        accuracy = analysis_results['accuracy_score']
        if accuracy >= 85:
            feedback.append("Outstanding accuracy! Your landing precision is excellent.")
        elif accuracy >= 65:
            feedback.append("Good accuracy. Work on consistent foot placement in target zones.")
        else:
            feedback.append("Accuracy needs improvement. Practice landing drills to build spatial awareness.")
        
        # Pro comparison feedback
        if 'pro_comparison' in analysis_results:
            pro_match = analysis_results['pro_comparison'].get('overall_match', 0)
            if pro_match >= 80:
                feedback.append("Your technique closely matches professional standards!")
            elif pro_match >= 60:
                feedback.append("You're developing good technique. Study pro athletes for subtle refinements.")
            else:
                feedback.append("Watch pro athletes' form videos to understand optimal movement patterns.")
        
        # Skill-specific feedback based on athlete level
        if athlete.skill_level.lower() == 'beginner':
            feedback.append("As a beginner, focus on mastering basic safety before adding complexity.")
        elif athlete.skill_level.lower() == 'intermediate':
            feedback.append("As an intermediate athlete, work on consistency and flow between obstacles.")
        elif athlete.skill_level.lower() in ['advanced', 'pro']:
            feedback.append("At your level, focus on micro-adjustments and efficiency gains.")
        
        return feedback

class ProgressTracker:
    """Tracks athlete progress over time"""
    
    def __init__(self):
        self.athletes = {}
        self.obstacles = {}
    
    def add_athlete(self, athlete):
        """Add athlete to tracking system"""
        self.athletes[athlete.id] = athlete
    
    def add_obstacle(self, obstacle):
        """Add obstacle to system"""
        self.obstacles[obstacle.name] = obstacle
    
    def generate_progress_charts(self, athlete_id, output_dir="progress_charts"):
        """Generate progress charts for an athlete"""
        if athlete_id not in self.athletes:
            print(f"Athlete {athlete_id} not found")
            return
        
        athlete = self.athletes[athlete_id]
        data = athlete.get_progress_chart_data()
        
        if len(data['dates']) < 2:
            print("Need at least 2 attempts to generate progress charts")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Progress Charts - {athlete.name}", fontsize=16)
        
        # Convert dates to string for plotting
        date_labels = []
        for d in data['dates']:
            if hasattr(d, 'strftime'):
                date_labels.append(d.strftime("%m/%d"))
            else:
                date_labels.append(str(d))
        
        # Chart 1: Overall Scores
        axes[0, 0].plot(range(len(date_labels)), data['agility'], 'o-', label='Agility', linewidth=2)
        axes[0, 0].plot(range(len(date_labels)), data['safety'], 's-', label='Safety', linewidth=2)
        axes[0, 0].plot(range(len(date_labels)), data['accuracy'], '^-', label='Accuracy', linewidth=2)
        axes[0, 0].set_title('Performance Scores Over Time')
        axes[0, 0].set_ylabel('Score (0-100)')
        axes[0, 0].set_xticks(range(len(date_labels)))
        axes[0, 0].set_xticklabels(date_labels, rotation=45)
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Chart 2: Completion Times
        axes[0, 1].plot(range(len(date_labels)), data['times'], 'o-', color='purple', linewidth=2)
        axes[0, 1].set_title('Completion Time Over Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_xticks(range(len(date_labels)))
        axes[0, 1].set_xticklabels(date_labels, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Chart 3: Improvement bars
        if len(data['dates']) > 1:
            improvements = [
                data['agility'][-1] - data['agility'][0],
                data['safety'][-1] - data['safety'][0],
                data['accuracy'][-1] - data['accuracy'][0]
            ]
            
            bars = axes[1, 0].bar(['Agility', 'Safety', 'Accuracy'], improvements,
                                 color=['green', 'blue', 'orange'])
            axes[1, 0].set_title('Total Improvement Since First Attempt')
            axes[1, 0].set_ylabel('Score Improvement')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:+.1f}', ha='center', va='bottom')
        
        # Chart 4: Latest performance breakdown
        if len(data['dates']) > 0:
            latest_scores = [data['agility'][-1], data['safety'][-1], data['accuracy'][-1]]
            labels = ['Agility', 'Safety', 'Accuracy']
            
            axes[1, 1].pie(latest_scores, labels=labels, autopct='%1.1f%%',
                          colors=['green', 'blue', 'orange'])
            axes[1, 1].set_title(f'Latest Performance Breakdown\n({date_labels[-1]})')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, f"{athlete.name}_progress_charts.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Progress charts saved: {chart_path}")
        return chart_path
def _display_results( results):
    """Display analysis results"""
    print("\n=== Analysis Results ===")
    print(f"Agility Score: {results['agility_score']:.1f}/100")
    print(f"Safety Score: {results['safety_score']:.1f}/100")
    print(f"Accuracy Score: {results['accuracy_score']:.1f}/100")
    print(f"Completion Time: {results['completion_time']:.2f}s")
    print(f"Performance Level: {results['motion_data_summary']['performance_level']}")
    
    if results['technical_errors']:
        print(f"\nTechnical Errors ({len(results['technical_errors'])}):")
        for error in results['technical_errors'][:3]:
            print(f"  - {error}")
    else:
        print("\nNo technical errors detected!")
    
    # Compare with pro
    if 'pro_comparison' in results:
        pro_match = results['pro_comparison'].get('overall_match', 0)
        print(f"\nPro Comparison: {pro_match:.1f}% match with Dom Tomato's technique")

def _view_progress(athlete):
    """View athlete progress"""
    print(f"\n=== {athlete.name}'s Progress ===")
    print(f"Total Attempts: {len(athlete.attempts)}")
    
    if athlete.attempts:
        latest = athlete.attempts[-1]
        print(f"\nLatest Attempt ({latest['timestamp']}):")
        print(f"  Agility: {latest['agility_score']:.1f}")
        print(f"  Safety: {latest['safety_score']:.1f}")
        print(f"  Accuracy: {latest['accuracy_score']:.1f}")
        print(f"  Time: {latest['completion_time']:.2f}s")
        
        if len(athlete.attempts) > 1:
            first = athlete.attempts[0]
            print(f"\nImprovement since first attempt:")
            print(f"  Agility: +{latest['agility_score'] - first['agility_score']:.1f}")
            print(f"  Safety: +{latest['safety_score'] - first['safety_score']:.1f}")
            print(f"  Accuracy: +{latest['accuracy_score'] - first['accuracy_score']:.1f}")
            if latest['completion_time'] and first['completion_time']:
                print(f"  Time: {first['completion_time'] - latest['completion_time']:.2f}s faster")
    else:
        print("No attempts recorded yet.")

def _add_new_athlete(tracker):
    """Add new athlete"""
    print("\n=== Add New Athlete ===")
    name = input("Name: ").strip()
    
    try:
        age = int(input("Age: ").strip())
        weight = float(input("Weight (kg): ").strip())
        height = float(input("Height (cm): ").strip())
        skill_level = input("Skill Level (beginner/intermediate/advanced/pro): ").strip().lower()
        
        # Validate skill level
        if skill_level not in ['beginner', 'intermediate', 'advanced', 'pro']:
            skill_level = 'intermediate'
        
        new_athlete = ParkourAthlete(
            athlete_id=len(tracker.athletes) + 1,
            name=name,
            age=age,
            weight=weight,
            height=height,
            skill_level=skill_level
        )
        tracker.add_athlete(new_athlete)
        print(f"Athlete {name} added successfully!")
    
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def main():
    """Main demo function"""
    print("=== Parkour Analysis Framework ===")
    print("Real-Time Assessment + Progress Tracking")
    print("No complex dependencies - using lightweight methods")
    print("=" * 50)
    
    # Initialize system
    analyzer = ParkourAnalyzer()
    tracker = ProgressTracker()
    
    # Create sample athlete
    athlete = ParkourAthlete(
        athlete_id=1,
        name="Sample Athlete",
        age=24,
        weight=75,
        height=180,
        skill_level="Intermediate"
    )
    tracker.add_athlete(athlete)
    
    # Create sample obstacle (Kong Vault)
    obstacle = ParkourObstacle(
        name="Kong Vault Box",
        obstacle_type="vault",
        difficulty=6,
        target_zones=[
            (300, 200, 500, 250),  # Takeoff zone
            (600, 180, 800, 220),  # Landing zone
            (100, 200, 150,250)
        ]
    )
    tracker.add_obstacle(obstacle)
    
    # Menu
    while True:
        print("\n=== Main Menu ===")
        print("1. Analyze Parkour Video (Webcam Demo)")
        print("2. Analyze Parkour Video (File)")
        print("3. View Athlete Progress")
        print("4. Generate Progress Charts")
        print("5. Add New Athlete")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            # Analyze webcam
            print("\nStarting webcam analysis...")
            print("Press 'q' to stop during processing")
            print("Make some parkour-like movements in front of the camera")
            
            try:
                results = analyzer.process_video(0, athlete, obstacle)
                
                if results:
                    _display_results(results)
            
            except Exception as e:
                print(f"Error during analysis: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == "2":
            # Analyze video file
            video_path = input("Enter video file path: ").strip()
            
            if not os.path.exists(video_path):
                print(f"File not found: {video_path}")
                print("Using sample webcam instead...")
                video_path = 0
            
            try:
                print("\nStarting analysis... (Press 'q' to quit during processing)")
                
                results = analyzer.process_video(video_path, athlete, obstacle)
                
                if results:
                    _display_results(results)
            
            except Exception as e:
                print(f"Error during analysis: {e}")
        
        elif choice == "3":
            # View progress
            _view_progress(athlete)
        
        elif choice == "4":
            # Generate charts
            if len(athlete.attempts) >= 2:
                chart_path = tracker.generate_progress_charts(athlete.id)
                if chart_path:
                    print(f"Progress charts generated: {chart_path}")
            else:
                print("Need at least 2 attempts to generate progress charts.")
        
        elif choice == "5":
            # Add new athlete
            _add_new_athlete(tracker)
        
        elif choice == "6":
            print("Exiting Parkour Analysis Framework. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")



# Monkey patch methods to main for simplicity
main._display_results = _display_results
main._view_progress = _view_progress
main._add_new_athlete = _add_new_athlete

if __name__ == "__main__":
    main()