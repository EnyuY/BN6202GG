#!/usr/bin/env python3
"""
Simplified Gesture Control System for Development with GUI Monitoring
简化手势控制系统 - 开发版本 (带GUI监控)

Gesture Controls:
1. E-BRAKE: Both hands open → Spacebar
2. ACCELERATE: Both hands closed (fists) → W key  
3. REVERSE: Right hand open, left hand closed → S key
4. START/IDLE: Right hand closed, left hand open → E key
5. STEERING: Hand height differences control steering (A/D keys)

GUI Features:
- Real-time video monitoring with hand landmarks
- Live gesture recognition status and confidence display
- FPS counter and performance monitoring
- On-screen gesture guide and instructions
- Interactive controls (F for FPS toggle, G for gesture info toggle)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import queue
import json
import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'

# Import pynput for keyboard control
try:
    import pynput
    from pynput import keyboard, mouse
    from pynput.keyboard import Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    print("Error: pynput not found. Please install: pip install pynput")
    PYNPUT_AVAILABLE = False

# Import pyautogui for additional input control
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    print("Warning: pyautogui not found. Some features may be limited.")
    PYAUTOGUI_AVAILABLE = False

# Simplified MediaPipe setup for development
print("MediaPipe initialized for development")

# Performance monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not found. Performance monitoring disabled.")
    PSUTIL_AVAILABLE = False

class GestureType(Enum):
    # Primary Gestures
    E_BRAKE = "e_brake"                    
    ACCELERATE = "accelerate"              
    REVERSE = "reverse"                    
    START_IDLE = "start_idle"              
    
    # Steering Gestures
    STEER_LEFT = "steer_left"              
    STEER_RIGHT = "steer_right"            
    
    # Combined Actions
    BRAKE_LEFT = "brake_left"              
    BRAKE_RIGHT = "brake_right"            
    ACCEL_LEFT = "accel_left"              
    ACCEL_RIGHT = "accel_right"            
    REVERSE_LEFT = "reverse_left"          
    REVERSE_RIGHT = "reverse_right"        
    IDLE_LEFT = "idle_left"                
    IDLE_RIGHT = "idle_right"              
    
    # Special States
    NO_GESTURE = "no_gesture"              
    UNKNOWN = "unknown"                    
    INVALID = "invalid"                    

class GameAction(Enum):
    ACCELERATE = "w"
    REVERSE = "s" 
    STEER_LEFT = "a"
    STEER_RIGHT = "d"
    BRAKE = "space"
    START_ENGINE = "e"
    HORN = "h"

@dataclass
class HandLandmarkData:
    landmarks: List[Tuple[float, float, float]]
    handedness: str
    confidence: float
    
    def __post_init__(self):
        # Simplified: only compute palm center
        self.palm_center = self._compute_palm_center()
    
    def _compute_palm_center(self) -> Tuple[float, float]:
        """Compute palm center for position tracking."""
        if len(self.landmarks) >= 21:
            # Use wrist (0) and middle finger base (9) for simple palm center
            wrist = self.landmarks[0]
            middle_base = self.landmarks[9]
            return ((wrist[0] + middle_base[0]) / 2, (wrist[1] + middle_base[1]) / 2)
        return (0.5, 0.5)
    
    def is_fist(self, threshold: float = 0.8) -> bool:
        """Simplified fist detection using fingertip vs palm distances."""
        if len(self.landmarks) < 21:
            return False
        
        palm_y = self.palm_center[1]
        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        
        # Count fingertips that are close to palm level (indicating closed fingers)
        closed_count = 0
        for tip_idx in fingertip_indices:
            tip_y = self.landmarks[tip_idx][1]
            if abs(tip_y - palm_y) < 0.05:  # Close to palm level
                closed_count += 1
        
        return (closed_count / len(fingertip_indices)) >= threshold
    
    def is_open_hand(self, threshold: float = 0.6) -> bool:
        """Simplified open hand detection using fingertip spread."""
        if len(self.landmarks) < 21:
            return False
        
        palm_y = self.palm_center[1]
        fingertip_indices = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky (skip thumb)
        
        # Count fingertips that are extended (far from palm)
        extended_count = 0
        for tip_idx in fingertip_indices:
            tip_y = self.landmarks[tip_idx][1]
            if abs(tip_y - palm_y) > 0.08:  # Extended from palm
                extended_count += 1
        
        return (extended_count / len(fingertip_indices)) >= threshold

@dataclass
class GestureFrame:
    left_hand: Optional[HandLandmarkData]
    right_hand: Optional[HandLandmarkData]
    gesture_type: GestureType
    confidence: float
    timestamp: float
    frame_number: int
    processing_time_ms: float
    system_latency_ms: float = 0.0
    
    def has_both_hands(self) -> bool:
        return self.left_hand is not None and self.right_hand is not None
    
    def has_any_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None
    
    def get_hand_count(self) -> int:
        return sum([self.left_hand is not None, self.right_hand is not None])

class PerformanceMonitor:
    """Simplified performance monitoring for development."""
    
    def __init__(self, window_size: int = 100, monitoring_interval: float = 1.0):
        self.total_frames = 0
        self.total_gestures = 0
        self.start_time = time.time()
        self.frame_times = deque(maxlen=30)
    
    def record_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
        self.total_frames += 1
    
    def record_processing_time(self, process_time: float):
        pass
    
    def record_gesture_time(self, gesture_time: float):
        self.total_gestures += 1
    
    def record_input_time(self, input_time: float):
        pass
    
    def record_detection_time(self, detection_time: float):
        pass
    
    def record_classification_time(self, classification_time: float):
        pass
    
    def record_gesture_accuracy(self, accuracy: float):
        pass
    
    def record_latency_violation(self):
        pass
    
    def record_error(self):
        pass
    
    def record_warning(self):
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        current_fps = 0.0
        if len(self.frame_times) > 0:
            avg_time = np.mean(list(self.frame_times))
            current_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        uptime = time.time() - self.start_time
        return {
            'fps': {'current': current_fps},
            'counters': {
                'total_frames': self.total_frames,
                'total_gestures': self.total_gestures,
                'uptime_seconds': uptime
            }
        }
    
    def get_performance_alerts(self) -> List[str]:
        return []
    
    def cleanup(self):
        pass

class ConfigurationManager:

    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._create_default_config()
        self.load_configuration()
        
        # Basic validation
        self._basic_config_check()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Simplified configuration for development."""
        return {
            "camera": {
                "device_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            
            "mediapipe": {
                "model_complexity": 1,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.5,
                "max_num_hands": 2
            },
            
            "gestures": {
                "confidence_threshold": 0.8,
                "stability_frames": 3,
                "fist_threshold": 0.8,
                "open_hand_threshold": 0.6
            },
            
            "controls": {
                "accelerate": "w",
                "reverse": "s",
                "steer_left": "a", 
                "steer_right": "d",
                "brake": "space",
                "start_engine": "e"
            },
            
            "display": {
                "show_video": True,
                "show_landmarks": True,
                "show_fps": True,
                "show_gesture_info": True,
                "window_size": [1280, 720]
            }
        }
    
    def load_configuration(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._merge_configs(self.config, user_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    def save_configuration(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def _merge_configs(self, default: dict, user: dict):
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _basic_config_check(self):
        required_sections = ["camera", "mediapipe", "gestures", "controls", "display"]
        for section in required_sections:
            if section not in self.config:
                print(f"Warning: Missing config section '{section}', using defaults")
                self.config[section] = {}
    
    def get(self, path: str, default=None):
        try:
            keys = path.split('.')
            value = self.config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except:
            return default

class GameInputController:
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.controls_config = config.get("controls", {})
        # Initialize keyboard controller
        self.pynput_keyboard = None
        if PYNPUT_AVAILABLE:
            self.pynput_keyboard = pynput.keyboard.Controller()
        
        # Simplified key mappings
        self.key_mappings = {
            GameAction.ACCELERATE: self.controls_config.get("accelerate", "w"),
            GameAction.REVERSE: self.controls_config.get("reverse", "s"),
            GameAction.STEER_LEFT: self.controls_config.get("steer_left", "a"),
            GameAction.STEER_RIGHT: self.controls_config.get("steer_right", "d"),
            GameAction.BRAKE: self.controls_config.get("brake", "space"),
            GameAction.START_ENGINE: self.controls_config.get("start_engine", "e")
        }
        
    def execute_gesture_action(self, gesture: GestureType) -> List[GameAction]:
        executed_actions = []
        
        try:
            # Map gestures to actions
            if gesture == GestureType.ACCELERATE:
                if self._execute_single_action(GameAction.ACCELERATE):
                    executed_actions.append(GameAction.ACCELERATE)
                    
            elif gesture == GestureType.REVERSE:
                if self._execute_single_action(GameAction.REVERSE):
                    executed_actions.append(GameAction.REVERSE)
                    
            elif gesture == GestureType.E_BRAKE:
                if self._execute_single_action(GameAction.BRAKE):
                    executed_actions.append(GameAction.BRAKE)
                    
            elif gesture == GestureType.START_IDLE:
                if self._execute_single_action(GameAction.START_ENGINE):
                    executed_actions.append(GameAction.START_ENGINE)
                    
            elif gesture == GestureType.STEER_LEFT:
                if self._execute_single_action(GameAction.STEER_LEFT):
                    executed_actions.append(GameAction.STEER_LEFT)
                    
            elif gesture == GestureType.STEER_RIGHT:
                if self._execute_single_action(GameAction.STEER_RIGHT):
                    executed_actions.append(GameAction.STEER_RIGHT)
                    
            # Note: Removed auto-brake for simplicity in development mode
        
        except Exception as e:
            print(f"Error executing gesture action: {e}")
        
        return executed_actions
    
    def _execute_single_action(self, action: GameAction) -> bool:
        """Simplified action execution."""
        try:
            key = self.key_mappings.get(action)
            if not key:
                return False
            
            # Try pynput first, then pyautogui as fallback
            if self.pynput_keyboard and PYNPUT_AVAILABLE:
                return self._execute_pynput_action(key)
            elif PYAUTOGUI_AVAILABLE:
                return self._execute_pyautogui_action(key)
            else:
                print(f"No input method available for {action}")
                return False
                
        except Exception as e:
            print(f"Error executing {action}: {e}")
            return False
    
    def _execute_pynput_action(self, key: str) -> bool:
        try:
            if key == "space":
                self.pynput_keyboard.press(Key.space)
                time.sleep(0.01)
                self.pynput_keyboard.release(Key.space)
            else:
                self.pynput_keyboard.press(key)
                time.sleep(0.01)
                self.pynput_keyboard.release(key)
            return True
        except Exception as e:
            print(f"PyInput error: {e}")
            return False
    
    def _execute_pyautogui_action(self, key: str) -> bool:
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            print(f"PyAutoGUI error: {e}")
            return False

class HandGestureRecognizer:
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        mediapipe_config = config.get("mediapipe", {})
        gesture_config = config.get("gestures", {})
        
        self.confidence_threshold = gesture_config.get("confidence_threshold", 0.85)
        self.stability_frames = gesture_config.get("stability_frames", 3)
        self.fist_threshold = gesture_config.get("fist_threshold", 0.8)
        self.open_hand_threshold = gesture_config.get("open_hand_threshold", 0.6)
        self.hand_alignment_tolerance = gesture_config.get("hand_alignment_tolerance", 0.12)
        self.vertical_separation_threshold = gesture_config.get("vertical_separation_threshold", 0.08)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        hands_params = {
            'static_image_mode': mediapipe_config.get('static_image_mode', False),
            'max_num_hands': mediapipe_config.get('max_num_hands', 2),
            'min_detection_confidence': mediapipe_config.get('min_detection_confidence', 0.7),
            'min_tracking_confidence': mediapipe_config.get('min_tracking_confidence', 0.5)
        }
        
        try:
            self.hands = self.mp_hands.Hands(**hands_params)
        except Exception as e:
            print(f"Error initializing MediaPipe Hands: {e}")
            self.hands = self.mp_hands.Hands()
        
        # Gesture history for stability
        self.gesture_history = deque(maxlen=self.stability_frames)
        self.palm_center_history = {'left': deque(maxlen=10), 'right': deque(maxlen=10)}
        
        # Performance tracking
        self.total_detections = 0
        self.successful_detections = 0
        self.gesture_classification_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandLandmarkData], np.ndarray]:
        start_time = time.time()
        self.total_detections += 1
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Always create output frame for GUI display
            output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            detected_hands = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    try:
                        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        hand_data = HandLandmarkData(
                            landmarks=landmarks,
                            handedness=handedness.classification[0].label,
                            confidence=handedness.classification[0].score
                        )
                        detected_hands.append(hand_data)
                        
                        # Draw landmarks for visual feedback
                        if self.config.get("display.show_landmarks", True):
                            self.mp_draw.draw_landmarks(output_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                    except Exception as e:
                        print(f"Error processing hand landmarks: {e}")
                        continue
            
            self.successful_detections += 1
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            detected_hands = []
            output_frame = frame.copy()
        
        processing_time = time.time() - start_time
        self.detection_times.append(processing_time)
        
        return detected_hands, output_frame
    
    def classify_gesture(self, hands: List[HandLandmarkData]) -> Tuple[GestureType, float]:
        classification_start = time.time()
        
        try:
            if not hands:
                return GestureType.NO_GESTURE, 0.0
            
            if len(hands) == 1:
                return self._classify_single_hand_gesture(hands[0])
            
            gesture, confidence = self._classify_two_hand_gesture(hands)
            
            # Apply temporal smoothing for stability
            gesture = self._apply_temporal_smoothing(gesture, confidence)
            
        except Exception as e:
            print(f"Error in gesture classification: {e}")
            gesture, confidence = GestureType.UNKNOWN, 0.0
        
        processing_time = time.time() - classification_start
        self.gesture_classification_times.append(processing_time)
        
        return gesture, confidence
    
    def _classify_single_hand_gesture(self, hand: HandLandmarkData) -> Tuple[GestureType, float]:
        # For single hand, default to no gesture (safety)
        return GestureType.NO_GESTURE, 0.5
    
    def _classify_two_hand_gesture(self, hands: List[HandLandmarkData]) -> Tuple[GestureType, float]:
        if len(hands) != 2:
            return GestureType.INVALID, 0.0
        
        # Identify left and right hands
        left_hand = None
        right_hand = None
        
        for hand in hands:
            if hand.handedness.lower() == 'left':
                left_hand = hand
            elif hand.handedness.lower() == 'right':
                right_hand = hand
        
        if not (left_hand and right_hand):
            return GestureType.UNKNOWN, 0.5
        
        # Check hand alignment
        left_y = left_hand.palm_center[1] 
        right_y = right_hand.palm_center[1]
        vertical_diff = abs(left_y - right_y)
        
        # Check hand states
        left_is_fist = left_hand.is_fist(self.fist_threshold)
        right_is_fist = right_hand.is_fist(self.fist_threshold)
        left_is_open = left_hand.is_open_hand(self.open_hand_threshold)
        right_is_open = right_hand.is_open_hand(self.open_hand_threshold)
        
        base_confidence = min(left_hand.confidence, right_hand.confidence)
        
        # Classify based on hand combinations
        if left_is_open and right_is_open:
            if vertical_diff <= self.hand_alignment_tolerance:
                return GestureType.E_BRAKE, base_confidence
        
        elif left_is_fist and right_is_fist:
            if vertical_diff <= self.hand_alignment_tolerance:
                return GestureType.ACCELERATE, base_confidence
        
        elif left_is_fist and right_is_open:
            if vertical_diff <= self.hand_alignment_tolerance:
                return GestureType.REVERSE, base_confidence
        
        elif left_is_open and right_is_fist:
            if vertical_diff <= self.hand_alignment_tolerance:
                return GestureType.START_IDLE, base_confidence
        
        # Check for steering gestures
        if vertical_diff > self.vertical_separation_threshold:
            if left_y < right_y:  # Left hand higher
                return GestureType.STEER_LEFT, base_confidence * 0.8
            else:  # Right hand higher
                return GestureType.STEER_RIGHT, base_confidence * 0.8
        
        return GestureType.UNKNOWN, base_confidence * 0.5
    
    def _apply_temporal_smoothing(self, gesture: GestureType, confidence: float) -> GestureType:
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) < self.stability_frames:
            return gesture
        
        # Count occurrences in history
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Return most frequent gesture if it appears enough times
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        if most_common[1] >= max(2, self.stability_frames // 2):
            return most_common[0]
        
        return gesture

class CameraManager:
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        camera_config = config.get("camera", {})
        
        # Camera settings
        self.device_id = camera_config.get('device_id', 0)
        self.width = camera_config.get('width', 1280)
        self.height = camera_config.get('height', 720)
        self.fps = camera_config.get('fps', 30)
        self.buffer_size = camera_config.get('buffer_size', 1)
        self.backend = camera_config.get('backend', 'auto')
        self.fourcc = camera_config.get('fourcc', 'MJPG')
        
        # Camera state
        self.camera = None
        self.is_initialized = False
        self.frame_count = 0
        self.last_frame = None
    
    def detect_available_cameras(self) -> List[int]:
        """Simple camera detection for development."""
        available_cameras = []
        
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                        print(f"Camera {i}: Available")
                    cap.release()
            except:
                continue
        
        return available_cameras if available_cameras else [0]
    
    def initialize_camera(self) -> bool:
        try:
            print(f"Initializing camera (device {self.device_id})...")
            
            # Simple camera initialization
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                print(f"Failed to open camera device {self.device_id}")
                return False
            
            # Basic camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test frame capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("Failed to capture test frame")
                return False
            
            self.is_initialized = True
            self._print_camera_info()
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def _print_camera_info(self):
        try:
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera initialized: {actual_width}x{actual_height} @ device {self.device_id}")
        except:
            print("Camera initialized successfully")
    
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Simple frame capture for development."""
        if not self.is_initialized or not self.camera:
            return False, None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self.frame_count += 1
                return True, frame
            else:
                return False, None
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return False, None
    
    def get_camera_statistics(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'frame_count': self.frame_count,
            'is_initialized': self.is_initialized
        }
    
    def cleanup(self):
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
            self.is_initialized = False
            print("Camera resources released")
        except Exception as e:
            print(f"Error during camera cleanup: {e}")

class GUIDisplayManager:
    """GUI display manager for video monitoring and gesture feedback."""
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.display_config = config.get("display", {})
        
        # Display settings
        self.show_video = self.display_config.get("show_video", True)
        self.show_fps = self.display_config.get("show_fps", True)
        self.show_gesture_info = self.display_config.get("show_gesture_info", True)
        self.window_width = self.display_config.get("window_size", [1280, 720])[0]
        self.window_height = self.display_config.get("window_size", [1280, 720])[1]
        
        # Status tracking
        self.current_gesture = GestureType.NO_GESTURE
        self.current_confidence = 0.0
        self.current_fps = 0.0
        self.frame_count = 0
        self.last_gesture_actions = []
        
        # Colors for different gesture types
        self.gesture_colors = {
            GestureType.E_BRAKE: (0, 0, 255),          # Red
            GestureType.ACCELERATE: (0, 255, 0),       # Green  
            GestureType.REVERSE: (255, 255, 0),        # Cyan
            GestureType.START_IDLE: (255, 0, 255),     # Magenta
            GestureType.STEER_LEFT: (0, 165, 255),     # Orange
            GestureType.STEER_RIGHT: (0, 165, 255),    # Orange
            GestureType.NO_GESTURE: (128, 128, 128),   # Gray
            GestureType.UNKNOWN: (255, 255, 255),      # White
            GestureType.INVALID: (0, 0, 128)           # Dark Red
        }
        
        # Initialize window if video display is enabled
        if self.show_video:
            try:
                # Use WINDOW_NORMAL which is more widely supported
                window_flag = getattr(cv2, 'WINDOW_NORMAL', 1)  # 1 is the value for WINDOW_NORMAL
                cv2.namedWindow("Gesture Control Monitor", window_flag)
                
                # Try to resize window
                cv2.resizeWindow("Gesture Control Monitor", self.window_width, self.window_height)
            except Exception as e:
                print(f"Note: Window setup issue (non-critical): {e}")
                # Create a simple window as fallback
                cv2.namedWindow("Gesture Control Monitor")
    
    def update_status(self, gesture: GestureType, confidence: float, fps: float, actions: List = None):
        """Update current status information."""
        self.current_gesture = gesture
        self.current_confidence = confidence
        self.current_fps = fps
        if actions:
            self.last_gesture_actions = actions
    
    def draw_overlay_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay information on the frame."""
        if not (self.show_fps or self.show_gesture_info):
            return frame
        
        overlay_frame = frame.copy()
        h, w = overlay_frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_frame, 0.3, 0, overlay_frame)
        
        y_offset = 30
        
        # Show FPS
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(overlay_frame, fps_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25
        
        # Show gesture information
        if self.show_gesture_info:
            gesture_color = self.gesture_colors.get(self.current_gesture, (255, 255, 255))
            
            # Current gesture
            gesture_text = f"Gesture: {self.current_gesture.value.upper()}"
            cv2.putText(overlay_frame, gesture_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
            y_offset += 25
            
            # Confidence
            conf_text = f"Confidence: {self.current_confidence:.2f}"
            cv2.putText(overlay_frame, conf_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            # Last actions
            if self.last_gesture_actions:
                actions_text = f"Actions: {', '.join([a.value for a in self.last_gesture_actions])}"
                cv2.putText(overlay_frame, actions_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw gesture instructions in the top right
        self._draw_gesture_guide(overlay_frame)
        
        return overlay_frame
    
    def _draw_gesture_guide(self, frame: np.ndarray):
        """Draw gesture guide on the frame."""
        h, w = frame.shape[:2]
        guide_x = w - 350
        guide_y = 30
        
        # Background for guide
        overlay = frame.copy()
        cv2.rectangle(overlay, (guide_x - 10, guide_y - 20), (w - 10, guide_y + 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Guide title
        cv2.putText(frame, "Gesture Guide:", (guide_x, guide_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gesture instructions
        instructions = [
            "Both Open = BRAKE (Space)",
            "Both Fists = ACCELERATE (W)",  
            "R-Open L-Fist = REVERSE (S)",
            "R-Fist L-Open = START (E)",
            "Height Diff = STEERING (A/D)"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = guide_y + 25 + (i * 20)
            cv2.putText(frame, instruction, (guide_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def display_frame(self, frame: np.ndarray) -> bool:
        """Display frame with overlay information."""
        if not self.show_video:
            return True
        
        try:
            # Add overlay information
            display_frame = self.draw_overlay_info(frame)
            
            # Show the frame
            cv2.imshow("Gesture Control Monitor", display_frame)
            
            # Handle window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                return False
            elif key == ord('f'):  # Toggle FPS display
                self.show_fps = not self.show_fps
                print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
            elif key == ord('g'):  # Toggle gesture info
                self.show_gesture_info = not self.show_gesture_info
                print(f"Gesture info: {'ON' if self.show_gesture_info else 'OFF'}")
                
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"GUI display error: {e}")
            # Try to continue without GUI
            self.show_video = False
            print("Continuing without GUI display...")
            return True
    
    def cleanup(self):
        """Cleanup display resources."""
        try:
            cv2.destroyAllWindows()
        except:
            pass

class SimpleLoggingManager:
    """Simplified logging for development."""
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.logger = logging.getLogger("GestureControl")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_system_information(self):
        self.logger.info("=== Gesture Control System Started ===")
    
    def log_gesture_detection(self, gesture, confidence, processing_time, actions, frame_number=0):
        if gesture != GestureType.NO_GESTURE:
            self.logger.info(f"Gesture: {gesture.value}, Confidence: {confidence:.2f}")
    
    def log_performance(self, fps, latency_ms=0):
        pass
    
    def log_error(self, error_msg, exception=None):
        if exception:
            self.logger.error(f"{error_msg}: {str(exception)}")
        else:
            self.logger.error(error_msg)
    
    def log_warning(self, warning_msg):
        self.logger.warning(warning_msg)
    
    def log_info(self, message):
        self.logger.info(message)
    
    def cleanup(self):
        self.logger.info("=== Gesture Control System Stopped ===")

class GGControlSystem:
    
    def __init__(self):
        self.system_start_time = time.time()
        
        # Initialize configuration system first
        self.config = ConfigurationManager()
        
        # Initialize logging system (simplified)
        self.logger = SimpleLoggingManager(self.config)
        self.logger.log_system_information()
        
        # Initialize performance monitoring (simplified)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize game input controller
        self.game_controller = GameInputController(self.config)
        
        # Initialize camera manager  
        self.camera_manager = CameraManager(self.config)
        
        # Initialize gesture recognizer
        self.gesture_recognizer = HandGestureRecognizer(self.config)
        
        # Initialize GUI display manager
        try:
            self.gui_display = GUIDisplayManager(self.config)
        except Exception as e:
            print(f"Warning: GUI initialization failed: {e}")
            print("Running in console-only mode...")
            # Create a minimal display manager that doesn't show GUI
            self.gui_display = self._create_minimal_display_manager()
        
        # System state
        self.is_running = False
        self.gesture_queue = queue.Queue()
        self.current_gesture = GestureType.NO_GESTURE
        self.frame_count = 0
        self.successful_inputs = 0
        self.failed_inputs = 0
        
        # FPS calculation
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        self.logger.log_info("GGControlSystem initialized successfully")
    
    def _create_minimal_display_manager(self):
        """Create a minimal display manager for console-only mode."""
        class MinimalDisplayManager:
            def __init__(self):
                self.show_video = False
                
            def update_status(self, gesture, confidence, fps, actions=None):
                pass
                
            def display_frame(self, frame):
                return True
                
            def cleanup(self):
                pass
        
        return MinimalDisplayManager()
    
    def start_system(self):
        """Start the gesture control system (simplified for development)."""
        print("Starting simplified gesture control system...")
        
        if not self.camera_manager.initialize_camera():
            print("Failed to initialize camera")
            return False
        
        self.is_running = True
        print("System started successfully")
        return True
    
    def run(self):
        """Main execution loop with GUI display."""
        if not self.start_system():
            return
        
        print("Gesture control running...")
        print("GUI Controls:")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 'f' to toggle FPS display")
        print("  - Press 'g' to toggle gesture info")
        
        try:
            while self.is_running:
                frame_start = time.time()
                
                # Capture frame
                ret, frame = self.camera_manager.capture_frame()
                if ret and frame is not None:
                    # Process frame for gestures and get annotated frame
                    hands, display_frame = self.gesture_recognizer.process_frame(frame)
                    gesture, confidence = self.gesture_recognizer.classify_gesture(hands)
                    
                    # Execute gesture actions
                    actions = []
                    if gesture != GestureType.NO_GESTURE:
                        actions = self.game_controller.execute_gesture_action(gesture)
                        if actions:
                            self.successful_inputs += 1
                        else:
                            self.failed_inputs += 1
                    
                    # Update current gesture state
                    self.current_gesture = gesture
                    
                    # Calculate FPS
                    current_fps = self._calculate_fps(frame_start)
                    
                    # Update GUI display with current status
                    self.gui_display.update_status(gesture, confidence, current_fps, actions)
                    
                    # Display frame with GUI overlays
                    if not self.gui_display.display_frame(display_frame):
                        # User pressed quit in GUI
                        break
                        
                    self.frame_count += 1
                else:
                    # No frame captured, brief pause
                    time.sleep(0.01)
                
                # Log gesture detection for non-idle gestures
                if gesture not in [GestureType.NO_GESTURE, GestureType.UNKNOWN]:
                    self.logger.log_gesture_detection(gesture, confidence, 0, actions, self.frame_count)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def _calculate_fps(self, frame_time: float) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        frame_duration = current_time - frame_time
        
        if frame_duration > 0:
            self.fps_counter.append(1.0 / frame_duration)
        
        # Update FPS every second
        if current_time - self.last_fps_time >= 1.0:
            self.performance_monitor.record_frame_time(frame_duration)
            self.last_fps_time = current_time
        
        # Return average FPS from recent frames
        if len(self.fps_counter) > 0:
            return sum(self.fps_counter) / len(self.fps_counter)
        return 0.0
    
    def cleanup(self):
        """Cleanup system resources."""
        self.is_running = False
        try:
            self.gui_display.cleanup()
            self.camera_manager.cleanup()
            self.logger.cleanup()
        except:
            pass
        print("System shutdown complete")

# Main execution
def main():
    """Main function for gesture control system."""
    try:
        system = GGControlSystem()
        system.run()
    except Exception as e:
        print(f"System error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())