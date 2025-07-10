# Fixed Player Tracking - Prevents High Player IDs
import cv2
import numpy as np
from ultralytics import YOLO
import os
import math

class RobustPlayerTracker:
    def __init__(self, max_distance=150, max_frames_lost=30, min_confidence=0.4, max_players=25):
        self.players = {}  # Active players: {id: {bbox, last_seen, frames_lost, history}}
        self.inactive_players = {}  # Recently lost players that can be reactivated
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.min_confidence = min_confidence
        self.max_players = max_players
        self.next_available_id = 1
        
        # Colors for each player ID
        self.colors = [
            (0, 255, 0),    # Green - Player 1
            (255, 0, 0),    # Blue - Player 2  
            (0, 0, 255),    # Red - Player 3
            (255, 255, 0),  # Cyan - Player 4
            (255, 0, 255),  # Magenta - Player 5
            (0, 255, 255),  # Yellow - Player 6
            (128, 0, 128),  # Purple - Player 7
            (255, 165, 0),  # Orange - Player 8
            (0, 128, 128),  # Teal - Player 9
            (128, 128, 0),  # Olive - Player 10
            (255, 192, 203), # Pink - Player 11
            (0, 128, 0),    # Dark Green - Player 12
            (128, 0, 0),    # Dark Red - Player 13
            (0, 0, 128),    # Navy - Player 14
            (128, 128, 128), # Gray - Player 15
            (255, 20, 147), # Deep Pink - Player 16
            (0, 191, 255),  # Deep Sky Blue - Player 17
            (50, 205, 50),  # Lime Green - Player 18
            (255, 140, 0),  # Dark Orange - Player 19
            (148, 0, 211),  # Dark Violet - Player 20
            (220, 20, 60),  # Crimson - Player 21
            (0, 100, 0),    # Dark Green - Player 22
            (139, 69, 19),  # Saddle Brown - Player 23
            (255, 105, 180), # Hot Pink - Player 24
            (70, 130, 180)  # Steel Blue - Player 25
        ]
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate center distance between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def get_similarity_score(self, bbox1, bbox2):
        """Calculate similarity score between two bounding boxes"""
        distance = self.calculate_distance(bbox1, bbox2)
        iou = self.calculate_iou(bbox1, bbox2)
        
        # Normalize distance (smaller is better)
        distance_score = max(0, 1 - (distance / self.max_distance))
        
        # Combined score (higher is better)
        return (distance_score * 0.6) + (iou * 0.4)
    
    def find_best_match(self, detection_bbox, player_pool):
        """Find the best matching player for a detection"""
        best_match = None
        best_score = 0.3  # Minimum threshold for matching
        
        for player_id, player_data in player_pool.items():
            score = self.get_similarity_score(detection_bbox, player_data['bbox'])
            
            if score > best_score:
                best_score = score
                best_match = player_id
        
        return best_match, best_score
    
    def get_next_id(self):
        """Get next available ID, reusing low numbers when possible"""
        # First, try to find an unused low number
        for i in range(1, self.max_players + 1):
            if i not in self.players and i not in self.inactive_players:
                return i
        
        # If all low numbers are used, find the next available
        while (self.next_available_id in self.players or 
               self.next_available_id in self.inactive_players):
            self.next_available_id += 1
        
        return min(self.next_available_id, self.max_players)
    
    def update_tracking(self, detections):
        """Update tracking with new detections"""
        if not detections:
            # No detections - increment frames_lost for all active players
            for player_id in list(self.players.keys()):
                self.players[player_id]['frames_lost'] += 1
                if self.players[player_id]['frames_lost'] > self.max_frames_lost:
                    # Move to inactive players
                    self.inactive_players[player_id] = self.players[player_id]
                    del self.players[player_id]
            return []
        
        # Filter valid detections
        valid_detections = []
        for detection in detections:
            if len(detection) >= 5 and detection[4] >= self.min_confidence:
                x, y, w, h = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                conf = float(detection[4])
                valid_detections.append((x, y, w, h, conf))
        
        if not valid_detections:
            return []
        
        # Track assignments
        used_detections = set()
        used_players = set()
        tracking_results = []
        
        # Step 1: Match detections to active players
        for det_idx, (x, y, w, h, conf) in enumerate(valid_detections):
            if det_idx in used_detections:
                continue
                
            detection_bbox = (x, y, w, h)
            best_match, score = self.find_best_match(detection_bbox, self.players)
            
            if best_match and best_match not in used_players:
                # Update existing active player
                self.players[best_match]['bbox'] = detection_bbox
                self.players[best_match]['frames_lost'] = 0
                self.players[best_match]['confidence'] = conf
                self.players[best_match]['last_seen'] = 0
                
                tracking_results.append({
                    'id': best_match,
                    'bbox': detection_bbox,
                    'confidence': conf
                })
                
                used_detections.add(det_idx)
                used_players.add(best_match)
        
        # Step 2: Try to reactivate inactive players for remaining detections
        for det_idx, (x, y, w, h, conf) in enumerate(valid_detections):
            if det_idx in used_detections:
                continue
                
            detection_bbox = (x, y, w, h)
            best_match, score = self.find_best_match(detection_bbox, self.inactive_players)
            
            if best_match and score > 0.4:  # Higher threshold for reactivation
                # Reactivate player
                self.players[best_match] = self.inactive_players[best_match]
                self.players[best_match]['bbox'] = detection_bbox
                self.players[best_match]['frames_lost'] = 0
                self.players[best_match]['confidence'] = conf
                self.players[best_match]['last_seen'] = 0
                
                del self.inactive_players[best_match]
                
                tracking_results.append({
                    'id': best_match,
                    'bbox': detection_bbox,
                    'confidence': conf
                })
                
                used_detections.add(det_idx)
        
        # Step 3: Create new players for remaining detections (limit to max_players)
        current_total = len(self.players) + len(self.inactive_players)
        
        for det_idx, (x, y, w, h, conf) in enumerate(valid_detections):
            if det_idx in used_detections:
                continue
                
            if current_total >= self.max_players:
                print(f"Warning: Maximum player limit ({self.max_players}) reached. Ignoring new detection.")
                continue
            
            detection_bbox = (x, y, w, h)
            new_id = self.get_next_id()
            
            self.players[new_id] = {
                'bbox': detection_bbox,
                'frames_lost': 0,
                'confidence': conf,
                'last_seen': 0,
                'history': [detection_bbox]
            }
            
            tracking_results.append({
                'id': new_id,
                'bbox': detection_bbox,
                'confidence': conf
            })
            
            current_total += 1
        
        # Step 4: Update frames_lost for unmatched active players
        for player_id in list(self.players.keys()):
            if player_id not in used_players:
                self.players[player_id]['frames_lost'] += 1
                if self.players[player_id]['frames_lost'] > self.max_frames_lost:
                    # Move to inactive
                    self.inactive_players[player_id] = self.players[player_id]
                    del self.players[player_id]
        
        # Step 5: Clean up very old inactive players
        for player_id in list(self.inactive_players.keys()):
            self.inactive_players[player_id]['frames_lost'] += 1
            if self.inactive_players[player_id]['frames_lost'] > self.max_frames_lost * 2:
                del self.inactive_players[player_id]
        
        return tracking_results
    
    def get_color(self, player_id):
        """Get color for a specific player ID"""
        if player_id <= len(self.colors):
            return self.colors[player_id - 1]
        else:
            # Fallback color
            return (255, 255, 255)
    
    def get_stats(self):
        """Get tracker statistics"""
        return {
            'active_players': len(self.players),
            'inactive_players': len(self.inactive_players),
            'total_tracked': len(self.players) + len(self.inactive_players)
        }

def robust_video_display():
    """Robust video display with limited player IDs"""
    print("Loading YOLO model...")
    
    try:
        model = YOLO("models/yolov11_model.pt")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    video_path = "data/15sec_input_720p.mp4"
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Error opening video file")
        return
    
    print("✓ Video loaded successfully")
    
    # Initialize tracker with reasonable limits
    tracker = RobustPlayerTracker(
        max_distance=150,     # Increased for better matching
        max_frames_lost=30,   # More tolerance for temporary occlusions
        min_confidence=0.4,   # Lower threshold for better detection
        max_players=25        # Maximum player limit
    )
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Create window
    cv2.namedWindow('Robust Player Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Robust Player Tracking', 1200, 800)
    
    frame_count = 0
    paused = False
    
    print("\nControls:")
    print("  SPACE = Pause/Resume")
    print("  Q = Quit")
    print("  R = Reset tracker")
    print("  S = Save current frame")
    print("\nStarting video playback...")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Extract detections
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # Convert to (x, y, w, h) format
                        x, y, w, h = x1, y1, x2-x1, y2-y1
                        detections.append([x, y, w, h, conf, cls])
            
            # Update tracking
            tracked_players = tracker.update_tracking(detections)
            
            # Draw tracking results
            for player in tracked_players:
                player_id = player['id']
                x, y, w, h = player['bbox']
                confidence = player['confidence']
                color = tracker.get_color(player_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 3)
                
                # Create label
                label = f"Player {player_id}"
                conf_text = f"{confidence:.2f}"
                
                # Calculate text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.6, thickness-1)
                
                # Draw label background
                bg_width = max(label_w, conf_w) + 20
                bg_height = label_h + conf_h + 20
                
                cv2.rectangle(frame, (int(x), int(y - bg_height)), 
                             (int(x + bg_width), int(y)), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (int(x + 10), int(y - conf_h - 10)), 
                           font, font_scale, (255, 255, 255), thickness)
                cv2.putText(frame, conf_text, (int(x + 10), int(y - 5)), 
                           font, 0.6, (255, 255, 255), thickness-1)
            
            # Get tracker stats
            stats = tracker.get_stats()
            
            # Draw enhanced info panel
            info_height = 140
            cv2.rectangle(frame, (10, 10), (450, 10 + info_height), (0, 0, 0), -1)
            
            info_y = 35
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, info_y), 
                       font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Active Players: {stats['active_players']}", (20, info_y + 30), 
                       font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inactive Players: {stats['inactive_players']}", (20, info_y + 60), 
                       font, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Total Tracked: {stats['total_tracked']}", (20, info_y + 90), 
                       font, 0.7, (0, 255, 255), 2)
            
            if paused:
                cv2.putText(frame, "PAUSED", (20, info_y + 120), 
                           font, 0.7, (0, 0, 255), 2)
            
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Active: {stats['active_players']}, "
                      f"Inactive: {stats['inactive_players']}, Total: {stats['total_tracked']}")
        
        # Display frame
        cv2.imshow('Robust Player Tracking', frame)
        
        # Handle key presses
        key = cv2.waitKey(1 if paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            tracker = RobustPlayerTracker(
                max_distance=150, max_frames_lost=30, min_confidence=0.4, max_players=25
            )
            print("Tracker reset")
        elif key == ord('s'):
            filename = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    final_stats = tracker.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Active Players: {final_stats['active_players']}")
    print(f"  Inactive Players: {final_stats['inactive_players']}")
    print(f"  Total Players Tracked: {final_stats['total_tracked']}")
    print("Video playback finished")

if __name__ == "__main__":
    print("=== Robust Player Tracking System ===")
    print("Features:")
    print("  ✓ Player IDs stay between 1-25")
    print("  ✓ IDs are reused when players return")
    print("  ✓ No duplicate IDs")
    print("  ✓ Persistent tracking across frames")
    print("  ✓ Handles temporary occlusions")
    print()
    
    robust_video_display()