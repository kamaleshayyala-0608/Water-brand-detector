import time

def get_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x1_i >= x2_i or y1_i >= y2_i:
        return 0.0
        
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

class TemporalBoxTracker:
    """Tracks boxes across frames. Keeps lost tracks for up to max_lost_frames to handle temporary occlusions/missed detections."""
    def __init__(self, iou_threshold=0.40, max_lost_frames=15):
        self.trackers = []
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 0
        
    def update(self, current_detections, current_frame_idx=0):
        # Match current detections with existing trackers (both active and lost)
        # Sort existing trackers: active first (lost_frames == 0) then lost trackers, to prioritize active ones
        sorted_trackers = sorted(self.trackers, key=lambda x: x.get('lost_frames', 0))
        
        updated_trackers = []
        used_detections = set()
        
        # Match
        for tr in sorted_trackers:
            best_match_idx = -1
            best_iou = self.iou_threshold
            
            for idx, det in enumerate(current_detections):
                if idx in used_detections:
                    continue
                # Both should be class 0 (bottle), but let's check class matching just in case
                if det.get('cls', 0) != tr.get('cls', 0):
                    continue
                
                iou = get_iou(det['box'], tr['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            if best_match_idx != -1:
                det = current_detections[best_match_idx]
                used_detections.add(best_match_idx)
                updated_trackers.append({
                    'id': tr['id'],
                    'box': det['box'],
                    'cls': det.get('cls', 0),
                    'conf': det['conf'],
                    'frames': tr['frames'] + 1,
                    'counted': tr.get('counted', False),
                    'start_time': tr.get('start_time', time.time()),
                    'start_frame': tr.get('start_frame', current_frame_idx),
                    'lost_frames': 0  # reset lost frames
                })
            else:
                # No match found for this tracker in the current frame
                lost_cnt = tr.get('lost_frames', 0) + 1
                if lost_cnt <= self.max_lost_frames:
                    # Keep the tracker but mark it as lost, retaining its previous box
                    updated_trackers.append({
                        'id': tr['id'],
                        'box': tr['box'],
                        'cls': tr.get('cls', 0),
                        'conf': tr['conf'],
                        'frames': tr['frames'],  # don't increment consecutive frames count
                        'counted': tr.get('counted', False),
                        'start_time': tr.get('start_time', time.time()),
                        'start_frame': tr.get('start_frame', current_frame_idx),
                        'lost_frames': lost_cnt
                    })
        
        # Unmatched detections are registered as new trackers (frames=1)
        for idx, det in enumerate(current_detections):
            if idx not in used_detections:
                updated_trackers.append({
                    'id': self.next_id,
                    'box': det['box'],
                    'cls': det.get('cls', 0),
                    'conf': det['conf'],
                    'frames': 1,
                    'counted': False,
                    'start_time': time.time(),
                    'start_frame': current_frame_idx,
                    'lost_frames': 0
                })
                self.next_id += 1
                
        self.trackers = updated_trackers
        # Only return active (not lost) trackers that have persisted for >= 3 frames
        return [tr for tr in self.trackers if tr.get('lost_frames', 0) == 0 and tr['frames'] >= 3]
