import os
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Add parent directory to path to enable relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))

from utils.ocr import detect_brand
from utils.water_level import estimate_water_level
from utils.tracker import TemporalBoxTracker

# Brand Details Database for Sidebar HUD and cards
KNOWN_BRANDS = {
    "bisleri": {
        "brand": "Bisleri",
        "category": "Packaged Water",
        "volume": "1 Litre",
        "manufacturer": "Bisleri International",
        "details": "Mineral enriched water"
    },
    "kinley": {
        "brand": "Kinley",
        "category": "Packaged Water",
        "volume": "1 Litre",
        "manufacturer": "Coca-Cola Company",
        "details": "Purified water with minerals"
    },
    "aquafina": {
        "brand": "Aquafina",
        "category": "Packaged Water",
        "volume": "1 Litre",
        "manufacturer": "PepsiCo",
        "details": "Purified drinking water"
    },
    "bailley": {
        "brand": "Bailley",
        "category": "Packaged Water",
        "volume": "1 Litre",
        "manufacturer": "Parle Agro",
        "details": "Purified drinking water"
    },
    "himalayan": {
        "brand": "Himalayan",
        "category": "Natural Mineral Water",
        "volume": "750 mL",
        "manufacturer": "Tata Consumer Products",
        "details": "Source: Himalayan aquifer"
    },
    "rail neer": {
        "brand": "Rail Neer",
        "category": "Packaged Water",
        "volume": "1 Litre",
        "manufacturer": "IRCTC",
        "details": "Indian Railways drinking water"
    }
}

def draw_hud(frame, fps, elapsed_time, brand_counts, total_session_detections, recent_detections, is_recording, show_fps):
    """Draws a professional, clean sidebar dashboard overlay on the frame"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (240, 630), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (240, 630), (100, 100, 100), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Section 1: System Info
    cv2.putText(frame, "SYSTEM STATUS", (25, 35), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, 43), (230, 43), (80, 80, 80), 1)
    
    y = 65
    cv2.putText(frame, "Model: YOLOv8 Bottle Det", (25, y), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y += 25
    
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    time_str = f"{mins:02d}:{secs:02d}"
    cv2.putText(frame, f"Session Time: {time_str}", (25, y), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    y += 25
    
    if show_fps:
        cv2.putText(frame, f"FPS: {int(fps)}", (25, y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 25
        
    # Section 2: Dynamic Brand Counts
    y += 10
    cv2.putText(frame, "BRAND COUNTS", (25, y), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, y + 7), (230, y + 7), (80, 80, 80), 1)
    y += 30
    
    displayed = 0
    # Prioritize popular brands in ordering, display others dynamically
    prio_keys = ["Bisleri", "Kinley", "Aquafina"]
    all_keys = list(brand_counts.keys())
    keys_sorted = sorted(all_keys, key=lambda k: prio_keys.index(k) if k in prio_keys else 999)
    
    for brand in keys_sorted:
        if displayed >= 4:
            break
        cnt = brand_counts[brand]
        brand_upper = brand.upper()
        
        if brand_upper == "BISLERI":
            color = (0, 255, 0)
        elif brand_upper == "KINLEY":
            color = (255, 180, 50)
        elif brand_upper == "AQUAFINA":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
            
        cv2.putText(frame, f"{brand[:10]}: {cnt}", (25, y), font, 0.55, color, 2, cv2.LINE_AA)
        y += 25
        displayed += 1
        
    y = y - 25 + 25
    cv2.line(frame, (20, y), (230, y), (80, 80, 80), 1)
    cv2.putText(frame, f"Total Dets: {total_session_detections}", (25, y + 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    y += 45
    
    # Section 3: Recent Detections
    cv2.putText(frame, "RECENT EVENTS", (25, y), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, y + 7), (230, y + 7), (80, 80, 80), 1)
    y += 30
    
    for idx, brand in enumerate(recent_detections):
        cv2.putText(frame, f"{idx+1}. {brand}", (25, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
        
    # Section 4: Controls
    y = 515
    cv2.putText(frame, "CONTROLS", (25, y), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, y + 7), (230, y + 7), (80, 80, 80), 1)
    y += 25
    cv2.putText(frame, "[Q] Quit   [S] Save Pic", (25, y), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "[R] Record [C] Reset", (25, y), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "[F] Toggle FPS Overlay", (25, y), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Recording blinker
    if is_recording:
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (35, 605), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC ACTIVE", (50, 610), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

def draw_detail_cards(frame, active_dets, ocr_cache):
    """Draws spatial, temporal and OCR metadata cards for tracked bottles on the RHS"""
    overlay = frame.copy()
    dets_to_draw = active_dets[:2]
    
    card_w = 230
    card_h = 230
    start_x = 640 - card_w - 10
    
    y = 50
    for det in dets_to_draw:
        cv2.rectangle(overlay, (start_x, y), (start_x + card_w, y + card_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (start_x, y), (start_x + card_w, y + card_h), (100, 100, 100), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"OBJECT DETAILS: #{det['id']}", (start_x + 15, y + 25), font, 0.45, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (start_x + 10, y + 32), (start_x + card_w - 10, y + 32), (80, 80, 80), 1)
        
        box = det['box']
        start_time_track = det.get('start_time', time.time())
        time_seen = time.time() - start_time_track
        
        # Position
        cx = (box[0] + box[2]) / 2.0
        pos_str = "Left" if cx < 213.3 else ("Right" if cx > 426.6 else "Center")
        
        # Status
        h, w = frame.shape[:2]
        is_border = (box[0] < 15 or box[2] > w - 15 or box[1] < 15 or box[3] > h - 15)
        status_str = "Border/Partial" if is_border else "Visible"
        
        # Cache retrieve
        cache_data = ocr_cache.get(det['id'], {
            "brand": "Processing...",
            "confidence": 1.0,
            "status": "Analyzing Label",
            "water_level": None,
            "metadata": {
                "brand": "Analyzing...",
                "category": "Analyzing...",
                "volume": "N/A",
                "manufacturer": "N/A",
                "details": "Running OCR engine..."
            }
        })
        
        brand_name = cache_data["brand"]
        status_label = cache_data["status"]
        water_level = cache_data.get("water_level")
        water_level_val = f"{water_level}%" if water_level is not None else "Analyzing..."
        meta = cache_data["metadata"]
        
        # Format rows
        row_y = y + 55
        details_rows = [
            ("Object Type", "Water Bottle"),
            ("Brand", brand_name),
            ("Water Level", water_level_val),
            ("Track ID", f"{det['id']}"),
            ("Position", pos_str),
            ("Status", status_str),
            ("Time Seen", f"{time_seen:.1f} sec"),
            ("Category", meta.get("category", "N/A")),
            ("Volume", meta.get("volume", "N/A"))
        ]
        
        for label, val in details_rows:
            cv2.putText(frame, f"{label}:", (start_x + 15, row_y), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, val, (start_x + 105, row_y), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            row_y += 18
            
        y += card_h + 15

def start_webcam_detection():
    # Resolve paths dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "best.pt"))
    coco_model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "yolov8n.pt"))
    screenshots_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "screenshots"))
    recordings_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "recordings"))
    logs_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "logs"))
    
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(recordings_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("=" * 55)
    print("      D Y N A M I C   B O T T L E   O C R   S Y S      ")
    print("=" * 55)
    
    # Load Model (prefer pre-trained COCO bottle detector for robust real-world detection)
    model_path = coco_model_path if os.path.exists(coco_model_path) else custom_model_path
    is_custom_model = (model_path == custom_model_path)
    
    if os.path.exists(model_path):
        print(f"YOLO Model Loaded: {os.path.basename(model_path)} (Custom detector: {is_custom_model})")
    else:
        print(f"Error: Model weights not found at {model_path}")
        return
        
    print("\nWebcam Connected")
    print("Starting Detection...")
    print("=" * 55)
    
    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    half_precision = (device == 0)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
        
    # State values
    brand_counts = {}
    total_session_detections = 0
    recent_detections = []
    
    ocr_cache = {}
    tracker = TemporalBoxTracker()
    active_dets = []
    
    prev_time = 0
    frame_count = 0
    start_time = time.time()
    
    is_recording = False
    video_writer = None
    show_fps = True
    
    toast_text = ""
    toast_timer = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_resized = cv2.resize(frame, (640, 640))
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time
        
        if frame_count % 2 == 0:
            results = model.predict(
                frame_resized,
                imgsz=640,
                conf=0.25,
                verbose=False,
                device=device,
                half=half_precision
            )
            
            # Filter detections depending on model type
            current_dets = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Custom model has class 0 as bottle. COCO model has class 39 as bottle.
                    if (is_custom_model and cls == 0) or (not is_custom_model and cls == 39):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        current_dets.append({'box': [x1, y1, x2, y2], 'cls': cls, 'conf': conf})
                        
            active_dets = tracker.update(current_dets)
            
        # Draw and run post-processing once per stabilized track
        new_events = []
        for det in active_dets:
            x1, y1, x2, y2 = det['box']
            track_id = det['id']
            
            # Post-process crop on stabilizer activation
            h_f, w_f = frame_resized.shape[:2]
            crop = frame_resized[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
            
            if track_id not in ocr_cache:
                brand_name = "UNKNOWN"
                water_level = None
                
                if crop.size > 0:
                    brand_name = detect_brand(crop)
                    water_level = estimate_water_level(crop)
                    
                meta_info = KNOWN_BRANDS.get(brand_name.lower(), {
                    "brand": brand_name.title(),
                    "category": "Generic Bottle",
                    "volume": "Unknown",
                    "manufacturer": "Unknown",
                    "details": "Post-processed dynamically identified bottle"
                })
                
                ocr_cache[track_id] = {
                    "brand": brand_name.title(),
                    "confidence": det['conf'],
                    "status": "Recognized" if brand_name != "UNKNOWN" else "Unknown Label",
                    "water_level": water_level,
                    "metadata": meta_info
                }
                
                # Update dynamic brand counts
                display_brand = brand_name.title()
                brand_counts[display_brand] = brand_counts.get(display_brand, 0) + 1
                total_session_detections += 1
                new_events.append(display_brand)
            else:
                # Update existing track's metadata dynamically
                cached = ocr_cache[track_id]
                
                # 1. Update water level estimation dynamically
                if crop.size > 0:
                    new_level = estimate_water_level(crop)
                    if new_level is not None:
                        cached["water_level"] = new_level
                
                # 2. Refine brand prediction if it is currently "Unknown"
                if cached["brand"].upper() == "UNKNOWN" and frame_count % 10 == 0:
                    if crop.size > 0:
                        brand_name = detect_brand(crop)
                        if brand_name != "UNKNOWN":
                            # Recognized a brand! Update the counts and cache
                            old_brand = cached["brand"]
                            new_brand = brand_name.title()
                            
                            # Decrement old count
                            if old_brand in brand_counts:
                                brand_counts[old_brand] -= 1
                                if brand_counts[old_brand] <= 0:
                                    brand_counts.pop(old_brand, None)
                            
                            # Increment new brand count
                            brand_counts[new_brand] = brand_counts.get(new_brand, 0) + 1
                            
                            # Update ocr_cache metadata
                            meta_info = KNOWN_BRANDS.get(brand_name.lower(), {
                                "brand": new_brand,
                                "category": "Generic Bottle",
                                "volume": "Unknown",
                                "manufacturer": "Unknown",
                                "details": "Post-processed dynamically identified bottle"
                            })
                            cached["brand"] = new_brand
                            cached["status"] = "Recognized"
                            cached["metadata"] = meta_info
                            new_events.append(new_brand)
                
            # Draw color-coded box based on identified brand
            cached = ocr_cache[track_id]
            brand = cached["brand"]
            conf = cached["confidence"]
            water_level = cached["water_level"]
            
            # Bisleri (Green), Kinley (Blue), Aquafina (Red), Other (Cyan), Unknown (Grey)
            brand_upper = brand.upper()
            if brand_upper == "BISLERI":
                color = (0, 255, 0)
            elif brand_upper == "KINLEY":
                color = (255, 180, 50)
            elif brand_upper == "AQUAFINA":
                color = (0, 0, 255)
            elif brand_upper == "UNKNOWN":
                color = (150, 150, 150)
            else:
                color = (255, 255, 0)
                
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 3)
            
            label_text = f"#{track_id} {brand}"
            if water_level is not None:
                label_text += f" W:{water_level}%"
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, 0.45, 2)
            cv2.rectangle(frame_resized, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            text_color = (255, 255, 255) if brand_upper in ["AQUAFINA", "UNKNOWN"] else (0, 0, 0)
            cv2.putText(frame_resized, label_text, (x1, y1 - 5), font, 0.45, text_color, 2, cv2.LINE_AA)
            
        # Add new event detections to history
        for brand in new_events:
            if not recent_detections or recent_detections[0] != brand:
                recent_detections.insert(0, brand)
                recent_detections = recent_detections[:5]
                
        # Draw HUD overlays
        elapsed_time = time.time() - start_time
        draw_hud(frame_resized, fps, elapsed_time, brand_counts, total_session_detections, recent_detections, is_recording, show_fps)
        draw_detail_cards(frame_resized, active_dets, ocr_cache)
        
        # Center title watermark
        watermark_text = "Water Brand & Level Detection System"
        font = cv2.FONT_HERSHEY_SIMPLEX
        tw, th = cv2.getTextSize(watermark_text, font, 0.55, 2)[0]
        cv2.rectangle(frame_resized, (320 - int(tw/2) - 15, 12), (320 + int(tw/2) + 15, 42), (0, 0, 0), -1)
        cv2.rectangle(frame_resized, (320 - int(tw/2) - 15, 12), (320 + int(tw/2) + 15, 42), (100, 100, 100), 1)
        cv2.putText(frame_resized, watermark_text, (320 - int(tw/2), 33), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        if is_recording and video_writer is not None:
            video_writer.write(frame_resized)
            
        if time.time() < toast_timer:
            t_size = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            tx = int((640 - t_size[0]) / 2)
            ty = 580
            cv2.rectangle(frame_resized, (tx - 15, ty - 22), (tx + t_size[0] + 15, ty + 10), (0, 0, 0), -1)
            cv2.rectangle(frame_resized, (tx - 15, ty - 22), (tx + t_size[0] + 15, ty + 10), (0, 255, 255), 1)
            cv2.putText(frame_resized, toast_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("Water Brand Detection System", frame_resized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, frame_resized)
            print(f"[INFO] Saved screenshot to: {filename}")
            toast_text = "Screenshot Saved"
            toast_timer = time.time() + 2.0
        elif key == ord('r'):
            if not is_recording:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(recordings_dir, f"demo_{timestamp}.mp4")
                h_rec, w_rec = frame_resized.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w_rec, h_rec))
                is_recording = True
                print(f"[INFO] Started recording: {filename}")
                toast_text = "Recording Started"
                toast_timer = time.time() + 2.0
            else:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("[INFO] Recording stopped and saved.")
                toast_text = "Recording Saved"
                toast_timer = time.time() + 2.0
        elif key == ord('f'):
            show_fps = not show_fps
            toast_text = "FPS Overlay: ON" if show_fps else "FPS Overlay: OFF"
            toast_timer = time.time() + 1.5
        elif key == ord('c'):
            brand_counts.clear()
            recent_detections.clear()
            ocr_cache.clear()
            tracker.next_id = 0
            tracker.trackers.clear()
            total_session_detections = 0
            start_time = time.time()
            print("[INFO] Counters and session metrics reset successfully.")
            toast_text = "Dashboard Reset"
            toast_timer = time.time() + 1.5
            
    cap.release()
    if video_writer is not None:
        video_writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
        
    # Write session summary text log
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(logs_dir, f"session_summary_{timestamp}.txt")
    
    with open(summary_file, "w") as sf:
        sf.write("==================================================\n")
        sf.write("        WATER BRAND DETECTION SESSION SUMMARY      \n")
        sf.write("==================================================\n")
        sf.write(f"Session Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        sf.write(f"Session End:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        sf.write(f"Elapsed Time:  {elapsed_time:.1f} seconds\n")
        sf.write(f"Frames:        {frame_count}\n")
        sf.write(f"Average FPS:   {avg_fps:.1f}\n")
        sf.write("--------------------------------------------------\n")
        sf.write("Detections:\n")
        for b_name, b_count in brand_counts.items():
            sf.write(f"  {b_name[:12].ljust(12)} Count: {b_count}\n")
        sf.write(f"  Total Session : {total_session_detections}\n")
        sf.write("==================================================\n")
        
    print(f"\n[INFO] Session summary saved to: {summary_file}")
    print("Application closed.")

if __name__ == "__main__":
    start_webcam_detection()
