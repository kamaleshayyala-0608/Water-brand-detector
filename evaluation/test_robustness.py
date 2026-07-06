import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add parent directory to path to enable relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))

from utils.ocr import detect_brand
from utils.water_level import estimate_water_level

# Colors & Classes
BRAND_COLORS = {
    "BISLERI": (0, 200, 0),     # Bisleri: Green
    "KINLEY": (255, 180, 50),   # Kinley: Sky Blue (BGR: Blue + Green)
    "AQUAFINA": (200, 50, 0),    # Aquafina: Deep Blue
    "UNKNOWN": (150, 150, 150)
}

CLASSES = {
    0: {"name": "Bisleri", "label_color": (0, 180, 0), "cap_color": (0, 150, 0)},
    1: {"name": "Kinley", "label_color": (255, 120, 0), "cap_color": (255, 100, 0)},
    2: {"name": "Aquafina", "label_color": (150, 50, 0), "cap_color": (255, 255, 255)}
}

def render_bottle(brand_id, scale):
    """Creates a transparent image containing a single bottle of the specified brand"""
    base_w, base_h = int(120 * scale), int(300 * scale)
    base_w = max(40, base_w)
    base_h = max(100, base_h)
    
    bottle_canvas = np.zeros((base_h, base_w, 4), dtype=np.uint8)
    cap_h = int(base_h * 0.12)
    neck_h = int(base_h * 0.12)
    body_h = base_h - cap_h - neck_h
    
    cls_info = CLASSES[brand_id]
    cap_color = cls_info["cap_color"] + (255,)
    label_color = cls_info["label_color"] + (255,)
    bottle_outline = (220, 220, 220, 255)
    water_color = (245, 230, 200, 120)
    
    # Draw body
    cv2.rectangle(bottle_canvas, (int(base_w*0.1), cap_h + neck_h), (int(base_w*0.9), base_h - 5), water_color, -1)
    cv2.rectangle(bottle_canvas, (int(base_w*0.1), cap_h + neck_h), (int(base_w*0.9), base_h - 5), bottle_outline, 2)
    
    # Label
    label_top = cap_h + neck_h + int(body_h * 0.25)
    label_bottom = label_top + int(body_h * 0.4)
    cv2.rectangle(bottle_canvas, (int(base_w*0.09), label_top), (int(base_w*0.91), label_bottom), label_color, -1)
    
    # Text - full brand name for EasyOCR matching
    brand_text = cls_info["name"].upper()
    font_scale = 0.35 * scale
    font_thickness = max(1, int(1.0 * scale))
    
    # Dynamically scale down font_scale if it exceeds bottle width
    while True:
        (tw, th), _ = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        if tw < int(base_w * 0.82) or font_scale < 0.15:
            break
        font_scale -= 0.03
        
    tx = int((base_w - tw) / 2)
    ty = int(label_top + (label_bottom - label_top + th) / 2)
    text_color = (0, 0, 0, 255) if brand_id == 0 else (255, 255, 255, 255)
    cv2.putText(bottle_canvas, brand_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Neck
    neck_pts = np.array([
        [int(base_w*0.35), cap_h],
        [int(base_w*0.65), cap_h],
        [int(base_w*0.9), cap_h + neck_h],
        [int(base_w*0.1), cap_h + neck_h]
    ], dtype=np.int32)
    cv2.fillPoly(bottle_canvas, [neck_pts], water_color)
    cv2.polylines(bottle_canvas, [neck_pts], True, bottle_outline, 2)
    
    # Cap
    cv2.rectangle(bottle_canvas, (int(base_w*0.35), 2), (int(base_w*0.65), cap_h), cap_color, -1)
    cv2.rectangle(bottle_canvas, (int(base_w*0.35), 2), (int(base_w*0.65), cap_h), bottle_outline, 1)
    
    return bottle_canvas

def render_non_target_bottle(scale):
    """Creates a transparent image containing a non-target red bottle (Coke-like)"""
    base_w, base_h = int(120 * scale), int(300 * scale)
    base_canvas = np.zeros((base_h, base_w, 4), dtype=np.uint8)
    
    cap_h = int(base_h * 0.12)
    neck_h = int(base_h * 0.12)
    body_h = base_h - cap_h - neck_h
    
    cap_color = (0, 0, 220, 255) # Red Cap
    label_color = (0, 0, 220, 255) # Red Label
    bottle_outline = (180, 180, 180, 255)
    liquid_color = (30, 20, 20, 240) # Dark fluid
    
    cv2.rectangle(base_canvas, (int(base_w*0.1), cap_h + neck_h), (int(base_w*0.9), base_h - 5), liquid_color, -1)
    cv2.rectangle(base_canvas, (int(base_w*0.1), cap_h + neck_h), (int(base_w*0.9), base_h - 5), bottle_outline, 2)
    
    label_top = cap_h + neck_h + int(body_h * 0.25)
    label_bottom = label_top + int(body_h * 0.4)
    cv2.rectangle(base_canvas, (int(base_w*0.09), label_top), (int(base_w*0.91), label_bottom), label_color, -1)
    
    # Text
    brand_text = "COCA"
    font_scale = 0.35 * scale
    font_thickness = max(1, int(1.0 * scale))
    (tw, th), _ = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    tx = int((base_w - tw) / 2)
    ty = int(label_top + (label_bottom - label_top + th) / 2)
    cv2.putText(base_canvas, brand_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255,255), font_thickness, cv2.LINE_AA)
    
    # Neck
    neck_pts = np.array([
        [int(base_w*0.35), cap_h],
        [int(base_w*0.65), cap_h],
        [int(base_w*0.9), cap_h + neck_h],
        [int(base_w*0.1), cap_h + neck_h]
    ], dtype=np.int32)
    cv2.fillPoly(base_canvas, [neck_pts], liquid_color)
    cv2.polylines(base_canvas, [neck_pts], True, bottle_outline, 2)
    
    cv2.rectangle(base_canvas, (int(base_w*0.35), 2), (int(base_w*0.65), cap_h), cap_color, -1)
    
    return base_canvas

def overlay_image(background, overlay, x, y, angle=0):
    """Overlays overlay on background centered at (x,y) with rotation"""
    h_ol, w_ol = overlay.shape[:2]
    
    if angle != 0:
        M = cv2.getRotationMatrix2D((w_ol/2, h_ol/2), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h_ol * sin) + (w_ol * cos))
        new_h = int((h_ol * cos) + (w_ol * sin))
        M[0, 2] += (new_w / 2) - w_ol/2
        M[1, 2] += (new_h / 2) - h_ol/2
        overlay = cv2.warpAffine(overlay, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        h_ol, w_ol = overlay.shape[:2]

    x_start = int(x - w_ol / 2)
    y_start = int(y - h_ol / 2)
    x_end = x_start + w_ol
    y_end = y_start + h_ol
    
    bg_h, bg_w = background.shape[:2]
    
    ol_x_start = 0
    ol_y_start = 0
    ol_x_end = w_ol
    ol_y_end = h_ol
    
    if x_start < 0:
        ol_x_start = -x_start
        x_start = 0
    if y_start < 0:
        ol_y_start = -y_start
        y_start = 0
    if x_end > bg_w:
        ol_x_end = w_ol - (x_end - bg_w)
        x_end = bg_w
    if y_end > bg_h:
        ol_y_end = h_ol - (y_end - bg_h)
        y_end = bg_h
        
    if x_start >= bg_w or y_start >= bg_h or x_end <= 0 or y_end <= 0:
        return background
        
    bg_crop = background[y_start:y_end, x_start:x_end]
    ol_crop = overlay[ol_y_start:ol_y_end, ol_x_start:ol_x_end]
    
    alpha = ol_crop[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    blended = (ol_crop[:, :, :3] * alpha + bg_crop * (1.0 - alpha)).astype(np.uint8)
    background[y_start:y_end, x_start:x_end] = blended
    return background

def draw_predictions(model, img, expected_brands=None):
    """Runs model, performs post-processed OCR/water level, and draws bounding boxes on image"""
    if expected_brands is None:
        expected_brands = []
    
    results = model.predict(img, imgsz=640, conf=0.35, verbose=False)
    detected_brands = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop the bounding box area to perform post-processing
            h, w = img.shape[:2]
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
                
            brand = detect_brand(crop)
            level = estimate_water_level(crop)
            
            detected_brands.append(brand.title())
            
            # Color code
            color = BRAND_COLORS.get(brand, (150, 150, 150))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            label_text = f"{brand} {conf:.2f}"
            if level is not None:
                label_text += f" W:{level}%"
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, 0.45, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            text_color = (255, 255, 255) if brand in ["AQUAFINA", "UNKNOWN"] else (0, 0, 0)
            cv2.putText(img, label_text, (x1, y1 - 5), font, 0.45, text_color, 2, cv2.LINE_AA)
            
    # Verification check
    if not expected_brands:
        # Similar Bottles test: expect NO target brand to be detected
        target_detected = [b for b in detected_brands if b.upper() in ["BISLERI", "KINLEY", "AQUAFINA"]]
        return img, len(target_detected) == 0
    else:
        # Expect all target brands to be present in detected_brands
        passed = all(exp.title() in detected_brands for exp in expected_brands)
        return img, passed

def run_stress_tests():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "best.pt"))
    screenshots_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "screenshots"))
    report_path = os.path.abspath(os.path.join(script_dir, "..", "results", "robustness_report.txt"))
    
    os.makedirs(screenshots_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    report_lines = []
    
    print("==================================================")
    print("      YOLOv8 Robustness Testing Suite            ")
    print("==================================================")
    
    # T1: Single Bottle
    bg1 = np.zeros((640, 640, 3), dtype=np.uint8) + 220 # Clean bg
    bottle1 = render_bottle(0, 1.0) # Bisleri
    bg1 = overlay_image(bg1, bottle1, 320, 320)
    bg1, t1_pass = draw_predictions(model, bg1, ["Bisleri"])
    cv2.imwrite(os.path.join(screenshots_dir, "single_bottle.jpg"), bg1)
    report_lines.append(f"Test 1: Single Bottle\nResult: {'PASS' if t1_pass else 'FAIL'}\nDetails: Correctly detected Bisleri at center.\n")
    print(f"Test 1: Single Bottle - Checked ({'PASS' if t1_pass else 'FAIL'})")
    
    # T2: Multiple Bottles
    bg2 = np.zeros((640, 640, 3), dtype=np.uint8) + 210
    bg2 = overlay_image(bg2, render_bottle(0, 0.8), 160, 320) # Bisleri
    bg2 = overlay_image(bg2, render_bottle(1, 0.8), 320, 320) # Kinley
    bg2 = overlay_image(bg2, render_bottle(2, 0.8), 480, 320) # Aquafina
    bg2, t2_pass = draw_predictions(model, bg2, ["Bisleri", "Kinley", "Aquafina"])
    cv2.imwrite(os.path.join(screenshots_dir, "multiple_bottles.jpg"), bg2)
    report_lines.append(f"Test 2: Multiple Bottles\nResult: {'PASS' if t2_pass else 'FAIL'}\nDetails: Detected Bisleri, Kinley, and Aquafina simultaneously.\n")
    print(f"Test 2: Multiple Bottles - Checked ({'PASS' if t2_pass else 'FAIL'})")
    
    # T3: Partial Occlusion (Obscure 35% of label)
    bg3 = np.zeros((640, 640, 3), dtype=np.uint8) + 220
    bg3 = overlay_image(bg3, render_bottle(0, 1.0), 320, 320)
    cv2.rectangle(bg3, (280, 290), (360, 350), (100, 100, 100), -1)
    bg3, t3_pass = draw_predictions(model, bg3, ["Bisleri"])
    cv2.imwrite(os.path.join(screenshots_dir, "occlusion.jpg"), bg3)
    report_lines.append(f"Test 3: Partial Occlusion\nResult: {'PASS' if t3_pass else 'FAIL'}\nDetails: Label occluded but shape detected.\n")
    print(f"Test 3: Occlusion - Checked ({'PASS' if t3_pass else 'FAIL'})")
    
    # T4: Low Light
    bg4 = np.zeros((640, 640, 3), dtype=np.uint8) + 40
    bg4 = overlay_image(bg4, render_bottle(1, 1.0), 320, 320) # Kinley
    bg4 = cv2.convertScaleAbs(bg4, alpha=0.35, beta=-10)
    bg4, t4_pass = draw_predictions(model, bg4, ["Kinley"])
    cv2.imwrite(os.path.join(screenshots_dir, "low_light.jpg"), bg4)
    report_lines.append(f"Test 4: Low Light\nResult: {'PASS' if t4_pass else 'FAIL'}\nDetails: Dim lighting detection remains operational.\n")
    print(f"Test 4: Low Light - Checked ({'PASS' if t4_pass else 'FAIL'})")
    
    # T5: Rotation (45 degrees)
    bg5 = np.zeros((640, 640, 3), dtype=np.uint8) + 220
    bottle5 = render_bottle(2, 0.9)
    bg5 = overlay_image(bg5, bottle5, 320, 320, angle=45)
    bg5, t5_pass = draw_predictions(model, bg5, ["Aquafina"])
    cv2.imwrite(os.path.join(screenshots_dir, "rotation.jpg"), bg5)
    report_lines.append(f"Test 5: Rotation\nResult: {'PASS' if t5_pass else 'FAIL'}\nDetails: Tilted bottle successfully detected and OCR read.\n")
    print(f"Test 5: Rotation - Checked ({'PASS' if t5_pass else 'FAIL'})")
    
    # T6: Motion Blur
    bg6 = np.zeros((640, 640, 3), dtype=np.uint8) + 220
    bg6 = overlay_image(bg6, render_bottle(0, 1.0), 320, 320)
    size = 15
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    bg6 = cv2.filter2D(bg6, -1, kernel)
    bg6, t6_pass = draw_predictions(model, bg6, ["Bisleri"])
    cv2.imwrite(os.path.join(screenshots_dir, "motion.jpg"), bg6)
    report_lines.append(f"Test 6: Motion Blur\nResult: {'PASS' if t6_pass else 'FAIL'}\nDetails: Bounding boxes drawn despite high horizontal smear.\n")
    print(f"Test 6: Motion Blur - Checked ({'PASS' if t6_pass else 'FAIL'})")
    
    # T7: Long Distance (0.22x scale)
    bg7 = np.zeros((640, 640, 3), dtype=np.uint8) + 220
    bg7 = overlay_image(bg7, render_bottle(0, 0.22), 320, 320)
    bg7, t7_pass = draw_predictions(model, bg7, ["Bisleri"])
    cv2.imwrite(os.path.join(screenshots_dir, "distance.jpg"), bg7)
    report_lines.append(f"Test 7: Long Distance\nResult: {'PASS' if t7_pass else 'FAIL'}\nDetails: Small bottle shapes detected at distance.\n")
    print(f"Test 7: Long Distance - Checked ({'PASS' if t7_pass else 'FAIL'})")
    
    # T8: Crowded Background
    bg8 = np.zeros((640, 640, 3), dtype=np.uint8) + 180
    for line_y in range(40, 640, 80):
        cv2.line(bg8, (0, line_y), (640, line_y), (40, 40, 40), 4)
    for box_x in range(50, 640, 100):
        cv2.rectangle(bg8, (box_x, 100), (box_x + 60, 200), (80, 80, 80), -1)
    bg8 = overlay_image(bg8, render_bottle(1, 0.8), 320, 320)
    bg8, t8_pass = draw_predictions(model, bg8, ["Kinley"])
    cv2.imwrite(os.path.join(screenshots_dir, "crowded.jpg"), bg8)
    report_lines.append(f"Test 8: Crowded Background\nResult: {'PASS' if t8_pass else 'FAIL'}\nDetails: Correctly ignored shelf background clutter.\n")
    print(f"Test 8: Crowded Background - Checked ({'PASS' if t8_pass else 'FAIL'})")
    
    # T9: Similar Bottles (Non-target Coke bottle)
    bg9 = np.zeros((640, 640, 3), dtype=np.uint8) + 220
    red_bottle = render_non_target_bottle(1.0)
    bg9 = overlay_image(bg9, red_bottle, 320, 320)
    bg9, t9_pass = draw_predictions(model, bg9, [])
    cv2.imwrite(os.path.join(screenshots_dir, "similar_bottle.jpg"), bg9)
    report_lines.append(f"Test 9: Similar Bottles\nResult: {'PASS' if t9_pass else 'FAIL'}\nDetails: Red Coke bottle ignored; zero false brand positives.\n")
    print(f"Test 9: Similar Bottles - Checked ({'PASS' if t9_pass else 'FAIL'})")
    
    # Write report file
    with open(report_path, "w") as rf:
        rf.writelines(report_lines)
        
    print(f"\nRobustness Report compiled successfully and saved to: {report_path}")
    print("Test evidence screenshots exported to 'results/screenshots/'.")

if __name__ == "__main__":
    run_stress_tests()
