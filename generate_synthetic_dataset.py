import os
import random
import cv2
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Classes: 0: Bisleri, 1: Kinley, 2: Aquafina
CLASSES = {
    0: {"name": "Bisleri", "label_color": (0, 180, 0), "cap_color": (0, 150, 0)},       # BGR: Green label, Green cap
    1: {"name": "Kinley", "label_color": (255, 120, 0), "cap_color": (255, 100, 0)},   # BGR: Blue label, Blue cap
    2: {"name": "Aquafina", "label_color": (150, 50, 0), "cap_color": (255, 255, 255)}  # BGR: Dark Blue label, White cap
}

def create_random_background(width, height):
    """Generates a random synthetic background (table, shelf, desk, wall)"""
    bg_type = random.choice(["solid", "gradient", "noisy", "desk"])
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if bg_type == "solid":
        color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
        img[:] = color
    elif bg_type == "gradient":
        c1 = np.array([random.randint(180, 220), random.randint(180, 220), random.randint(180, 220)])
        c2 = np.array([random.randint(80, 120), random.randint(80, 120), random.randint(80, 120)])
        for y in range(height):
            alpha = y / height
            img[y, :] = (1 - alpha) * c1 + alpha * c2
    elif bg_type == "noisy":
        color = (random.randint(140, 170), random.randint(120, 150), random.randint(100, 130))
        img[:] = color
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else: # desk
        split = random.randint(int(height * 0.3), int(height * 0.6))
        wall_color = (random.randint(190, 220), random.randint(190, 220), random.randint(190, 220))
        desk_color = (random.randint(80, 120), random.randint(100, 140), random.randint(120, 160))
        img[:split] = wall_color
        img[split:] = desk_color
        cv2.line(img, (0, split), (width, split), (50, 50, 50), 2)
        
    return img

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
    
    # Text
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

def overlay_image(background, overlay, x, y, angle=0):
    """Overlays the transparent image onto the background at (x,y) with rotation"""
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
        return background, None
        
    bg_crop = background[y_start:y_end, x_start:x_end]
    ol_crop = overlay[ol_y_start:ol_y_end, ol_x_start:ol_x_end]
    
    alpha = ol_crop[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    blended = (ol_crop[:, :, :3] * alpha + bg_crop * (1.0 - alpha)).astype(np.uint8)
    background[y_start:y_end, x_start:x_end] = blended
    
    mask = ol_crop[:, :, 3] > 10
    if not np.any(mask):
        return background, None
        
    y_indices, x_indices = np.where(mask)
    bbox_x_min = x_start + np.min(x_indices)
    bbox_y_min = y_start + np.min(y_indices)
    bbox_x_max = x_start + np.max(x_indices)
    bbox_y_max = y_start + np.max(y_indices)
    
    box_w = (bbox_x_max - bbox_x_min)
    box_h = (bbox_y_max - bbox_y_min)
    center_x = bbox_x_min + box_w / 2.0
    center_y = bbox_y_min + box_h / 2.0
    
    if box_w < 10 or box_h < 25:
        return background, None
        
    yolo_x = center_x / bg_w
    yolo_y = center_y / bg_h
    yolo_w = box_w / bg_w
    yolo_h = box_h / bg_h
    
    return background, (yolo_x, yolo_y, yolo_w, yolo_h)

def generate_dataset(num_images=600):
    """Generates the dataset with Train/Valid/Test splits and augmentations"""
    img_size = 640
    
    # 70% Train, 20% Valid, 10% Test
    num_train = int(num_images * 0.7)
    num_val = int(num_images * 0.2)
    num_test = num_images - num_train - num_val
    
    print(f"Generating {num_images} synthetic images ({num_train} train, {num_val} valid, {num_test} test)...")
    
    for i in range(num_images):
        if i < num_train:
            split_dir = "train"
        elif i < num_train + num_val:
            split_dir = "valid"
        else:
            split_dir = "test"
            
        img = create_random_background(img_size, img_size)
        
        # Plurality of bottles (1 to 4)
        num_bottles = random.choices([1, 2, 3, 4], weights=[45, 35, 15, 5])[0]
        bboxes = []
        
        # Distribute bottles on screen
        x_offsets = np.linspace(85, img_size - 85, num_bottles)
        if num_bottles > 1:
            x_offsets += np.random.uniform(-25, 25, num_bottles)
            
        for b_idx in range(num_bottles):
            brand_id = random.randint(0, 2)
            
            # Distance / scale variation
            scale = random.uniform(0.55, 1.25)
            
            # Y position based on scale
            y_base = int(img_size * 0.5) + int(img_size * 0.25 * (scale - 0.55))
            y_pos = y_base + random.randint(-30, 30)
            x_pos = int(np.clip(x_offsets[b_idx], 60, img_size - 60))
            
            # Roboflow rotation augmentation (±15°)
            angle = random.uniform(-15, 15)
            
            bottle_overlay = render_bottle(brand_id, scale)
            img, bbox = overlay_image(img, bottle_overlay, x_pos, y_pos, angle)
            
            if bbox is not None:
                bboxes.append((brand_id, bbox[0], bbox[1], bbox[2], bbox[3]))
                
        # Roboflow Exposure & Brightness Simulation (±20%)
        alpha = random.uniform(0.8, 1.2)  # contrast/exposure
        beta = random.randint(-20, 20)    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Roboflow Blur Simulation
        if random.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
        # Roboflow Horizontal Flip Simulation (50% chance)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            flipped_bboxes = []
            for bbox in bboxes:
                brand_id, x_c, y_c, w, h = bbox
                flipped_bboxes.append((brand_id, 1.0 - x_c, y_c, w, h))
            bboxes = flipped_bboxes
            
        img_filename = f"{split_dir}_{i}.jpg"
        txt_filename = f"{split_dir}_{i}.txt"
        
        img_path = os.path.join("dataset", "images", split_dir, img_filename)
        txt_path = os.path.join("dataset", "labels", split_dir, txt_filename)
        
        cv2.imwrite(img_path, img)
        
        with open(txt_path, "w") as f:
            for bbox in bboxes:
                # Write 0 as the class ID for the 'bottle' class
                f.write(f"0 {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_images} images...")
            
    print("Dataset generation completed successfully!")

if __name__ == "__main__":
    generate_dataset(num_images=600)
