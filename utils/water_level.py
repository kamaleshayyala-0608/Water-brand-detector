import cv2
import numpy as np

def estimate_water_level(bottle_crop):
    """
    Estimates the water level percentage in the bottle crop using Canny Edge Detection
    and Hough Line Transform.
    
    Returns the estimated water percentage (0-100) or None if not found.
    """
    if bottle_crop is None or bottle_crop.size == 0:
        return None
        
    try:
        h, w = bottle_crop.shape[:2]
        if h < 20 or w < 20:
            return None
            
        gray = cv2.cvtColor(bottle_crop, cv2.COLOR_BGR2GRAY)
        
        # Blur slightly to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dynamically set minimum line length to 35% of crop width, minimum 15 pixels
        min_line_length = max(15, int(w * 0.35))
        max_line_gap = max(5, int(w * 0.10))
        
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=25,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is None:
            return None
            
        best_y = None
        
        # Iterate over lines to find horizontal-ish lines representing water level
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is close to horizontal (small delta Y compared to width)
            delta_y = abs(y2 - y1)
            line_w = abs(x2 - x1)
            
            # Line must be reasonably horizontal: slope angle < 8 degrees
            if delta_y < 5 or (line_w > 0 and (delta_y / line_w) < 0.15):
                # We want a line in the middle/upper area (water line is usually upper/middle)
                # Ignore lines too close to bottom (y > 90% of h) or top (y < 10% of h)
                if 0.1 * h < y1 < 0.9 * h:
                    if best_y is None:
                        best_y = y1
                    elif y1 > best_y:
                        # Taking the lowest valid horizontal line in bounds (larger Y value)
                        best_y = y1
                        
        if best_y is None:
            return None
            
        # Calculate water level percentage
        # bottom is h, surface is best_y. So water depth is (h - best_y)
        water_percent = ((h - best_y) / h) * 100.0
        
        # Limit to reasonable range
        water_percent = max(0.0, min(100.0, water_percent))
        
        return round(water_percent, 1)
        
    except Exception as e:
        print(f"[WARN] Water Level Estimation Error: {e}")
        
    return None
