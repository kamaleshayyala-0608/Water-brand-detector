import easyocr
import torch

# Initialize reader once with GPU support if available
print("Initializing EasyOCR Engine...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
print(f"EasyOCR Engine initialized successfully (GPU: {torch.cuda.is_available()}).")

KNOWN_BRANDS = [
    "BISLERI",
    "KINLEY",
    "AQUAFINA",
    "BAILLEY",
    "HIMALAYAN",
    "RAIL NEER"
]

def detect_brand(image):
    """
    Detects brand name inside the bottle bounding box crop using EasyOCR.
    Returns:
      - The matched brand name from KNOWN_BRANDS if found.
      - The most prominent read label text (largest bbox area) if it's not a known brand.
      - 'UNKNOWN' if no readable text is found.
    """
    if image is None or image.size == 0:
        return "UNKNOWN"
        
    try:
        results = reader.readtext(image)
        if not results:
            return "UNKNOWN"
            
        # 1. Direct or Substring match against KNOWN_BRANDS
        # Results format: [([[x0, y0], ...], text, confidence), ...]
        all_texts = []
        for bbox, text, conf in results:
            text_upper = text.strip().upper()
            all_texts.append(text_upper)
            
            # Direct match
            word_clean = "".join([c for c in text_upper if c.isalnum()]).strip()
            if word_clean in KNOWN_BRANDS:
                return word_clean
                
            # Substring match
            for kb in KNOWN_BRANDS:
                if len(word_clean) >= 3 and (word_clean in kb or kb in word_clean):
                    return kb

        # Combine read text to do a string match against KNOWN_BRANDS
        combined_text = " ".join(all_texts)
        for kb in KNOWN_BRANDS:
            if kb in combined_text:
                return kb

        # 2. Dynamic brand recognition: Find the most prominent text (largest bounding box area)
        largest_area = 0
        prominent_text = ""
        
        for bbox, text, conf in results:
            try:
                # Calculate bounding box area
                # bbox is [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                w_text = abs(bbox[1][0] - bbox[0][0])
                h_text = abs(bbox[2][1] - bbox[1][1])
                area = w_text * h_text
            except Exception:
                area = 0
                
            text_clean = text.strip().upper()
            
            # Filter out numbers only, short texts, or empty texts
            clean_alnum = "".join([c for c in text_clean if c.isalnum()])
            if len(clean_alnum) >= 3 and not clean_alnum.isdigit():
                if area > largest_area:
                    largest_area = area
                    prominent_text = text_clean
                    
        if prominent_text:
            return prominent_text
            
        # 3. Fallback: First read text that has at least 2 characters and is not digits
        for bbox, text, conf in results:
            text_clean = text.strip().upper()
            clean_alnum = "".join([c for c in text_clean if c.isalnum()])
            if len(clean_alnum) >= 2 and not clean_alnum.isdigit():
                return text_clean
                
    except Exception as e:
        print(f"[WARN] OCR Error: {e}")
        
    return "UNKNOWN"

