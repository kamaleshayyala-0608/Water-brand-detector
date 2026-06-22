import os
import cv2
import yaml

def validate():
    print("==================================================")
    print("      YOLOv8 Three-Split Dataset Validator        ")
    print("==================================================")
    
    # Verify data.yaml existence
    yaml_path = "data.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: YOLOv8 configuration file '{yaml_path}' is missing.")
        return False
        
    try:
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        print("data.yaml loaded successfully.")
        print(f"  path: {cfg.get('path')}")
        print(f"  train: {cfg.get('train')}")
        print(f"  val: {cfg.get('val')}")
        print(f"  test: {cfg.get('test')}")
        print(f"  classes: {cfg.get('names')}")
    except Exception as e:
        print(f"Error loading '{yaml_path}': {e}")
        return False
        
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return False
        
    splits = ["train", "valid", "test"]
    all_ok = True
    
    # Store verification results
    stats = {
        "train": {"images": 0, "labels": 0, "empty_labels": 0, "invalid_classes": 0, "out_of_bounds": 0},
        "valid": {"images": 0, "labels": 0, "empty_labels": 0, "invalid_classes": 0, "out_of_bounds": 0},
        "test": {"images": 0, "labels": 0, "empty_labels": 0, "invalid_classes": 0, "out_of_bounds": 0}
    }
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        
        if not os.path.exists(img_dir):
            print(f"Error: Image directory missing: {img_dir}")
            all_ok = False
            continue
        if not os.path.exists(lbl_dir):
            print(f"Error: Label directory missing: {lbl_dir}")
            all_ok = False
            continue
            
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        stats[split]["images"] = len(images)
        
        for img_file in images:
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)
            txt_path = os.path.join(lbl_dir, base_name + ".txt")
            
            # Check if image opens correctly
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[{split.upper()}] Corrupt image or unable to open: {img_file}")
                    all_ok = False
                    continue
            except Exception as e:
                print(f"[{split.upper()}] Error reading image {img_file}: {e}")
                all_ok = False
                continue
                
            # Check if label exists
            if not os.path.exists(txt_path):
                print(f"[{split.upper()}] Missing matching label for image: {img_file}")
                all_ok = False
                continue
                
            stats[split]["labels"] += 1
            
            # Check label contents
            with open(txt_path, "r") as f:
                lines = f.readlines()
                
            if not lines:
                print(f"[{split.upper()}] Empty label file found: {base_name}.txt")
                stats[split]["empty_labels"] += 1
                all_ok = False
                continue
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"[{split.upper()}] Invalid label format in {base_name}.txt: '{line}'")
                    all_ok = False
                    continue
                    
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    print(f"[{split.upper()}] Non-numeric values in {base_name}.txt: '{line}'")
                    all_ok = False
                    continue
                    
                # Class validation
                if cls_id not in [0]:
                    print(f"[{split.upper()}] Invalid class ID {cls_id} in {base_name}.txt")
                    stats[split]["invalid_classes"] += 1
                    all_ok = False
                    
                # Bounding box bounds validation
                for coord in coords:
                    if coord < 0.0 or coord > 1.0:
                        print(f"[{split.upper()}] Bounding box out of bounds [0, 1] in {base_name}.txt: {coord}")
                        stats[split]["out_of_bounds"] += 1
                        all_ok = False

    # Check split ratio
    total_imgs = sum(stats[s]["images"] for s in splits)
    if total_imgs > 0:
        print("\nDataset Splits Summary:")
        for split in splits:
            pct = stats[split]["images"] / total_imgs * 100
            print(f"  {split.capitalize()}: {stats[split]['images']} images ({pct:.1f}%)")
    else:
        print("Error: No images found in the dataset.")
        all_ok = False
        
    print("\nDetailed statistics:")
    for split in splits:
        print(f"  {split.capitalize()}: Labels = {stats[split]['labels']}, Empty Labels = {stats[split]['empty_labels']}, Invalid Classes = {stats[split]['invalid_classes']}")
    
    if all_ok:
        print("\n[OK] Verification Successful: Dataset is fully compliant with YOLOv8 standard!")
    else:
        print("\n[FAIL] Verification Failed: Please check the errors listed above.")
        
    return all_ok

if __name__ == "__main__":
    validate()
