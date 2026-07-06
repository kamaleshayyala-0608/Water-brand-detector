import os
import torch
from ultralytics import YOLO

def train_model():
    # Load pre-trained model (from models/yolov8n.pt or standard weights)
    model_path = os.path.join("models", "yolov8n.pt")
    if os.path.exists(model_path):
        print(f"Loading local pre-trained model: {model_path}")
        model = YOLO(model_path)
    else:
        print("Local pre-trained yolov8n.pt not found, loading default weights from web...")
        model = YOLO("yolov8n.pt")

    # Detect device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}")

    # Run YOLOv8 fine-tuning
    results = model.train(
        data="data.yaml",
        epochs=2,          # fine-tune for 2 epochs for quick verification
        imgsz=640,
        batch=16,
        workers=0,          # 0 is required on Windows CPU to prevent pickle errors
        device=device,
        patience=1,
        project="water_brand_training",
        name="water_brand_detector"
    )
    print("Training process finished!")
    
    # Copy best weights to models/best.pt
    try:
        best_weights = os.path.join(model.trainer.save_dir, "weights", "best.pt")
        if os.path.exists(best_weights):
            dest_path = os.path.join("models", "best.pt")
            os.makedirs("models", exist_ok=True)
            import shutil
            shutil.copy(best_weights, dest_path)
            print(f"Successfully copied best model weights to: {dest_path}")
        else:
            print(f"[WARN] Trained weights not found at: {best_weights}")
    except Exception as e:
        print(f"Error copying weights: {e}")

if __name__ == "__main__":
    train_model()
