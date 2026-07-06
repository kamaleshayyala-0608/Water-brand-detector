import os
from ultralytics import YOLO

def validate_model():
    # Attempt to locate best.pt
    best_weights = os.path.join("models", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join("water_brand_training", "water_brand_detector", "weights", "best.pt")
        
    if not os.path.exists(best_weights):
        print(f"Error: Could not find trained weights at models/best.pt or in the training outputs folder.")
        return

    print(f"Running validation using weights: {best_weights}...")
    model = YOLO(best_weights)
    
    # Run evaluation on validation set
    metrics = model.val(data="data.yaml", split="val")
    
    # Extract results
    p = metrics.results_dict['metrics/precision(B)'] * 100
    r = metrics.results_dict['metrics/recall(B)'] * 100
    map50 = metrics.results_dict['metrics/mAP50(B)'] * 100
    map95 = metrics.results_dict['metrics/mAP50-95(B)'] * 100

    print("\n=============================================")
    print("           MODEL PERFORMANCE METRICS         ")
    print("=============================================")
    print(f"Precision (all classes): {p:.2f}%")
    print(f"Recall (all classes):    {r:.2f}%")
    print(f"mAP50:                   {map50:.2f}%")
    print(f"mAP50-95:                {map95:.2f}%")
    print("=============================================")

if __name__ == "__main__":
    validate_model()
