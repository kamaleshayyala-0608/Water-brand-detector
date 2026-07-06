import os
import shutil
from ultralytics import YOLO

def evaluate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "best.pt"))
    data_path = os.path.abspath(os.path.join(script_dir, "..", "data.yaml"))
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Evaluating on test split using: {data_path}")
    metrics = model.val(
        data=data_path,
        split="test",
        project="water_brand_evaluation",
        name="test_evaluation"
    )
    
    p = metrics.results_dict['metrics/precision(B)'] * 100
    r = metrics.results_dict['metrics/recall(B)'] * 100
    map50 = metrics.results_dict['metrics/mAP50(B)'] * 100
    map95 = metrics.results_dict['metrics/mAP50-95(B)'] * 100
    
    print("\n==================================================")
    print("           TEST EVALUATION RESULTS                 ")
    print("==================================================")
    print(f"Test Images: 60")
    print(f"Precision:   {p:.2f}%")
    print(f"Recall:      {r:.2f}%")
    print(f"mAP50:       {map50:.2f}%")
    print(f"mAP50-95:    {map95:.2f}%")
    print("==================================================")
    
    # Run test_images folder prediction as well
    test_images_dir = os.path.abspath(os.path.join(script_dir, "..", "test_images"))
    if os.path.exists(test_images_dir) and os.listdir(test_images_dir):
        print(f"\nRunning prediction on unseen images in: {test_images_dir}")
        model.predict(
            source=test_images_dir,
            conf=0.5,
            save=True,
            project="water_brand_evaluation",
            name="predictions"
        )
        print("Predictions saved to 'water_brand_evaluation/predictions/'.")

if __name__ == "__main__":
    evaluate()
