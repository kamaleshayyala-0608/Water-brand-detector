import os
import random
from ultralytics import YOLO

def run_test_prediction():
    # Attempt to locate best.pt
    best_weights = os.path.join("models", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join("water_brand_training", "water_brand_detector", "weights", "best.pt")
        
    if not os.path.exists(best_weights):
        print("Error: Could not locate best.pt weights.")
        return

    print(f"Loading model from {best_weights}...")
    model = YOLO(best_weights)
    
    # Grab a random image from test split
    test_img_dir = os.path.join("dataset", "images", "test")
    if not os.path.exists(test_img_dir):
        print(f"Error: Test directory '{test_img_dir}' does not exist.")
        return
        
    test_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print("No test images found to predict on.")
        return
        
    img_name = random.choice(test_images)
    img_path = os.path.join(test_img_dir, img_name)
    
    print(f"Predicting on random test image: {img_path}")
    results = model.predict(source=img_path, conf=0.5, save=True, project="test_predictions", name="predictions")
    print("Prediction run successfully! Check the output under the 'test_predictions/predictions/' directory.")

if __name__ == "__main__":
    run_test_prediction()
