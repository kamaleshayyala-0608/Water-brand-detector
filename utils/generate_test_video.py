import os
import cv2

def make_test_video():
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    test_img_dir = os.path.join("dataset", "images", "test")
    if not os.path.exists(test_img_dir):
        print(f"Error: Test image directory missing at {test_img_dir}")
        return
        
    images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("Error: No test images found.")
        return
        
    print(f"Found {len(images)} images in test set. Compiling 60 frames into test video...")
    
    # Read first frame for size
    first_frame = cv2.imread(os.path.join(test_img_dir, images[0]))
    h, w = first_frame.shape[:2]
    
    output_path = os.path.join(videos_dir, "input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))  # 10 fps
    
    for i in range(60):
        img_path = os.path.join(test_img_dir, images[i % len(images)])
        img = cv2.imread(img_path)
        out.write(img)
        
    out.release()
    print(f"Successfully generated test video: {output_path}")

if __name__ == "__main__":
    make_test_video()
