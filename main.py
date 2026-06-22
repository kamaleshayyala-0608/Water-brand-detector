import os
import sys
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    while True:
        clear_screen()
        print("=" * 55)
        print("     W A T E R   B R A N D   D E T E C T O R")
        print("               M A I N   L A U N C H E R")
        print("=" * 55)
        print("  1. Start Live Webcam Detection")
        print("     [Q] Quit Window | [S] Screenshot | [R] Record Video")
        print("     [F] Toggle FPS   | [C] Reset Counters")
        print("-" * 55)
        print("  2. Process Test Video File (videos/input.mp4)")
        print("  3. Run Robustness & Real-World Stress Tests")
        print("  4. Evaluate Model Metrics on Test Split")
        print("  5. Exit")
        print("=" * 55)
        
        try:
            choice = input("Enter selection [1-5]: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if choice == '1':
            print("\nInitializing Webcam Detection system...")
            try:
                from inference.webcam import start_webcam_detection
                start_webcam_detection()
            except Exception as e:
                print(f"\nError running webcam detection: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == '2':
            print("\nInitializing Video Processing...")
            input_video = os.path.join(script_dir, "videos", "input.mp4")
            if not os.path.exists(input_video):
                print(f"\n[WARN] Input video {input_video} not found. Generating synthetic test video...")
                try:
                    from utils.generate_test_video import generate_video
                    generate_video()
                except Exception as e:
                    print(f"Error generating test video: {e}")
            try:
                from inference.video import process_video_detection
                process_video_detection()
            except Exception as e:
                print(f"Error processing video: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == '3':
            print("\nInitializing Robustness Testing Suite...")
            try:
                from evaluation.test_robustness import run_stress_tests
                run_stress_tests()
            except Exception as e:
                print(f"Error running robustness tests: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == '4':
            print("\nInitializing Validation and Evaluation checks...")
            try:
                from evaluation.validation import evaluate
                evaluate()
            except Exception as e:
                print(f"Error running validation: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == choice == '5':
            print("\nExiting Launcher. Goodbye!")
            break
        else:
            print("\nInvalid selection. Please enter a number between 1 and 5.")
            time.sleep(1.5)

if __name__ == "__main__":
    main()