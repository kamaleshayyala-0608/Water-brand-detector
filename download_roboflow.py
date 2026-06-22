import os
import sys

def check_and_install_roboflow():
    try:
        import roboflow
    except ImportError:
        print("roboflow package not found. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        print("roboflow successfully installed!")

def download():
    check_and_install_roboflow()
    from roboflow import Roboflow

    print("==================================================")
    print("      Roboflow Dataset Downloader Tool            ")
    print("==================================================")
    api_key = input("Enter your Roboflow API Key: ").strip()
    workspace = input("Enter your Roboflow Workspace ID: ").strip()
    project_id = input("Enter your Roboflow Project ID: ").strip()
    version = input("Enter your Roboflow Project Version (integer): ").strip()
    
    try:
        version_int = int(version)
    except ValueError:
        print("Error: Version must be an integer (e.g. 1, 2, 3)")
        return
        
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_id)
        print(f"Downloading version {version_int} of project '{project_id}'...")
        
        # Download in yolov8 format into the dataset directory
        dataset = project.version(version_int).download("yolov8", location="./dataset")
        print("\nDataset downloaded successfully to './dataset'!")
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please check your API key, workspace, project ID, and version and try again.")

if __name__ == "__main__":
    download()
