import os
import time
import shutil

# Config
SHARED_FOLDER = "/home/testunot/IndoorUAV-Agent/online_eval/vla_eval/shared_folder"
SIM_INPUT_DIR = os.path.join(SHARED_FOLDER, "sim_input")
SIM_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "sim_output")
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input")
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output")
CONTROLLER_INPUT = os.path.join(SHARED_FOLDER, "controller_input")


def move_files(src_dir, dst_dir, prefix=""):
    """Move the file and add a prefix"""
    moved = False
    for file_name in os.listdir(src_dir):
        if not file_name.endswith('.json'):
            continue

        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, f"{prefix}{file_name}")

        try:
            shutil.move(src_path, dst_path)
            print(f"Moved file: {file_name} -> {os.path.basename(dst_path)}")
            moved = True
        except Exception as e:
            print(f"Failed to move files: {str(e)}")

    return moved


def main():
    print("File monitoring service starts...")

    try:
        while True:
            # Monitor and move files
            moved = False

            # Send mobile simulator output to controller input (add the 'sim_' prefix)
            if move_files(SIM_OUTPUT_DIR, CONTROLLER_INPUT, "sim_"):
                moved = True

            # Move the model output to the controller input (add the 'model_' prefix).
            if move_files(MODEL_OUTPUT_DIR, CONTROLLER_INPUT, "model_"):
                moved = True

            # If no files have been moved, wait a while.
            if not moved:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("File monitoring service stopped")


if __name__ == "__main__":
    main()