import os
import json
import time
from test_sim import setup_simulator, get_img
import cv2

os.environ["EGL_DEVICE_ID"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__GLVND_EXPOSE_NATIVE_CONTEXTS"] = "1"

# Config
SHARED_FOLDER = "/home/testunot/IndoorUAV-Agent/online_eval/vla_eval/shared_folder"
SIM_INPUT_DIR = os.path.join(SHARED_FOLDER, "sim_input")
SIM_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "sim_output")
IMAGE_STORAGE = os.path.join(SHARED_FOLDER, "images")
os.makedirs(SIM_INPUT_DIR, exist_ok=True)
os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_STORAGE, exist_ok=True)


class SimulatorService:
    def __init__(self):
        self.sim = None
        self.agent = None
        self.current_glb_path = None

    def process_file(self, file_path):
        try:
            # 1. Check for termination signal
            if "terminate.json" in file_path:
                print("Received termination signal, shut down emulator")
                if self.sim:
                    self.sim.close()
                    self.sim = None
                os.remove(file_path)
                return True
            
            # Potential Problem if it cant load f or f does not exist.
            time.sleep(0.2)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 3. Parse Data
            episode_key = data.get("episode_key", "")
            coords = data.get("coordinates", [])
            glb_path = data.get("glb_path", None)
            is_new_scene = data.get("is_new_scene", False)
            print(f"Processing episode: {episode_key}")
            # 4. Initialize Simulator if needed
            if is_new_scene and glb_path:
                if self.sim:
                    self.sim.close()
                print(f"Initialize the scene: {glb_path}")
                self.sim = setup_simulator(glb_path)
                self.agent = self.sim.initialize_agent(0)
                self.current_glb_path = glb_path
            elif not self.sim:
                print("Error: Scene not initialized")
                return False

            # 5. Render Image
            timestamp = time.time()
            safe_episode_key = episode_key.replace('/', '_').replace(':', '_').replace(' ', '_')
            image_filename = f"image_{safe_episode_key}_{timestamp}.png"
            image_path = os.path.join(IMAGE_STORAGE, image_filename)

            # Create temp file for test_sim (legacy requirement)
            temp_coords_file = f"temp_coords_{timestamp}.json"
            with open(temp_coords_file, 'w') as f:
                json.dump({"action": coords}, f)

            # Get image from Habitat
            frame = get_img(temp_coords_file, self.sim, self.agent)

            # Save Image to disk
            cv2.imwrite(image_path, frame)

            # Clean up temp file
            os.remove(temp_coords_file)

            # Write Output for Controller
            output_file = os.path.join(SIM_OUTPUT_DIR, f"sim_output_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "episode_key": episode_key,
                    "coordinates": coords,
                    "image_path": image_path
                }, f)

            print(f"Image generated: {image_path}")
            return True

        except Exception as e:
            print(f"Unable to process: {file_path} Error: {str(e)}")
            return False
        finally:
            # Always clean up the input file so we don't process it twice
            if os.path.exists(file_path) and "terminate.json" not in file_path:
                os.remove(file_path)


def main():
    print("Simulation service starts...")
    simulator = SimulatorService()

    try:
        while True:
            # Scan directory for new JSON files
            processed = False
            for file_name in os.listdir(SIM_INPUT_DIR):
                if not file_name.endswith('.json'):
                    continue

                file_path = os.path.join(SIM_INPUT_DIR, file_name)
                if simulator.process_file(file_path):
                    processed = True

            # Sleep to save CPU if no work was done
            if not processed:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Simulator Stopped")

    finally:
        if simulator.sim:
            simulator.sim.close()


if __name__ == "__main__":
    main()
