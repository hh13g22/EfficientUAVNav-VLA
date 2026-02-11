import os
import json
import time
import numpy as np
from PIL import Image


# Initialize model
def init_model():
    from openpi.training import config
    from openpi.policies import policy_config

    config = config.get_config("pi0_uav_low_mem_finetune") # Needs to be addressed. I cant find this config probably something else.
    checkpoint_dir = "/home/testunot/IndoorUAV-Agent/checkpoint/29999"
    return policy_config.create_trained_policy(config, checkpoint_dir)


def infer(policy, inputs):
    return policy.infer(inputs)["actions"]


policy = init_model()

# Config
SHARED_FOLDER = "/home/testunot/IndoorUAV-Agent/online_eval/vla_eval/shared_folder"
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input")
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output")
INSTRUCTIONS_DIR = os.path.join(SHARED_FOLDER, "instructions")
os.makedirs(MODEL_INPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)


class ModelService:
    def __init__(self):
        self.current_episode = None
        self.instruction = None
        self.end_coords = None
        self.ref_image_array = None
        self.last_start_image_path = None  # Follow the path of the last loaded image

    def load_instruction(self):
        """Load current instructions and update reference image"""
        instruction_file = os.path.join(INSTRUCTIONS_DIR, "current_instruction.json")
        if os.path.exists(instruction_file):
            time.sleep(0.2)
            with open(instruction_file, 'r') as f:
                data = json.load(f)

            # Check if it's a new episode.
            if self.current_episode != data.get("episode_key"):
                self.current_episode = data.get("episode_key")
                self.instruction = data.get("instruction")
                self.end_coords = data.get("end_coords")
                self.last_start_image_path = None  # Reset image path

            # Get the starting image path
            start_image_path = data.get("start_image_path")

            # Check if there is a new starting image path.
            if start_image_path and start_image_path != self.last_start_image_path:
                self.last_start_image_path = start_image_path

                # Load reference image
                if os.path.exists(start_image_path):
                    ref_img = Image.open(start_image_path).convert('RGB')
                    self.ref_image_array = np.asarray(ref_img, dtype=np.uint8)
                    print(f"Update reference image: {os.path.basename(start_image_path)}")
                else:
                    print(f"Error: Reference Image Does not exist: {start_image_path}")
                    self.ref_image_array = None

    def process_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            episode_key = data.get("episode_key", "")
            image_path = data.get("image_path", "")
            coordinates = data.get("coordinates", [])
            print(f"Processing episode: {episode_key}, current: {self.current_episode}")
            # Load current instructions
            self.load_instruction()

            # Check if it matches the current episode
            if episode_key != self.current_episode:
                print(f"Error: Not the same episode ({episode_key} vs {self.current_episode})")
                return False

            # Get input image
            if not os.path.exists(image_path):
                print(f"Error: Inpout Image file does not exist - {image_path}")
                return False

            img = Image.open(image_path).convert('RGB')
            img_array = np.asarray(img, dtype=np.uint8)

            # Ensure that coordinates has 4 dimensions
            if len(coordinates) < 4:
                coordinates = coordinates + [0.0] * (4 - len(coordinates))

            state = np.array(coordinates[:4], dtype=np.float32)

            print(f"Ref image exists: {self.ref_image_array is not None}")

            # Prepare model input
            example = {
                "observation/image": img_array, # model input from simulation
                "observation/ref_image": self.ref_image_array, # Model input from screenshot
                "observation/state": state,
                "task": self.instruction
            }

            # Perform inference
            output_all = infer(policy, example) # Action Chunk Output
            output = output_all[9] # Selects 10th action
            new_coords = output[:4].tolist() # Base action head consists of more action. Instead take first 4 xyz yaw.
            # Save model output
            timestamp = time.time()
            output_file = os.path.join(MODEL_OUTPUT_DIR, f"model_output_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "episode_key": self.current_episode,
                    "coordinates": new_coords # model output
                }, f)

            print(f"Inference Complete - New Coordinates: {new_coords}")
            return True

        except Exception as e:
            print(f"Unable to process file: {file_path} Error: {str(e)}")
            return False
        finally:
            # Clean input files
            if os.path.exists(file_path):
                os.remove(file_path)


def main():
    print("Model inference service started...")
    model_service = ModelService()

    try:
        while True:
            # Regularly check for instruction updates
            model_service.load_instruction()
            # Check input directory
            processed = False
            for file_name in os.listdir(MODEL_INPUT_DIR):
                if not file_name.endswith('.json'):
                    continue

                # found simulator output
                file_path = os.path.join(MODEL_INPUT_DIR, file_name)
                if model_service.process_file(file_path):
                    processed = True

            # If no files are being processed, wait a while.
            if not processed:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Model inference service stoopped")


if __name__ == "__main__":

    main()
