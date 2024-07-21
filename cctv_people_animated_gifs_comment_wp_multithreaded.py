import cv2
import numpy as np
import os
import time
from datetime import datetime
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
from wordpress_publisher import WordPressPublisher
from config import WP_SITE_URL, WP_USERNAME, WP_APP_PASSWORD
import base64
import random
from openai import OpenAI
import  queue
import threading
from typing import List

load_dotenv()

# Global variables
interaction_queue = queue.Queue()
stop_event = threading.Event()

# Define the captured_frames directory
captured_frames_dir = "captured_frames"
os.makedirs(captured_frames_dir, exist_ok=True)

# Define the interactions directory
interactions_dir = "interactions"
os.makedirs(interactions_dir, exist_ok=True)

# Initialize WordPress publisher
wp_publisher = WordPressPublisher(WP_SITE_URL, WP_USERNAME, WP_APP_PASSWORD)

# Constants
FRAME_SKIP = 1  # Process every 5th frame
PERSON_PERSISTENCE = 3  # Require person to be in 3 consecutive frames
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960

# Get the script's filename (without extension)
script_name = os.path.splitext(os.path.basename(__file__))[0]

client = OpenAI()

# Set up logging
def setup_logger():
    # Get the current date and time
    now = datetime.now()
    current_date_formatted = now.strftime('%Y-%m-%d')
    current_time_formatted = now.strftime('%H-%M-%S')


    # Create the LogFiles folder if it doesn't already exist
    if not os.path.exists('LogFiles'):
        os.makedirs('LogFiles')

    # Use the script's filename and the current date to create the log filename
    log_filename = f'LogFiles/{script_name}-{current_date_formatted}-{current_time_formatted}.log'

    logger = logging.getLogger('cctv_capture')
    logger.setLevel(logging.DEBUG)
    
    # Create a rotating file handler
    file_handler = RotatingFileHandler(log_filename, maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def process_interaction_worker():
    while not stop_event.is_set():
        try:
            # Try to get an interaction from the queue
            interaction_data = interaction_queue.get(timeout=1)
            
            # Unpack the interaction data
            frames, interaction_start_time, interaction_duration = interaction_data
            
            # Process the interaction
            base_gif_path = os.path.join(interactions_dir, f"interaction_{int(time.time())}")
            gif_paths = create_multiple_gifs(frames, base_gif_path, fps=8, final_pause=4, max_size_mb=20)
            
            if not gif_paths:
                logger.warning("No GIFs were created for this interaction")
                continue

            # Generate commentary and post to WordPress for each GIF
            for i, gif_path in enumerate(gif_paths, 1):
                commentary, prompt_used = get_gpt4_commentary(gif_path)
                
                # Post to WordPress
                post_title = f"Interaction Detected (Part {i}/{len(gif_paths)}) - {interaction_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                post_content = f"Part {i} of an interaction detected lasting {interaction_duration:.2f} seconds.\n\nAI Commentary: {commentary}\n\nPrompt Used: {prompt_used}"
                wp_publisher.create_post(post_title, post_content, gif_path, interaction_start_time)
            
            logger.info(f"Processed and published interaction from {interaction_start_time}")
            
        except queue.Empty:
            # No interaction in queue, continue waiting
            continue
        except Exception as e:
            logger.error(f"Error processing interaction: {str(e)}", exc_info=True)

def start_worker_thread():
    worker_thread = threading.Thread(target=process_interaction_worker)
    worker_thread.start()
    return worker_thread

def stop_worker_thread(worker_thread):
    stop_event.set()
    worker_thread.join()
    logger.info("Worker thread stopped")

class WebcamStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.running = False
        self.frame = None

    def start(self):
        if not self.stream.isOpened():
            logger.error("Failed to open webcam")
            return None
        self.running = True
        self.frame = self.read()
        logger.info("Webcam stream started")
        return self

    def read(self):
        if not self.running:
            return None
        ret, frame = self.stream.read()
        if not ret:
            logger.warning("Failed to read frame from webcam")
            return None
        return frame

    def stop(self):
        self.running = False
        if self.stream.isOpened():
            self.stream.release()
        logger.info("Webcam stream stopped")

class ObjectDetector:
    def __init__(self):
        logger.info("Initializing ObjectDetector")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {cfg.MODEL.DEVICE}")
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        logger.info("ObjectDetector initialized")

    def detect_objects(self, image):
        return self.predictor(image)

    def draw_detection_borders(self, image, outputs):
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

def get_gpt4_commentary(gif_path):
    # Encode the GIF
    with open(gif_path, "rb") as image_file:
        encoded_gif = base64.b64encode(image_file.read()).decode('utf-8')

    # Set the prompt
    system_prompt = "You are a helpful CCTV monitor tasked with keeping an eye on security an reporting anything suspicious or untoward."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what the people are doing in this animation. Ignore the coloured object detection boxes, labels and likelihoods."},
                        {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{encoded_gif}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content, system_prompt
    except Exception as e:
        logger.error(f"Error generating AI commentary: {str(e)}")
        return "Unable to generate commentary at this time."

def estimate_gif_size(frames, fps, duration):
    """
    Estimate the size of a GIF based on a sample of frames.
    """
    sample_size = min(len(frames), 10)  # Use up to 10 frames for estimation
    sample_frames = frames[:sample_size]
    
    temp_path = "temp_estimation.gif"
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in sample_frames]
    images[0].save(
        temp_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    file_size = os.path.getsize(temp_path)
    os.remove(temp_path)
    
    estimated_size = (file_size / sample_size) * len(frames)
    return estimated_size

def create_gif(frames, output_path, fps=8, final_pause=4, max_size_mb=20):
    """
    Create a GIF from a list of frames with a pause at the end, ensuring the file size is under max_size_mb.
    """
    if len(frames) == 0:
        logger.warning(f"No frames provided to create GIF: {output_path}")
        return 0, 0

    duration = int(1000 / fps)
    estimated_size = estimate_gif_size(frames, fps, duration)
    target_size = max_size_mb * 1024 * 1024 * 0.9  # 90% of max size to leave some margin

    if estimated_size <= target_size:
        frames_to_use = len(frames)
    else:
        frames_to_use = int((target_size / estimated_size) * len(frames))

    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames[:frames_to_use]]
    images.extend([images[-1]] * final_pause)  # Add pause frames

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"GIF created and saved to: {output_path} with fps: {fps}, frames: {frames_to_use}, final pause: {final_pause} frames ({file_size_kb:.0f} KB)")
    return file_size_kb, frames_to_use

def create_multiple_gifs(frames, base_output_path, fps=8, final_pause=4, max_size_mb=20):
    """
    Create multiple GIFs from a list of frames, each under the size limit.
    """
    gif_paths = []
    start_frame = 0
    gif_number = 1

    while start_frame < len(frames):
        gif_path = f"{base_output_path}_{gif_number}.gif"
        file_size_kb, frames_used = create_gif(frames[start_frame:], gif_path, fps, final_pause, max_size_mb)
        if file_size_kb > 0 and frames_used > 0:
            gif_paths.append(gif_path)
            start_frame += frames_used
            gif_number += 1
        else:
            logger.warning(f"Failed to create GIF {gif_number}, skipping remaining frames")
            break

    return gif_paths

def create_window():
    cv2.namedWindow("CCTV Feed with Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV Feed with Object Detection", WINDOW_WIDTH, WINDOW_HEIGHT)
    logger.info("Display window created")

def main():
    logger.info(f"Starting script: {script_name}.py...")
    webcam_stream = WebcamStream().start()
    if webcam_stream is None:
        logger.error("Failed to start webcam stream. Exiting.")
        return

    object_detector = ObjectDetector()
    create_window()

    # Start the worker thread
    worker_thread = start_worker_thread()

    frame_count = 0
    person_detected_count = 0
    current_interaction = []
    interaction_start_time = None
    last_detection_time = 0
    detection_display_duration = 1.0  # Display detection for 1 second

    try:
        while webcam_stream.running:
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            frame = webcam_stream.read()
            if frame is None:
                logger.warning("Failed to read frame, skipping")
                continue

            current_time = time.time()
            outputs = object_detector.detect_objects(frame)
            detected_objects = outputs["instances"].pred_classes.tolist()

            if 0 in detected_objects:  # 0 is the class ID for person in COCO dataset
                person_detected_count += 1
                if person_detected_count >= PERSON_PERSISTENCE:
                    frame_with_detection = object_detector.draw_detection_borders(frame, outputs)
                    last_detection_time = current_time
                    
                    # Start a new interaction if not already started
                    if not interaction_start_time:
                        interaction_start_time = datetime.now()
                        logger.info("New interaction started")
                    
                    # Add frame to current interaction
                    current_interaction.append(frame_with_detection)
            else:
                person_detected_count = 0
                
                # End the current interaction if no person is detected
                if current_interaction:
                    interaction_end_time = datetime.now()
                    interaction_duration = (interaction_end_time - interaction_start_time).total_seconds()
                    logger.info(f"Interaction ended. Duration: {interaction_duration:.2f} seconds. Frames captured: {len(current_interaction)}")
                    
                    # Add the interaction to the queue for processing
                    interaction_queue.put((current_interaction, interaction_start_time, interaction_duration))
                    
                    # Clear the current interaction
                    current_interaction = []
                    interaction_start_time = None

            # Determine which frame to display
            if current_time - last_detection_time < detection_display_duration and 'frame_with_detection' in locals():
                display_frame = frame_with_detection
            else:
                display_frame = frame

            # Display the frame
            cv2.imshow("CCTV Feed with Object Detection", display_frame)

            if cv2.waitKey(1) in [27, ord("q")]:
                logger.info("User requested to quit")
                break

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        webcam_stream.stop()
        cv2.destroyAllWindows()
        stop_worker_thread(worker_thread)
        logger.info("CCTV Capture People script ended")

if __name__ == "__main__":
    main()
