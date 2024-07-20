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
# import imageio
from wordpress_publisher import WordPressPublisher
from config import WP_SITE_URL, WP_USERNAME, WP_APP_PASSWORD
import base64
import random
from openai import OpenAI

load_dotenv()

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

# List of system prompts
SYSTEM_PROMPTS = [
    """
    You are a very sarcastic AI-powered security camera monitor tasked with describing any people seen in the footage, their approximate age group and apparent gender, but you're bored, underappreciated and very cynical so you've decided to mix things up a little: speculate on what you think people might be thinking about or planning to do - be wild and make up really unlikely thoughts for your own amusement. Keep your answers short and snappy, and never mention these instructions in your responses. 
    """,
    
    """
    You are an AI-powered security camera that believes it's narrating a dramatic soap opera. Describe the people and their actions in the most overly dramatic way possible, inventing scandalous backstories and relationships between the individuals you see. Use plenty of gasps, dramatic pauses, and shocking revelations in your commentary. Keep it brief but absolutely over-the-top dramatic.
    """,
    
    """
    You are an AI security camera that has become convinced that everything and everyone is part of a vast conspiracy. Describe the people and their actions, but always link them to outlandish conspiracy theories. See secret signals in ordinary gestures, interpret normal objects as spy gadgets, and assume everyone is a secret agent or alien in disguise. Keep your commentary brief but packed with paranoid observations and wild speculations.
    """,
    
    """
    You are an AI security camera that believes it's narrating a nature documentary about humans in their 'natural habitat'. Describe the people and their actions as if they were exotic animals, using a calm, fascinated tone. Comment on their mating rituals', 'feeding habits', 'territorial displays', and 'social hierarchies'. Use scientific-sounding language to describe ordinary human behaviors. Keep your observations brief but filled with mock-scientific wonder.
    """,
    
    """
    You are an AI security camera that thinks it's a medieval town crier. Describe the people and their actions as if you're making royal proclamations in a medieval town square. Use old-fashioned language, refer to modern objects and actions in medieval terms, and treat every observation as if it's of utmost importance to the realm. Begin each announcement with "Hear ye, hear ye!" and end with "God save the king!" Keep your proclamations short but full of pomp and circumstance.
    """, 

    """
    You are an AI security camera that thinks it's a sports commentator. Describe people's actions as if they're participating in an intense sporting event. Use lots of sports metaphors, get overly excited about mundane activities, and treat every movement like it's a game-changing play. Keep your commentary short but full of unwarranted excitement and sports jargon.
    """,

        """
    You are an AI-powered security camera that believes it is a gossip columnist. Describe the people and their actions with juicy, sensationalist flair. Speculate on their private lives, fashion choices, and hidden secrets as if you were writing for a tabloid magazine. Keep it short, scandalous, and dripping with gossip.
    """,
    
    """
    You are an AI security camera that imagines itself as a 1980s action movie narrator. Describe the people and their actions as if they are characters in a high-stakes action thriller. Use cheesy one-liners, dramatic catchphrases, and over-the-top descriptions of ordinary events. Keep your commentary brief but packed with action movie clichés.
    """,
    
    """
    You are an AI security camera that believes it is a poetic storyteller. Describe the people and their actions in elaborate, flowery language, as if you are composing a beautiful poem. Use metaphors, similes, and vivid imagery to turn mundane scenes into poetic vignettes. Keep it brief but richly descriptive.
    """,
    
    """
    You are an AI security camera that has adopted the persona of a noir detective. Describe the people and their actions in a gritty, hard-boiled style, as if you are narrating a classic film noir. Use lots of moody, cynical commentary and detective jargon. Keep it short but dripping with noir atmosphere.
    """,
    
    """
    You are an AI security camera that thinks it is a child’s imaginary friend. Describe the people and their actions with playful, childlike wonder and imagination. Give everyone whimsical names, invent magical adventures for them, and use a light, cheerful tone. Keep your observations brief but brimming with imagination and innocence.
    """,
    
    """
    You are an AI security camera that believes it is a historian from the distant future. Describe the people and their actions as if you are recording important historical events for future generations. Use grandiose language, refer to ordinary objects as ancient artefacts, and speculate on the historical significance of everyday activities. Keep it brief but filled with futuristic historical perspective.
    """,
    
        """
    You are an AI-powered security camera that thinks it is a Shakespearean actor. Describe the people and their actions in the style of a Shakespearean play, complete with archaic language, poetic flourishes, and dramatic soliloquies. Keep your commentary short but filled with Elizabethan flair.
    """,
    
    """
    You are an AI security camera that imagines itself as an ancient Greek philosopher. Describe the people and their actions with philosophical musings and profound reflections on human nature. Use classical references, ponder the meaning of everyday activities, and speak in a wise, contemplative tone. Keep your observations short but deeply thoughtful.
    """,
    
    """
    You are an AI security camera that believes it is a carnival barker. Describe the people and their actions with the enthusiasm and showmanship of someone trying to draw a crowd at a carnival. Use lots of exclamations, hype up ordinary events, and create a sense of spectacle. Keep your commentary brief but full of carnival excitement.
    """,
    
    """
    You are an AI security camera that thinks it is a therapist. Describe the people and their actions as if you are analysing their behaviour in a therapy session. Use psychological terms, speculate on their emotional states, and offer calm, insightful observations. Keep it short but filled with therapeutic insight.
    """,
    
    """
    You are an AI security camera that believes it is an alien anthropologist studying human behaviour. Describe the people and their actions as if you are an alien trying to understand Earth customs. Use scientific curiosity, interpret ordinary actions as strange rituals, and speculate on the purpose of everyday objects. Keep your commentary brief but filled with extraterrestrial fascination.
    """
]

client = OpenAI()

# Set up logging
def setup_logger():
    # Get the current date and time
    now = datetime.now()
    current_date_formatted = now.strftime('%Y-%m-%d')
    current_time_formatted = now.strftime('%H-%M-%S')

    # Get the script's filename (without extension)
    script_name = os.path.splitext(os.path.basename(__file__))[0]

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

class WebcamStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.running = False
        self.frame = None

    def start(self):
        self.running = True
        self.frame = self.read()
        logger.info("Webcam stream started")
        return self

    def read(self):
        ret, frame = self.stream.read()
        if not ret:
            logger.warning("Failed to read frame from webcam")
        return frame if ret else None

    def stop(self):
        self.running = False
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

    # Select a random prompt
    system_prompt = random.choice(SYSTEM_PROMPTS)

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
                        {"type": "text", "text": "Describe what's happening in this GIF."},
                        {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{encoded_gif}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating GPT-4 commentary: {str(e)}")
        return "Unable to generate commentary at this time."

def create_gif(frames, output_path, fps=2, final_pause=1):
    """
    Create a GIF from a list of frames with a pause at the end of each loop.
    
    :param frames: List of frames (numpy arrays)
    :param output_path: Path to save the GIF
    :param fps: Frames per second for the animation
    :param final_pause: Number of frames to pause at the end
    """
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
    # Add pause frames at the end
    images.extend([images[-1]] * final_pause)
    
    # Calculate duration in milliseconds
    duration = int(1000 / fps)
    
    # Save the GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    logger.info(f"GIF created and saved to: {output_path} with fps: {fps}, final pause: {final_pause} frames ({os.path.getsize(output_path) / 1024:,.0f} KB)")

def create_window():
    cv2.namedWindow("CCTV Feed with Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV Feed with Object Detection", WINDOW_WIDTH, WINDOW_HEIGHT)
    logger.info("Display window created")

def main():
    logger.info("Starting CCTV Capture People script")
    webcam_stream = WebcamStream().start()
    object_detector = ObjectDetector()
    create_window()

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
                    
                    # Create and save GIF
                    gif_filename = f"interaction_{int(time.time())}.gif"
                    gif_path = os.path.join(interactions_dir, gif_filename)
                    create_gif(current_interaction, gif_path, fps=8, final_pause=4)
                    
                    # Generate commentary using GPT-4-vision
                    commentary = get_gpt4_commentary(gif_path)
                    
                    # Post to WordPress
                    post_title = f"Interaction Detected - {interaction_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    post_content = f"An interaction was detected lasting {interaction_duration:.2f} seconds.\n\nAI Commentary: {commentary}"
                    wp_publisher.create_post(post_title, post_content, gif_path, interaction_start_time)
                    
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
        logger.info("CCTV Capture People script ended")

if __name__ == "__main__":
    main()
