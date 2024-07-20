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
import logging
from logging.handlers import RotatingFileHandler

# Define the captured_frames directory
captured_frames_dir = "captured_frames"
os.makedirs(captured_frames_dir, exist_ok=True)

# Constants
FRAME_SKIP = 1  # Process every 5th frame
PERSON_PERSISTENCE = 3  # Require person to be in 3 consecutive frames
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960

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
                    
                    # Save the frame
                    timestamp = int(current_time)
                    image_filename = f"person_detected_{timestamp}.jpg"
                    image_path = os.path.join(captured_frames_dir, image_filename)
                    cv2.imwrite(image_path, frame_with_detection)
                    logger.info(f"Person detected! Image saved to: {image_path}")
            else:
                person_detected_count = 0
                
                # End the current interaction if no person is detected
                if current_interaction:
                    interaction_end_time = datetime.now()
                    interaction_duration = (interaction_end_time - interaction_start_time).total_seconds()
                    logger.info(f"Interaction ended. Duration: {interaction_duration:.2f} seconds. Frames captured: {len(current_interaction)}")
                    
                    # Here you could add logic to save the interaction as a group or create a GIF
                    # For now, we'll just clear the current interaction
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
