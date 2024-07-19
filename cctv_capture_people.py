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

# Define the captured_frames directory
captured_frames_dir = "captured_frames"
os.makedirs(captured_frames_dir, exist_ok=True)

# Constants
FRAME_SKIP = 5  # Process every 5th frame
PERSON_PERSISTENCE = 3  # Require person to be in 3 consecutive frames
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960

class WebcamStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.running = False
        self.frame = None

    def start(self):
        self.running = True
        self.frame = self.read()
        return self

    def read(self):
        ret, frame = self.stream.read()
        return frame if ret else None

    def stop(self):
        self.running = False
        self.stream.release()

class ObjectDetector:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def detect_objects(self, image):
        return self.predictor(image)

    def draw_detection_borders(self, image, outputs):
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

def create_window():
    cv2.namedWindow("CCTV Feed with Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV Feed with Object Detection", WINDOW_WIDTH, WINDOW_HEIGHT)

def main():
    webcam_stream = WebcamStream().start()
    object_detector = ObjectDetector()
    create_window()

    frame_count = 0
    person_detected_count = 0
    current_interaction = []
    interaction_start_time = None

    try:
        while webcam_stream.running:
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            frame = webcam_stream.read()
            if frame is None:
                continue

            outputs = object_detector.detect_objects(frame)
            detected_objects = outputs["instances"].pred_classes.tolist()

            if 0 in detected_objects:  # 0 is the class ID for person in COCO dataset
                person_detected_count += 1
                if person_detected_count >= PERSON_PERSISTENCE:
                    frame_with_detection = object_detector.draw_detection_borders(frame, outputs)
                    
                    # Start a new interaction if not already started
                    if not interaction_start_time:
                        interaction_start_time = datetime.now()
                    
                    # Add frame to current interaction
                    current_interaction.append(frame_with_detection)
                    
                    # Save the frame
                    timestamp = int(time.time())
                    image_filename = f"person_detected_{timestamp}.jpg"
                    image_path = os.path.join(captured_frames_dir, image_filename)
                    cv2.imwrite(image_path, frame_with_detection)
                    print(f"Person detected! Image saved to: {image_path}")
            else:
                person_detected_count = 0
                
                # End the current interaction if no person is detected
                if current_interaction:
                    interaction_end_time = datetime.now()
                    interaction_duration = (interaction_end_time - interaction_start_time).total_seconds()
                    print(f"Interaction ended. Duration: {interaction_duration:.2f} seconds. Frames captured: {len(current_interaction)}")
                    
                    # Here you could add logic to save the interaction as a group or create a GIF
                    # For now, we'll just clear the current interaction
                    current_interaction = []
                    interaction_start_time = None

            # Display the frame
            display_frame = frame_with_detection if 'frame_with_detection' in locals() else frame
            cv2.imshow("CCTV Feed with Object Detection", display_frame)

            if cv2.waitKey(1) in [27, ord("q")]:
                break

    except KeyboardInterrupt:
        print("Script interrupted by user")
    finally:
        webcam_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
