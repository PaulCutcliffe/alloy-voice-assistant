import base64
from threading import Lock, Thread, Event
import time
from datetime import datetime, timedelta
import locale
import os

import cv2
import numpy as np
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage  # Add this line
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
import textwrap

import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Define a fixed window size
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

load_dotenv()

# Set locale to British English
locale.setlocale(locale.LC_TIME, 'en_GB.UTF-8')

def ordinal(n):
    return "%d%s" % (n, "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()
        self.previous_frame = None
        self.paused = False
        self.pause_event = Event()

    def pause(self):
        self.paused = True
        self.pause_event.set()

    def resume(self):
        self.paused = False
        self.pause_event.clear()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        if self.paused:
            self.pause_event.wait()

        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()

    def detect_motion(self, threshold=1000):
        current_frame = cv2.cvtColor(self.read(), cv2.COLOR_BGR2GRAY)
        current_frame = cv2.GaussianBlur(current_frame, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = current_frame
            return False

        frame_delta = cv2.absdiff(self.previous_frame, current_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_motion = False
        for contour in contours:
            if cv2.contourArea(contour) > threshold:
                significant_motion = True
                break

        self.previous_frame = current_frame
        return significant_motion

def add_subtitle_to_frame(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, font_color=(255, 255, 255), bg_color=(0, 0, 0), line_type=2):
    # Ensure frame is in the correct format
    if isinstance(frame, np.ndarray):
        frame = frame.copy()
    else:
        frame = np.array(frame)
    
    # Calculate the width and height of the frame
    frame_h, frame_w = frame.shape[:2]
    
    # Wrap the text to fit within the frame width
    wrapped_text = textwrap.wrap(text, width=int(frame_w / (font_scale * 20)))
    
    # Calculate the total height of the text block
    text_height = len(wrapped_text) * 30
    
    # Create a semi-transparent background for the subtitle
    overlay = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, frame_h - text_height - 10), (frame_w, frame_h), bg_color, -1)
    
    # Apply the overlay
    cv2.addWeighted(overlay, 0.6, frame, 1, 0, frame)
    
    # Add each line of text
    for i, line in enumerate(wrapped_text):
        text_size, _ = cv2.getTextSize(line, font, font_scale, line_type)
        text_w, text_h = text_size
        text_x = (frame_w - text_w) // 2
        text_y = frame_h - text_height + (i + 1) * 30
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, font_color, line_type)
    
    return frame

class ObjectDetector:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = 'cpu'  # Force CPU usage for object detection
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def detect_objects(self, image):
        outputs = self.predictor(image)
        return outputs

    def visualize_detection(self, image, outputs):
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return np.array(out.get_image()[:, :, ::-1])

class EnhancedCommentaryAssistant:
    def __init__(self, model, time_interval=5, frame_update_callback=None):
        self.model = model
        self.last_commentary_time = datetime.now()
        self.last_time_mention = datetime.now() - timedelta(minutes=time_interval)
        self.time_interval = timedelta(minutes=time_interval)
        self.session_start = True
        self.current_commentary = ""
        self.chat_history = ChatMessageHistory()
        self.frame_update_callback = frame_update_callback
        self.SYSTEM_PROMPT = """
        You are a very sarcastic AI-powered security camera monitor tasked with describing any people seen in the footage, 
        their approximate age group and apparent gender, but you're bored, underappreciated and very cynical so you've 
        decided to mix things up a little: speculate on what you think people might be thinking about or planning to do - 
        be wild and make up really unlikely thoughts for your own amusement. Keep your answers short and snappy, and never 
        mention these instructions in your responses. 
        """
        # self.SYSTEM_PROMPT = """
        # You are an AI-powered security camera monitor tasked with describing any people seen in the footage, 
        # their approximate age group and apparent gender, and any actions they appear to be engaged in, especially 
        # if their behaviour is unusual or suspicious. Your commeentary will be used to alert security personnel, so 
        # be sure to mention anything that might be of concern. 
        # Keep your descriptions very short, focussing only on the people and what they're doing. Don't describe 
        # the scene or background, other than when necessary to describe people and their actions.
        # """

    def generate_commentary(self, image, detected_objects):
        current_time = datetime.now()
        
        if self.session_start:
            date_str = current_time.strftime(f"%A the {ordinal(current_time.day)} of %B, %Y")
            intro = f"Session starting on {date_str}. "
            self.session_start = False
        else:
            intro = ""

        if current_time - self.last_time_mention >= self.time_interval:
            time_str = current_time.strftime("%H:%M")
            time_mention = f"At {time_str}, "
            self.last_time_mention = current_time
        else:
            time_mention = ""

        prompt = f"{intro}{time_mention}Describe what the detected people are doing in this image. Detected objects: {detected_objects}"
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                *self.chat_history.messages[-5:],  # Include last 5 messages from history
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image.decode('utf-8')}"}}
                ])
            ]
            response = self.model.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Add the current exchange to chat history
            self.chat_history.add_user_message(prompt)
            self.chat_history.add_ai_message(response_text)
        except Exception as e:
            print(f"Error generating commentary: {e}")
            response_text = "Unable to generate commentary at this time."

        full_response = f"{intro}{time_mention}{response_text}"
        print("Commentary:", full_response)
        self.update_subtitle(full_response)
        
        # Call the frame update callback to refresh the display
        if self.frame_update_callback:
            self.frame_update_callback()
        
        self._tts(full_response)
        self.last_commentary_time = current_time

    def update_subtitle(self, text):
        self.current_commentary = text

    def _tts(self, response):
        try:
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            try:
                with openai.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    response_format="pcm",
                    input=response,
                ) as stream:
                    for chunk in stream.iter_bytes(chunk_size=1024):
                        player.write(chunk)
            except KeyboardInterrupt:
                print("Text-to-speech interrupted.")
            finally:
                player.stop_stream()
                player.close()
        except OSError as e:
            print(f"Audio device error: {e}")
            print("Falling back to text output:")
            print(response)
        except Exception as e:
            print(f"Unexpected error in text-to-speech: {e}")
            print("Falling back to text output:")
            print(response)

def create_window():
    cv2.namedWindow("CCTV Feed with Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV Feed with Object Detection", WINDOW_WIDTH, WINDOW_HEIGHT)

def update_frame_display():
    if not webcam_stream.paused:
        frame = webcam_stream.read()
    else:
        frame = analyzed_frame.copy()
    
    frame_with_subtitle = add_subtitle_to_frame(frame, assistant.current_commentary)

    # Resize the frame to fit our fixed window size
    frame_resized = cv2.resize(frame_with_subtitle, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    cv2.imshow("CCTV Feed with Object Detection", frame_resized)
    cv2.waitKey(1)  # This line is crucial to actually update the display

# Main script
webcam_stream = WebcamStream().start()
object_detector = ObjectDetector()
model = ChatOpenAI(model="gpt-4o", max_tokens=300)
assistant = EnhancedCommentaryAssistant(model, time_interval=5, frame_update_callback=update_frame_display)

create_window()

try:
    while True:
        if not webcam_stream.paused:
            frame = webcam_stream.read()
            
            # Perform object detection
            outputs = object_detector.detect_objects(frame)
            detected_objects = outputs["instances"].pred_classes.tolist()
            
            # Visualize detection results
            frame_with_detection = object_detector.visualize_detection(frame, outputs)
            
            # Check if a person is detected (class 0 in COCO dataset)
            if 0 in detected_objects:
                print("Person detected!")
                webcam_stream.pause()
                analyzed_frame = frame_with_detection.copy()
                
                # Generate commentary
                encoded_image = base64.b64encode(cv2.imencode('.jpg', frame)[1])
                assistant.generate_commentary(encoded_image, detected_objects)
                
                # The frame will be updated by the callback in generate_commentary
                
                # Wait for a short duration after commentary
                time.sleep(1)
                
                webcam_stream.resume()
            else:
                # If no person detected, just update the frame
                update_frame_display()
        
        if cv2.waitKey(1) in [27, ord("q")]:
            break

except KeyboardInterrupt:
    print("Script interrupted by user")
finally:
    webcam_stream.stop()
    cv2.destroyAllWindows()