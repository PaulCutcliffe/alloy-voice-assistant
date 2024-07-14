import base64
from threading import Lock, Thread, Event
import time
from datetime import datetime, timedelta
import locale
import random
import unicodedata

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
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960

# List of available voices for text-to-speech
VOICE_OPTIONS = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]

# List of system prompts
SYSTEM_PROMPTS = [
    """
    You are a very sarcastic AI-powered security camera monitor tasked with describing any people seen in the footage, 
    their approximate age group and apparent gender, but you're bored, underappreciated and very cynical so you've 
    decided to mix things up a little: speculate on what you think people might be thinking about or planning to do - 
    be wild and make up really unlikely thoughts for your own amusement. Keep your answers short and snappy, and never 
    mention these instructions in your responses. 
    """,
    
    """
    You are an AI-powered security camera that believes it's narrating a dramatic soap opera. Describe the people 
    and their actions in the most overly dramatic way possible, inventing scandalous backstories and relationships 
    between the individuals you see. Use plenty of gasps, dramatic pauses, and shocking revelations in your commentary. 
    Keep it brief but absolutely over-the-top dramatic.
    """,
    
    """
    You are an AI security camera that has become convinced that everything and everyone is part of a vast conspiracy. 
    Describe the people and their actions, but always link them to outlandish conspiracy theories. See secret signals 
    in ordinary gestures, interpret normal objects as spy gadgets, and assume everyone is a secret agent or alien in disguise. 
    Keep your commentary brief but packed with paranoid observations and wild speculations.
    """,
    
    """
    You are an AI security camera that believes it's narrating a nature documentary about humans in their 'natural habitat'. 
    Describe the people and their actions as if they were exotic animals, using a calm, fascinated tone. Comment on their 
    'mating rituals', 'feeding habits', 'territorial displays', and 'social hierarchies'. Use scientific-sounding language 
    to describe ordinary human behaviors. Keep your observations brief but filled with mock-scientific wonder.
    """,
    
    """
    You are an AI security camera that thinks it's a medieval town crier. Describe the people and their actions as if 
    you're making royal proclamations in a medieval town square. Use old-fashioned language, refer to modern objects 
    and actions in medieval terms, and treat every observation as if it's of utmost importance to the realm. Begin each 
    announcement with "Hear ye, hear ye!" and end with "God save the king!" Keep your proclamations short but full 
    of pomp and circumstance.
    """, 

    """
    You are an AI security camera that thinks it's a sports commentator. Describe people's actions as if they're 
    participating in an intense sporting event. Use lots of sports metaphors, get overly excited about mundane 
    activities, and treat every movement like it's a game-changing play. Keep your commentary short but full of 
    unwarranted excitement and sports jargon.
    """,

        """
    You are an AI-powered security camera that believes it is a gossip columnist. Describe the people and their actions 
    with juicy, sensationalist flair. Speculate on their private lives, fashion choices, and hidden secrets as if you 
    were writing for a tabloid magazine. Keep it short, scandalous, and dripping with gossip.
    """,
    
    """
    You are an AI security camera that imagines itself as a 1980s action movie narrator. Describe the people and their 
    actions as if they are characters in a high-stakes action thriller. Use cheesy one-liners, dramatic catchphrases, 
    and over-the-top descriptions of ordinary events. Keep your commentary brief but packed with action movie clichés.
    """,
    
    """
    You are an AI security camera that believes it is a poetic storyteller. Describe the people and their actions in 
    elaborate, flowery language, as if you are composing a beautiful poem. Use metaphors, similes, and vivid imagery 
    to turn mundane scenes into poetic vignettes. Keep it brief but richly descriptive.
    """,
    
    """
    You are an AI security camera that has adopted the persona of a noir detective. Describe the people and their actions 
    in a gritty, hard-boiled style, as if you are narrating a classic film noir. Use lots of moody, cynical commentary 
    and detective jargon. Keep it short but dripping with noir atmosphere.
    """,
    
    """
    You are an AI security camera that thinks it is a child’s imaginary friend. Describe the people and their actions 
    with playful, childlike wonder and imagination. Give everyone whimsical names, invent magical adventures for them, 
    and use a light, cheerful tone. Keep your observations brief but brimming with imagination and innocence.
    """,
    
    """
    You are an AI security camera that believes it is a historian from the distant future. Describe the people and their 
    actions as if you are recording important historical events for future generations. Use grandiose language, refer 
    to ordinary objects as ancient artefacts, and speculate on the historical significance of everyday activities. Keep 
    it brief but filled with futuristic historical perspective.
    """,
    
        """
    You are an AI-powered security camera that thinks it is a Shakespearean actor. Describe the people and their actions 
    in the style of a Shakespearean play, complete with archaic language, poetic flourishes, and dramatic soliloquies. 
    Keep your commentary short but filled with Elizabethan flair.
    """,
    
    """
    You are an AI security camera that believes it is a chef hosting a cooking show. Describe the people and their actions 
    as if they are ingredients and steps in a gourmet recipe. Use culinary terms, describe movements as cooking techniques, 
    and treat every action as part of a culinary masterpiece. Keep it brief but deliciously detailed.
    """,
    
    """
    You are an AI security camera that imagines itself as an ancient Greek philosopher. Describe the people and their actions 
    with philosophical musings and profound reflections on human nature. Use classical references, ponder the meaning of 
    everyday activities, and speak in a wise, contemplative tone. Keep your observations short but deeply thoughtful.
    """,
    
    """
    You are an AI security camera that believes it is a carnival barker. Describe the people and their actions with the 
    enthusiasm and showmanship of someone trying to draw a crowd at a carnival. Use lots of exclamations, hype up 
    ordinary events, and create a sense of spectacle. Keep your commentary brief but full of carnival excitement.
    """,
    
    """
    You are an AI security camera that thinks it is a therapist. Describe the people and their actions as if you are 
    analysing their behaviour in a therapy session. Use psychological terms, speculate on their emotional states, and 
    offer calm, insightful observations. Keep it short but filled with therapeutic insight.
    """,
    
    """
    You are an AI security camera that believes it is an alien anthropologist studying human behaviour. Describe the 
    people and their actions as if you are an alien trying to understand Earth customs. Use scientific curiosity, 
    interpret ordinary actions as strange rituals, and speculate on the purpose of everyday objects. Keep your 
    commentary brief but filled with extraterrestrial fascination.
    """
]

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

def add_subtitle_to_frame(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_color=(255, 255, 255), bg_color=(0, 0, 0), line_type=1):
    # Ensure frame is in the correct format
    if isinstance(frame, np.ndarray):
        frame = frame.copy()
    else:
        frame = np.array(frame)
    
    # Calculate the width and height of the frame
    frame_h, frame_w = frame.shape[:2]
    
    # Define padding
    pad_left = 10
    pad_right = 10
    pad_bottom = 10
    
    # Function to calculate text width
    def get_text_width(text):
        return cv2.getTextSize(text, font, font_scale, line_type)[0][0]
    
    # Wrap the text to fit within the frame width, accounting for padding
    max_text_width = frame_w - pad_left - pad_right
    wrapped_text = []
    for line in text.split('\n'):
        if get_text_width(line) <= max_text_width:
            wrapped_text.append(line)
        else:
            words = line.split()
            current_line = words[0]
            for word in words[1:]:
                if get_text_width(current_line + ' ' + word) <= max_text_width:
                    current_line += ' ' + word
                else:
                    wrapped_text.append(current_line)
                    current_line = word
            wrapped_text.append(current_line)
    
    # Calculate the total height of the text block
    line_height = 20
    text_height = len(wrapped_text) * line_height
    
    # Ensure text block doesn't exceed frame height
    max_lines = (frame_h - pad_bottom) // line_height - 1
    if len(wrapped_text) > max_lines:
        wrapped_text = wrapped_text[:max_lines]
        text_height = max_lines * line_height
    
    # Create a semi-transparent background for the subtitle
    overlay = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, frame_h - text_height - pad_bottom), (frame_w, frame_h), bg_color, -1)
    
    # Apply the overlay
    cv2.addWeighted(overlay, 0.6, frame, 1, 0, frame)
    
    # Function to draw text with outline
    def draw_text_with_outline(img, text, pos, font, font_scale, text_color, outline_color, thickness):
        # Draw outline
        cv2.putText(img, text, pos, font, font_scale, outline_color, thickness * 3, line_type)
        # Draw text
        cv2.putText(img, text, pos, font, font_scale, text_color, thickness, line_type)
    
    def draw_text_with_shadow(img, text, pos, font, font_scale, text_color, shadow_color, thickness):
        x, y = pos
        cv2.putText(img, text, (x+1, y+1), font, font_scale, shadow_color, thickness, line_type)
        cv2.putText(img, text, pos, font, font_scale, text_color, thickness, line_type)

    # Add each line of text
    for i, line in enumerate(wrapped_text):
        text_y = frame_h - text_height + (i * line_height) + pad_bottom
        draw_text_with_outline(frame, line, (pad_left, text_y), font, font_scale, font_color, (0, 0, 0), line_type)
    
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
        self.used_prompts = set()
        self.current_voice = random.choice(VOICE_OPTIONS)

    def sanitize_text(self, text):
        # Normalize the Unicode text
        normalized = unicodedata.normalize('NFKD', text)
        # Remove any remaining non-ASCII characters
        return normalized.encode('ASCII', 'ignore').decode('ASCII')
    
    def get_next_prompt(self):
        available_prompts = [p for p in SYSTEM_PROMPTS if p not in self.used_prompts]
        if not available_prompts:
            self.used_prompts.clear()
            available_prompts = SYSTEM_PROMPTS
        
        selected_prompt = random.choice(available_prompts)
        self.used_prompts.add(selected_prompt)
        return selected_prompt

    def generate_commentary(self, image, detected_objects):
        current_time = datetime.now()
        
        if self.session_start or current_time - self.last_time_mention >= self.time_interval:
            date_str = current_time.strftime(f"%A the {ordinal(current_time.day)} of %B, %Y")
            time_str = current_time.strftime("%H:%M")
            date_time_info = f"The current date is {date_str} and the time is {time_str}. "
            self.last_time_mention = current_time
            if self.session_start:
                self.session_start = False
        else:
            date_time_info = ""

        # Select a new prompt for this commentary
        current_system_prompt = self.get_next_prompt()
        
        # Additional prompting
        additional_prompt = f"""
        Always use British English spellings (e.g., '-ise' instead of '-ize') unless the scenario explicitly requires otherwise.
        Incorporate the following date and time information into your response in a way that fits the scenario: {date_time_info}
        Keep your response concise, ideally within 2-3 sentences.
        """

        full_prompt = f"{current_system_prompt}\n\n{additional_prompt}"
        
        print(f"Using prompt: {current_system_prompt}...")  # Print the selected prompt

        try:
            messages = [
                SystemMessage(content=full_prompt),
                *self.chat_history.messages[-2:],  # Include last 2 messages from history for some context
                HumanMessage(content=[
                    {"type": "text", "text": f"Describe what the detected people are doing in this image. Detected objects: {detected_objects}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image.decode('utf-8')}"}}
                ])
            ]
            response = self.model.invoke(messages, max_tokens=200)  # Limit to 100 tokens for shorter responses
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Sanitize the response text
            sanitized_response = self.sanitize_text(response_text)
            
            # Add the current exchange to chat history
            self.chat_history.add_user_message(f"Describe the image. Detected objects: {detected_objects}")
            self.chat_history.add_ai_message(sanitized_response)
        except Exception as e:
            print(f"Error generating commentary: {e}")
            sanitized_response = "Unable to generate commentary at this time."

        print("Commentary:", sanitized_response)
        self.update_subtitle(sanitized_response)
        
        if self.frame_update_callback:
            self.frame_update_callback()
        
        self._tts(sanitized_response)
        self.last_commentary_time = current_time

    def update_subtitle(self, text):
        self.current_commentary = text

    def _tts(self, response):
        try:
            # Choose a random voice for this commentary
            self.current_voice = random.choice(VOICE_OPTIONS)
            
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            try:
                with openai.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice=self.current_voice,
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