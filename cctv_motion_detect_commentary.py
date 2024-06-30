import base64
from threading import Lock, Thread
import time
from datetime import datetime, timedelta
import locale

import cv2
import numpy as np
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16

load_dotenv()

# Set locale to British English
locale.setlocale(locale.LC_TIME, 'en_GB.UTF-8')

class WebcamStream:
    # ... [Keep the WebcamStream class as is] ...

class TimestampedMotionCommentaryAssistant:
    def __init__(self, model, time_interval=5):
        self.chain = self._create_inference_chain(model)
        self.last_commentary_time = datetime.now()
        self.last_time_mention = datetime.now() - timedelta(minutes=time_interval)
        self.time_interval = timedelta(minutes=time_interval)
        self.session_start = True

    def generate_commentary(self, image):
        current_time = datetime.now()
        
        if self.session_start:
            date_str = current_time.strftime("%A the %d of %B, %Y")
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

        prompt = f"{intro}{time_mention}Describe what you see in this image, focusing on any changes or movements."
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        full_response = f"{intro}{time_mention}{response}"
        print("Commentary:", full_response)
        self._tts(full_response)
        self.last_commentary_time = current_time

    def _tts(self, response):
        # ... [Keep the _tts method as is] ...

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are an observant AI assistant tasked with providing commentary 
        of a CCTV or webcam feed when motion is detected. Describe what you see 
        in each image, focusing on:
        
        1. The nature and extent of the detected movement.
        2. Notable objects, people, or activities involved in the motion.
        3. Any potential security concerns or unusual activities.

        Keep your descriptions concise and to the point. Don't use emoticons or emojis.
        Avoid speculation and stick to what you can actually observe.
        The date and time will be provided to you when necessary. Do not generate
        or mention timestamps yourself.
        """

        # ... [Rest of the method remains the same] ...

webcam_stream = WebcamStream().start()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
# Uncomment the following line to use OpenAI's GPT-4 instead:
# model = ChatOpenAI(model="gpt-4-vision-preview")

assistant = TimestampedMotionCommentaryAssistant(model, time_interval=5)

try:
    while True:
        frame = webcam_stream.read()
        cv2.imshow("CCTV Feed", frame)

        if webcam_stream.detect_motion():
            print("Motion detected!")
            encoded_image = webcam_stream.read(encode=True)
            assistant.generate_commentary(encoded_image)

        if cv2.waitKey(1) in [27, ord("q")]:
            break
finally:
    webcam_stream.stop()
    cv2.destroyAllWindows()