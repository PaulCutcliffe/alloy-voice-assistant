import base64
from threading import Lock, Thread
import time
from datetime import datetime, timedelta
import locale
from collections import deque
import os

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
    def __init__(self, buffer_size=10):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()
        self.previous_frame = None
        self.frame_buffer = deque(maxlen=buffer_size)

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
        
        self.frame_buffer.append(self.read())
        
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

    def get_frame_bundle(self):
        return list(self.frame_buffer)

class EnhancedMotionCommentaryAssistant:
    def __init__(self, model, time_interval=5):
        self.model = model
        self.chain = self._create_inference_chain()
        self.last_commentary_time = datetime.now()
        self.last_time_mention = datetime.now() - timedelta(minutes=time_interval)
        self.time_interval = timedelta(minutes=time_interval)
        self.session_start = True

    def generate_commentary(self, frame_bundle):
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

        encoded_frames = [base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode() for frame in frame_bundle]
        prompt = f"{intro}{time_mention}Analyse the sequence of images and describe any changes, movements or activities you observe."
        
        try:
            response = self.chain.invoke(
                {"prompt": prompt, "images": encoded_frames},
                config={"configurable": {"session_id": "unused"}},
            )
            if isinstance(response, str):
                response = response.strip()
            else:
                response = str(response)
        except Exception as e:
            print(f"Error generating commentary: {e}")
            response = "Unable to generate commentary at this time."

        full_response = f"{intro}{time_mention}{response}"
        print("Commentary:", full_response)
        self._tts(full_response)
        self.last_commentary_time = current_time

    def _tts(self, response):
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

    def _create_inference_chain(self):
        SYSTEM_PROMPT = """
        You are an observant AI assistant tasked with providing commentary 
        of a CCTV or webcam feed when motion is detected. You will be given a 
        sequence of images to analyze. Describe what you see, focusing on:
        
        1. The nature and extent of the detected movement or activity.
        2. How the scene changes across the image sequence.
        3. Notable objects, people, or activities involved.
        4. Any potential security concerns or unusual activities.

        Keep your descriptions concise but informative. Don't use emoticons or emojis.
        Avoid speculation and stick to what you can actually observe.
        The date and time will be provided to you when necessary. Do not generate
        or mention timestamps yourself.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        *[{"type": "image_url", "image_url": f"data:image/jpeg;base64,{{image}}"} for _ in range(10)],
                    ],
                ),
            ]
        )

        chain = prompt_template | self.model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

webcam_stream = WebcamStream(buffer_size=10).start()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
# Uncomment the following line to use OpenAI's GPT-4 instead:
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))

assistant = EnhancedMotionCommentaryAssistant(model, time_interval=5)

try:
    while True:
        frame = webcam_stream.read()
        cv2.imshow("CCTV Feed", frame)

        if webcam_stream.detect_motion():
            print("Motion detected!")
            frame_bundle = webcam_stream.get_frame_bundle()
            assistant.generate_commentary(frame_bundle)

        if cv2.waitKey(1) in [27, ord("q")]:
            break
finally:
    webcam_stream.stop()
    cv2.destroyAllWindows()