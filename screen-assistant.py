import base64
from threading import Lock, Thread
import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError
import pyautogui
import numpy as np
import cv2
from threading import Lock, Thread
import base64

load_dotenv()

class ScreenCapture:
    def __init__(self, monitor_number=0):
        self.running = False
        self.lock = Lock()
        self.frame = None
        self.monitors = pyautogui.size()
        self.monitor_number = monitor_number
        self.capture_region = self._get_capture_region()

    def _get_capture_region(self):
        if self.monitor_number == 0:
            return (0, 0, self.monitors[0], self.monitors[1])
        else:
            # For multiple monitors, you'd need to implement logic to determine
            # the correct region for each monitor. This is a placeholder.
            return (0, 0, self.monitors[0], self.monitors[1])

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screenshot = pyautogui.screenshot(region=self.capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy() if self.frame is not None else None
        self.lock.release()

        if encode and frame is not None:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will view the user's screen and provide a constant commentary of their activities.

        Use few words and get straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


screen_capture = ScreenCapture(monitor_number=0).start()

# You can use Gemini Flash model by uncommenting the following line:
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model by uncommenting the following line:
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))

# You can use Claude 3.5 Sonnet by uncommenting the following line:
# model = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL"), anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

# You can use Claude 3 Opus by uncommenting the following line:
# model = ChatAnthropic(model="claude-3-opus-20240229")

# Choose one of the above models and uncomment it

assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, screen_capture.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")

recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("screen", screen_capture.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

screen_capture.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
