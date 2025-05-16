import cv2
import numpy as np
import requests
from keras.models import model_from_json
import sys
import os
import time
import datetime
import joblib
import threading
import cohere
import pyttsx3
import subprocess
import platform
import speech_recognition as sr
import webbrowser
import urllib.parse
import pyautogui
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QLineEdit, QPushButton, QMenu, QAction,QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5 import QtCore
import pyautogui
import pytesseract
import cv2
import numpy as np
from PIL import Image
import asyncio
import edge_tts
from playsound import playsound
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import winsound
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

WEATHER_API_KEY= os.getenv("WEATHER_API_KEY")  

# === Load Emotion Detection Model ===
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")


# === Load Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# Common phrases for time queries
TIME_QUERIES = {
    "what is the time", "what time is it", "time", "current time", "time please",
    "time?", "time now", "time batao", "time kya hua", "time kya"
}

def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

# Load the trained intent model
intent_model = joblib.load("intent_model_LR.pkl")
embed_model = joblib.load("embedding_model.pkl")

AVATAR_VOICE_MAP = {
    "Japesh": 0,    
    "Samiksha": 1, 
    "Samdisha": 2,
    #"Akshina": 3, 
}

AVATAR_CLOUD_VOICE_MAP = {
    "Japesh": "en-US-GuyNeural",
    "Samiksha": "en-IN-NeerjaNeural",
    "Samdisha": "en-US-JennyNeural",
    "Akshina": "en-GB-SoniaNeural",
    "Sahil": "en-IN-PrabhatNeural",
    "Simar": "en-US-AndrewNeural",
    "Sirjan": "en-US-AnaNeural",
}

prompt_map = {
    "sad": "You are an emotional support assistant. The user is feeling sad. Ask gently why they are feeling this way and try to cheer them up with comforting words or a light-hearted joke.",
    "happy": "The user is happy. Celebrate their joy and tell a couple of fun or uplifting jokes to keep the mood high.",
    "angry": "The user looks angry. Ask them what happened, validate their feelings, and suggest ways to cool down calmly.",
    "fear": "The user is feeling afraid. Ask them what is making them anxious or scared, and reassure them with comforting advice.",
    "disgust": "The user looks disgusted. Ask what triggered that emotion and offer a light or humorous distraction.",
    "surprise": "The user is surprised. Ask what surprised them and show excitement or curiosity.",
    "neutral": "The user's expression is neutral. Gently ask how their day is going or offer something positive to brighten their moment."
}

NIRCMD_PATH = "D:\\Projects\\Desktopmate\\ni\\nircmd.exe"#D:\Projects\Desktopmate\ni\nircmd.exe

class DesktopMate(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.avatar_path = "assets/idle/Japesh"
        avatar_name = os.path.basename(self.avatar_path).lower()
        self.idle_frames = self.load_frames(self.avatar_path)
        if not self.idle_frames:
            raise FileNotFoundError(f"No PNG images found in {self.avatar_path}")

        self.current_frame = 0
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.is_paused = False
        self.speech_queue = []

        self.label = QLabel(self)
        self.label.setPixmap(self.idle_frames[0])
        self.label.move(0, 0)

        avatar_height = self.idle_frames[0].height()
        self.resize(self.idle_frames[0].width(), avatar_height + 80)

        self.response_label = QLabel(self)
        self.response_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150); padding: 5px; border-radius: 10px;")
        self.response_label.setWordWrap(True)
        self.response_label.setFixedWidth(200)
        self.response_label.move(20, avatar_height - 60)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Ask me something...")
        self.input_box.returnPressed.connect(self.ask_cohere_from_text)
        self.input_box.setFixedWidth(160)
        self.input_box.move(10, avatar_height + 10)

        self.mic_button = QPushButton("🎙", self)
        self.mic_button.setFixedSize(30, 30)
        self.mic_button.move(180, avatar_height + 10)
        self.mic_button.clicked.connect(self.listen_to_voice)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setFixedSize(50, 30)
        self.pause_button.move(220, avatar_height + 10)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(150)

        self.offset = QPoint()
        if "japesh" in avatar_name:
            self.wake_word = "sevak"
        elif "samiksha" in avatar_name:
            self.wake_word = "sakhi"
        else:
            self.wake_word = "assistant"  # Fallback wake word
        #self.wake_word = "sevak"
        threading.Thread(target=self.wake_word_listener, daemon=True).start()

        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)  # Open the default camera
        self.emotion_detected = False  # Flag to track if emotion has been detected
        self.last_emotion = None  # Store the last detected emotion

        # Start capturing frames in a separate thread
        threading.Thread(target=self.capture_frames, daemon=True).start()


    def capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.detect_emotion_from_frame(frame)
            else:
                print("Failed to capture frame.")

    def detect_emotion_from_frame(self, frame):
        """ret, frame = self.cap.read()
        if not ret:
            return"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if not self.emotion_detected and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            features = extract_features(face)
            prediction = model.predict(features)
            emotion = labels[np.argmax(prediction)]
            self.last_emotion = emotion
            self.emotion_detected = True
            print(f"Detected emotion: {emotion}")
            # Map the detected emotion to the corresponding prompt
            prompt = prompt_map.get(emotion, "The user is feeling something. Ask them how they feel.")
            print(f"Mapped prompt: {prompt}")
            # Pass the mapped prompt to the ask_cohere method
            self.ask_cohere(prompt)

            # Draw face box
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, self.last_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def load_frames(self, folder):
        files = sorted(f for f in os.listdir(folder) if f.endswith(".png") and not f.startswith("."))
        return [QPixmap(os.path.join(folder, f)) for f in files]

    
    def update_animation(self):
        if self.idle_frames:
            self.label.setPixmap(self.idle_frames[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.idle_frames)

    def set_voice_for_avatar(self, avatar_name):
        voices = self.engine.getProperty('voices')
        voice_index = AVATAR_VOICE_MAP.get(avatar_name.capitalize())

        if voice_index is not None and voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            print(f"[Voice - pyttsx3] Set for {avatar_name}: {voices[voice_index].name}")
        else:
            print(f"[Voice - pyttsx3] Voice index not found or invalid for avatar '{avatar_name}'.")

        """voices = self.engine.getProperty('voices')
        if 'samiksha' in avatar_name:
            # Attempt to set a female voice
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"[Voice] Female voice set: {voice.name}")
                    break
        elif 'japesh' in avatar_name:
            # Attempt to set a male voice
            for voice in voices:
                if "male" in voice.name.lower() or "david" in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"[Voice] Male voice set: {voice.name}")
                    break
                    ``````
        voices = self.engine.getProperty('voices')
        voice_index = AVATAR_VOICE_MAP.get(avatar_name.capitalize())

        if voice_index is not None and voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            print(f"[Voice] Set for {avatar_name}: {voices[voice_index].name}")
        else:
            print(f"[Voice] Voice index not found or invalid for avatar '{avatar_name}'.")
"""

    def set_avatar(self, folder):
        new_frames = self.load_frames(folder)
        avatar_name = os.path.basename(folder).lower()
        if "japesh" in avatar_name:
            self.wake_word = "sevak"
        elif "samiksha" in avatar_name:
            self.wake_word = "sakhi"
        else:
            self.wake_word = "assistant"  # default fallback
        print(f"[Wake Word] Set to: {self.wake_word}")
        if new_frames:
            self.avatar_path = folder
            self.idle_frames = new_frames
            self.current_frame = 0
            pixmap = self.idle_frames[0]
            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.size())

            avatar_height = pixmap.height()
            self.resize(pixmap.width(), avatar_height + 80)

            self.response_label.move(10, avatar_height - 30)
            self.input_box.move(10, avatar_height + 10)
            self.mic_button.move(180, avatar_height + 10)
            self.pause_button.move(220, avatar_height + 10)

            # Change voice based on avatar name
            self.set_voice_for_avatar(os.path.basename(folder).lower())
        else:
            print(f"No PNG images found in {folder}")

    def speak(self, text):
        def run():
            try:
                if self.is_paused:
                    self.speech_queue.append(text)
                    return

                avatar_name = os.path.basename(self.avatar_path).capitalize()
                cloud_voice = AVATAR_CLOUD_VOICE_MAP.get(avatar_name)

                if cloud_voice:
                    try:
                        asyncio.run(self.speak_with_edge_tts(text, cloud_voice))
                        return
                    except Exception as e:
                        print(f"[Edge TTS Fallback]: {e}")

                # Fallback to pyttsx3
                self.engine.say(text)
                self.engine.runAndWait()

            except Exception as e:
                print("[TTS Error]", e)

        threading.Thread(target=run, daemon=True).start()


    async def speak_with_edge_tts(self, text, voice_name):
        output_path = "output.mp3"
        communicate = edge_tts.Communicate(text=text, voice=voice_name)
        await communicate.save(output_path)


        sound = AudioSegment.from_file(output_path, format="mp3")
        play(sound)

    # Get Current Time
    def get_time(self):
        return datetime.now().strftime("%H:%M:%S")
    
    # Get Current Date
    def get_current_date(self):
        return datetime.now().strftime("%Y-%m-%d")
    
    # Get Weather Information
    def get_weather(self,city):
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no"
        try:
            response = requests.get(url)
            data = response.json()
            if response.status_code == 200:
                return f"The current temperature in {city} is {data['current']['temp_c']}°C with {data['current']['condition']['text']}."
            return f"Sorry, unable to fetch weather data for {city}."
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"

    def open_application(self, app_name):
        app_paths = {
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "whatsapp": ["C:\\Program Files\\WhatsApp\\WhatsApp.exe"],
            "camera": ["start", "microsoft.windows.camera:"],
            "settings": ["start", "ms-settings:"],
            "control_panel": ["control"],
            "cmd": ["cmd.exe"],
            "terminal": ["wt.exe"],
            "explorer": ["explorer.exe"],
            "task_manager": ["taskmgr.exe"],
            "device_manager": ["devmgmt.msc"],  
           # "excel": [r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"]   #"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Excel.lnk" 
        }
        if app_name in app_paths:
            subprocess.Popen(app_paths[app_name], shell=True)
            return f"Opening {app_name.capitalize()}..."
        return "Application not found."
    
    def open_website(self, url):
        webbrowser.open(url)
        return f"Opening {url}..."

    def play_youtube_video(self, query):
        search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
        webbrowser.open(search_url)
        return "Opening YouTube for your query..."

    def search_file_in_drive(self, drive, filename):
        search_command = f"where /r {drive}:\\ {filename}"
        os.system(search_command)
        return f"Searching for {filename} in {drive} drive..."

    def extract_time_and_set_alarm(self, command):
        import re
        match = re.search(r"(\d{1,2}:\d{2})", command)
        if match:
            time_str = match.group(1)
            self.set_alarm(time_str)
            return "alarm_set"
        else:
            self.show_response("❌ Please specify time like 'set alarm for 07:30'")
            return "alarm_not_set"


    def set_alarm(self,time_str):
        #time_str, ok = QInputDialog.getText(self, 'Set Alarm', 'Enter time in HH:MM (24-hour format):')
        
            try:
                alarm_time=datetime.strptime(time_str, "%H:%M").strftime("%H:%M")
                thread = threading.Thread(target=self.alarm_checker, args=(time_str,), daemon=True)
                thread.start()
                self.show_response(f"⏰ Alarm set for {time_str}")
            except ValueError:
                self.show_response("❌ Please enter time in 24-hour HH:MM format (e.g., 14:30 for 2:30 PM).")

    def alarm_checker(self, alarm_time):
        while True:
            now = datetime.now().strftime("%H:%M")
            print(f"[Alarm Checker] Current time: {now}, Alarm time: {alarm_time}")  # Debugging line
            if now == alarm_time:
                self.alarm_alert()
                break
            time.sleep(1)

    def alarm_alert(self):
        self.show_response("⏰ Alarm ringing! Time's up!")
        try:
            #playsound("alarm.wav")  # Make sure alarm.wav is in your working directory
            winsound.Beep(1000, 1000)
        except Exception as e:
            print("[Alarm Sound Error]:", e)


    def launch_excel(self):
        possible_paths = [
            r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE",
            r"C:\Program Files\Microsoft Office\Office16\EXCEL.EXE",
            r"C:\Program Files\Microsoft Office\Office15\EXCEL.EXE",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                subprocess.Popen(f'start "" "{path}"', shell=True)
                return "✅ Excel is launching..."
        return "❌ Excel not found in default locations."

    def execute_action(self, intent, user_command):
        actions = {
            "open_whatsapp": lambda: self.open_application("whatsapp"),
            "open_notepad": lambda: self.open_application("notepad"),
            "open_calculator": lambda: self.open_application("calculator"),
            "open_camera": lambda: self.open_application("camera"),
            "open_settings": lambda: self.open_application("settings"),
            "open_control_panel": lambda: self.open_application("control_panel"),
            "open_cmd": lambda: self.open_application("cmd"),
            "open_terminal": lambda: self.open_application("terminal"),
            "open_explorer": lambda: self.open_application("explorer"),
            "open_task_manager": lambda: self.open_application("task_manager"),
            "open_device_manager": lambda: self.open_application("device_manager"),
            #"open_excel": lambda: self.launch_excel(),

            "open_excel": lambda: os.startfile(r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"),
            "open_facebook": lambda: self.open_website("https://www.facebook.com"),
            "open_instagram": lambda: self.open_website("https://www.instagram.com"),
            "open_twitter": lambda: self.open_website("https://www.twitter.com"),
            "open_linkedin": lambda: self.open_website("https://www.linkedin.com"),
            "open_github": lambda: self.open_website("https://www.github.com"),
            "open_mail": lambda: self.open_website("https://mail.google.com"),
            "open_calendar": lambda: self.open_website("https://calendar.google.com"),
            "open_maps": lambda: self.open_website("https://www.google.com/maps"),
            "open_drive": lambda: self.open_website("https://drive.google.com"),
            "open_classroom": lambda: self.open_website("https://classroom.google.com"),
            "open_meet": lambda: self.open_website("https://meet.google.com"),
            "open_youtube": lambda: self.open_website("https://www.youtube.com"),
            "play_youtube": lambda: self.play_youtube_video(user_command),
            "open_news": lambda: self.open_website("https://news.google.com"),
            "shutdown_pc": lambda: os.system("shutdown /s /t 1"),
            "restart_pc": lambda: os.system("shutdown /r /t 1"),
            "sleep_pc": lambda: os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0"),
            "lock_screen": lambda: os.system("rundll32.exe user32.dll,LockWorkStation"),
            "log_out": lambda: os.system("shutdown -l"),
            "bluetooth_on": lambda: os.system("start ms-settings:bluetooth"),
            "bluetooth_off": lambda: os.system("start ms-settings:bluetooth"),
            "wifi_on": lambda: os.system("start ms-settings:network-wifi"),
            "wifi_off": lambda: os.system("start ms-settings:network-wifi"),
            "volume_up": lambda: os.system(f'"{NIRCMD_PATH}" changesysvolume 2000'),
            "volume_down": lambda: os.system(f'"{NIRCMD_PATH}" changesysvolume -2000'),
            "mute": lambda: os.system(f'"{NIRCMD_PATH}" mutesysvolume 1'),
            "unmute": lambda: os.system(f'"{NIRCMD_PATH}" mutesysvolume 0'),
            "brightness_up": lambda: os.system("powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,100)"),
            "brightness_down": lambda: os.system("powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,30)"),
            "search_file_c": lambda: self.search_file_in_drive("C", user_command),
            "search_file_d": lambda: self.search_file_in_drive("D", user_command),
            "airplane_mode": lambda: "⚠ Airplane mode is not implemented yet.",
            "brighten_lights": lambda: "⚠ Brighten lights is not implemented yet.",
            "change_wallpaper": lambda: "⚠ Change wallpaper is not implemented yet.",
            "check_messages": lambda: "⚠ Check messages is not implemented yet.",
            "check_schedule": lambda: "⚠ Check schedule is not implemented yet.",
            "clean_floor": lambda: "⚠ Clean floor is not implemented yet.",
            "clear_notifications": lambda: "⚠ Clear notifications is not implemented yet.",
            "close_garage": lambda: "⚠ Close garage is not implemented yet.",
            "decrease_contrast": lambda: "⚠ Decrease contrast is not implemented yet.",
            "delete_file": lambda: "⚠ Delete file is not implemented yet.",
            "dim_lights": lambda: "⚠ Dim lights is not implemented yet.",
            "dnd_off": lambda: "⚠ DND off is not implemented yet.",
            "dnd_on": lambda: "⚠ DND on is not implemented yet.",
            "edit_document": lambda: "⚠ Edit document is not implemented yet.",
            "fan_off": lambda: "⚠ Fan off is not implemented yet.",
            "fan_on": lambda: "⚠ Fan on is not implemented yet.",
            #"get_date": lambda: "⚠ Date retrieval is not implemented yet.",
            #"get_time": lambda: "⚠ Time retrieval is not implemented yet.",
            "increase_contrast": lambda: "⚠ Increase contrast is not implemented yet.",
            "join_meeting": lambda: "⚠ Join meeting is not implemented yet.",
            "light_off": lambda: "⚠ Light off is not implemented yet.",
            "light_on": lambda: "⚠ Light on is not implemented yet.",
            "lock_door": lambda: "⚠ Lock door is not implemented yet.",
            "make_call": lambda: "⚠ Make call is not implemented yet.",
            "move_file_desktop": lambda: "⚠ Move file to Desktop is not implemented yet.",
            "move_file_documents": lambda: "⚠ Move file to Documents is not implemented yet.",
            "new_document": lambda: "⚠ New document creation is not implemented yet.",
            "next_song": lambda: "⚠ Next song is not implemented yet.",
            "night_light_off": lambda: "⚠ Night light off is not implemented yet.",
            "night_light_on": lambda: "⚠ Night light on is not implemented yet.",
            "open_desktop": lambda: "⚠ Open desktop is not implemented yet.",
            "open_discord": lambda: "⚠ Open Discord is not implemented yet.",
            "open_documents": lambda: "⚠ Open Documents is not implemented yet.",
            "open_downloads": lambda: "⚠ Open Downloads is not implemented yet.",
            "open_file_explorer": lambda: "⚠ Open File Explorer is not implemented yet.",
            "open_gallery": lambda: "⚠ Open gallery is not implemented yet.",
            "open_garage": lambda: "⚠ Open garage is not implemented yet.",
            "open_music": lambda: "⚠ Open Music is not implemented yet.",
            "open_notes": lambda: "⚠ Open notes is not implemented yet.",
            "open_outlook": lambda: "⚠ Open Outlook is not implemented yet.",
            "open_pictures": lambda: "⚠ Open Pictures is not implemented yet.",
            "open_powerpoint": lambda: "⚠ Open PowerPoint is not implemented yet.",
            "open_skype": lambda: "⚠ Open Skype is not implemented yet.",
            "open_slack": lambda: "⚠ Open Slack is not implemented yet.",
            "open_spotify": lambda: "⚠ Open Spotify is not implemented yet.",
            "open_teams": lambda: "⚠ Open Microsoft Teams is not implemented yet.",
            "open_user_settings": lambda: "⚠ Open user settings is not implemented yet.",
            "pause_song": lambda: "⚠ Pause music is not implemented yet.",
            "play_radio": lambda: "⚠ Play radio is not implemented yet.",
            "play_song": lambda: "⚠ Play music is not implemented yet.",
            "previous_song": lambda: "⚠ Previous song is not implemented yet.",
            "print_page": lambda: "⚠ Print page is not implemented yet.",
            "record_video": lambda: "⚠ Record video is not implemented yet.",
            "rename_file": lambda: "⚠ Rename file is not implemented yet.",
            "repeat_song": lambda: "⚠ Repeat song is not implemented yet.",
            "resume_song": lambda: "⚠ Resume song is not implemented yet.",
            "save_file": lambda: "⚠ Save file is not implemented yet.",
            "search_file": lambda: "⚠ File search is not implemented yet.",
            "search_file_desktop": lambda: "⚠ File search on Desktop not implemented yet.",
            "search_file_documents": lambda: "⚠ File search in Documents not implemented yet.",
            "search_file_downloads": lambda: "⚠ File search in Downloads not implemented yet.",
            "search_file_music": lambda: "⚠ File search in Music not implemented yet.",
            "search_file_pictures": lambda: "⚠ File search in Pictures not implemented yet.",
            "search_file_videos": lambda: "⚠ File search in Videos not implemented yet.",
            "send_message": lambda: "⚠ Send message is not implemented yet.",
            "set_alarm": lambda: self.extract_time_and_set_alarm(user_command),
            "set_dark_mode": lambda: "⚠ Dark mode toggle is not implemented yet.",
            "set_light_mode": lambda: "⚠ Light mode toggle is not implemented yet.",
            "set_meeting": lambda: "⚠ Set meeting is not implemented yet.",
            "set_reminder": lambda: "⚠ Set reminder is not implemented yet.",
            "set_temperature": lambda: "⚠ Set thermostat is not implemented yet.",
            "set_timer": lambda: "⚠ Set timer is not implemented yet.",
            "set_todo": lambda: "⚠ Set to-do list is not implemented yet.",
            "shuffle_songs": lambda: "⚠ Shuffle songs is not implemented yet.",
            "start_meeting": lambda: "⚠ Start meeting is not implemented yet.",
            "start_video_call": lambda: "⚠ Start video call is not implemented yet.",
            "stop_song": lambda: "⚠ Stop music is not implemented yet.",
            "switch_user": lambda: "⚠ Switch user is not implemented yet.",
            "take_selfie": lambda: "⚠ Take selfie is not implemented yet.",
            "unlock_door": lambda: "⚠ Unlock door is not implemented yet.",
        }

        if intent in actions:
            try:
                return actions[intent]()
            except Exception as e:
                return f"❌ Failed to execute action: {str(e)}"
        return "⚠️ Intent not recognized."

    def select_thing_on_screen(self):
        self.show_response("Move your mouse over the thing you want to select and wait...")

        def countdown_and_select():
            for i in range(5, 0, -1):
                QtCore.QMetaObject.invokeMethod(
                    self.response_label,
                    "setText",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, f"Selecting in {i}...")
                )
                time.sleep(1)

            x, y = pyautogui.position()
            pyautogui.click(x, y)
            QtCore.QMetaObject.invokeMethod(
                self,
                "show_response",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Selected at ({x}, {y})")
            )

        threading.Thread(target=countdown_and_select, daemon=True).start()

    def change_avatar(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Avatar Folder", os.getcwd())
        if folder:
            new_frames = self.load_frames(folder)
            if new_frames:
                self.avatar_path = folder
                self.idle_frames = new_frames
                self.current_frame = 0

                self.label.setPixmap(self.idle_frames[0])
                self.resize(self.idle_frames[0].width(), self.idle_frames[0].height())
            else:
                print("No PNG images found in the selected folder.")


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.offset)

    pytesseract.pytesseract.tesseract_cmd = r"D:\Projects\Tesseract\tesseract.exe"

    def select_text_on_screen(self,target_text):
        try:
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Use pytesseract to get data with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                word = data['text'][i].strip().lower()
                if target_text.lower() in word:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]

                    # Center of the detected text
                    center_x = x + w // 2
                    center_y = y + h // 2

                    pyautogui.click(center_x, center_y)
                    return f"✅ Clicked on: '{word}' at ({center_x}, {center_y})"

            return f"❌ Could not find '{target_text}' on screen."

        except Exception as e:
            return f"❌ Error selecting text: {str(e)}"        

    def handle_command(self, prompt):
        prompt = prompt.lower()
        if "open calculator" in prompt:
            self.show_response("Opening calculator...")
            subprocess.Popen("calc")
            return True
        elif "select the thing" in prompt or "select this" in prompt or "click this" in prompt:
            self.select_thing_on_screen()
            return True
        
        elif "click" in prompt:
            words = prompt.split()
            index = words.index("click")
            if index + 1 < len(words):
                target = words[index + 1]
                response = self.select_text_on_screen(target)
                self.show_response(response)
                return True
            
        
        return False

    def ask_cohere_from_text(self):
        prompt = self.input_box.text().strip()
        if not prompt:
            return
        self.input_box.clear()
        self.ask_cohere(prompt)

    def ask_cohere(self, prompt):
        self.response_label.setText("Thinking...")

        if self.handle_command(prompt):
            return

        def run():
            try:
                lower_prompt = prompt.lower()
                # Handle time request
                if "time" in lower_prompt:
                    time_info = self.get_time()
                    self.show_response(time_info)
                    return

                # Handle date request
                if "date" in lower_prompt:
                    date_info = self.get_current_date()
                    self.show_response(date_info)
                    return

                # Handle weather request
                if "weather in" in lower_prompt:
                    import re
                    city = lower_prompt.split("weather in")[-1].strip()
                    weather_info = self.get_weather(city)
                    self.show_response(weather_info)
                    return
                
                #intent = intent_model.predict([prompt])[0]
                result = None
                if hasattr(intent_model, "predict_proba"):
                    embedding = embed_model.encode([prompt])
                    probs = intent_model.predict_proba(embedding)[0]
                    max_prob = max(probs)
                    intent = intent_model.classes_[np.argmax(probs)]
                    
                    print(f"[Predicted Intent]: {intent} | Confidence: {max_prob:.2f}")

                    if max_prob > 0.7:
                        result = self.execute_action(intent, prompt)
                        if isinstance(result, str) and result not in ["Intent not recognized", None, ""]:
                            if result not in ["alarm_set", "alarm_error"]:
                                self.show_response(result)
                            return
                    else:
                        print("[Intent Detection] Low confidence. Falling back to Cohere.")

                """if isinstance(result, str) and "Intent not recognized" not in result:
                    self.show_response(result)
                    return
                if isinstance(result, str) and result not in ["Intent not recognized", None, ""]:
                    if result not in ["alarm_set", "alarm_error"]:  # Do not trigger Cohere for alarm
                        self.show_response(result)
                    return"""
            except Exception as e:
                print("[Intent Error]:", e)

            try:
                response = co.chat(message=prompt)
                answer = response.text.strip()
                print("[Cohere Response]:", answer)
                self.show_response(answer)
            except Exception as e:
                print("[Cohere Error]:", e)
                self.show_response("Sorry, something went wrong.")

        threading.Thread(target=run, daemon=True).start()

    def show_response(self, text):
        self.response_label.setText(text)
        self.response_label.adjustSize()
        self.speak(text)

    """def speak(self, text):
        def run():
            try:
                if self.is_paused:
                    self.speech_queue.append(text)
                    return
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print("[TTS Error]", e)
        threading.Thread(target=run, daemon=True).start()"""

    def listen_to_voice(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        def capture():
            try:
                with mic as source:
                    self.response_label.setText("Listening...")
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5)
                    query = recognizer.recognize_google(audio)
                    print("[Voice Input]:", query)
                    self.ask_cohere(query)
            except sr.WaitTimeoutError:
                self.response_label.setText("Didn't hear anything.")
            except sr.UnknownValueError:
                self.response_label.setText("Could not understand.")
            except Exception as e:
                print("[Voice Error]:", e)
                self.response_label.setText("Voice error.")

        threading.Thread(target=capture, daemon=True).start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_button.setText("Resume" if self.is_paused else "Pause")
        if not self.is_paused:
            for sentence in self.speech_queue:
                self.speak(sentence)
            self.speech_queue.clear()
        else:
            self.engine.stop()

    def wake_word_listener(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print("[Wake Listener] Started. Waiting for wake word...")
        print(f"[Wake Listener] Expected wake word: {self.wake_word}")
        while True:
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("[Wake Listener] Listening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    phrase = recognizer.recognize_google(audio).lower()
                    print("[Wake Listener] Heard:", phrase)
                    if self.wake_word in phrase:
                        print("[Wake Listener] Wake word detected!")
                        self.listen_to_voice()
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print("[Wake Listener Error]:", e)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #1e1e1e; color: white; border: 1px solid #444; padding: 6px; border-radius: 10px; }
            QMenu::item { padding: 8px 25px; margin: 4px 8px; background-color: transparent; border-radius: 6px; }
            QMenu::item:selected { background-color:  #333333; }
            QMenu::separator { height: 1px; background: #444; margin: 5px 10px; }
        """)

        change_avatar_menu = menu.addMenu("Change Avatar")
        avatar_root = "assets/idle"
        for folder in sorted(os.listdir(avatar_root)):
            folder_path = os.path.join(avatar_root, folder)
            if os.path.isdir(folder_path):
                action = QAction(folder.capitalize(), self)
                action.triggered.connect(lambda checked, path=folder_path: self.set_avatar(path))
                change_avatar_menu.addAction(action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)
        menu.exec_(event.globalPos())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mate = DesktopMate()
    mate.show()
    sys.exit(app.exec_())
