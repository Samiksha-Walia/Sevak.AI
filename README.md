
# 🧠 Sevak.AI – AI-Powered Virtual Assistant

**Sevak.AI** is an intelligent, emotion-aware virtual desktop assistant built with Python. It integrates speech recognition, real-time emotion detection, avatar animation, and AI-driven responses using Cohere. The assistant offers voice-controlled automation, wake-word activation, and can visually react to the user's emotional state.

## 🚀 Features

- 🎙️ **Voice Interaction**: Wake-word activated and microphone-enabled voice commands.
- 🤖 **AI Chat**: Intelligent response generation powered by Cohere LLM API.
- 😃 **Emotion Detection**: Uses webcam to analyze facial expressions and respond empathetically.
- 🎭 **Customizable Avatars**: Load animated avatar sets from folders.
- 📋 **Intent Recognition**: Pre-trained model to map user queries to actions (open apps, websites, etc.).
- 🗓️ **Time, Date, Weather**: Real-time information retrieval via voice.
- 🔊 **Text-to-Speech**: Realistic voices using `edge-tts` and `pyttsx3`.
- 📂 **Desktop Control**: Perform tasks like launching apps, setting alarms, searching files, etc.
- 🖱️ **GUI Overlay**: PyQt5 interface with draggable, animated assistant.



## 🛠️ Built With

- **Python 3**
- `OpenCV` - Webcam and image processing
- `PyQt5` - GUI framework
- `keras` - Emotion recognition model
- `Cohere` - AI chat capabilities
- `pyttsx3`, `edge-tts` - Text-to-Speech
- `SpeechRecognition` - Voice input
- `joblib`, `sklearn` - Intent classification
- `MediaPipe`, `Pytesseract` - Facial and text recognition
- `playsound`, `winsound`, `pydub` - Audio alerts

## 📁 Directory Structure

```

project-root/
├── boy\_J.py                # Main application file
├── assets/                 # Avatar image folders
│   ├── idle/
│       ├── Sevak/
│       ├── Sakhi/
├── emotiondetector.json    # Emotion model architecture
├── emotiondetector.h5      # Emotion model weights
├── intent\_model\_LR.pkl     # Intent recognition model
├── embedding\_model.pkl     # Sentence embedding model
├── .env                    # Environment variables (Cohere & Weather API Keys)
├── ni/nircmd.exe           # Audio control utility

````

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/Sevak.AI.git
cd Sevak.AI
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Manually install additional packages if needed:

```bash
pip install opencv-python pyqt5 cohere edge-tts speechrecognition pyttsx3 pydub numpy keras pytesseract
```

### 4. Setup API Keys

Create a `.env` file with your keys:

```env
COHERE_API_KEY=your_cohere_api_key
WEATHER_API_KEY=your_weatherapi_key
```

### 5. Run the Application

```bash
python boy_J.py
```

## 🔒 Permissions Required

* Webcam Access (for emotion detection)
* Microphone Access (for voice commands)
* Internet Access (for Cohere API, Weather API)

## 📌 Future Improvements

* Add support for reminders and scheduling
* Integrate with calendars, email, or messaging apps
* Improve emotion detection using MediaPipe
* Cross-platform packaging (e.g., PyInstaller)

## 👤 Author

**Samiksha Walia**
[GitHub](https://github.com/Samiksha-Walia) • [LinkedIn](https://linkedin.com/in/samiksha-walia) 


