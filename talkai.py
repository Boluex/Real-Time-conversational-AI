# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# from datetime import datetime
# import os
# import cv2
# import json
# import requests
# from faster_whisper import WhisperModel
# from TTS.api import TTS



# # === Settings ===
# DURATION = 5  # seconds
# SAMPLE_RATE = 16000
# AUDIO_FILENAME = "temp_audio.wav"
# CAPTURE_FOLDER = "captured_images"
# RESPONSE_AUDIO = "response.wav"
# LLM_ENDPOINT = "http://localhost:11434/api/generate"
# MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"




# # Create folder if not exists
# BASE_DIR = "mcp_ai"
# CAPTURE_FOLDER = os.path.join(BASE_DIR, "captured_images")
# os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# # Load Whisper Model
# print("Loading Whisper...")
# model = WhisperModel("small", device="cpu", compute_type="int8")
# # # Init TTS and Whisper
# tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)



# # === Function to record microphone ===
# def record_audio(filename, duration, sample_rate):
#     print("🎤 Listening...")
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
#     sd.wait()
#     sf.write(filename, audio, sample_rate)
#     print("✅ Audio recorded")

# # === Function to transcribe ===
# def transcribe_audio(filename):
#     segments, _ = model.transcribe(filename)
#     full_text = " ".join([seg.text.lower() for seg in segments])
#     print(f"🧠 Heard: {full_text}")
#     return full_text


# def query_llm(prompt, image_path=None):
#     print("🤖 Generating response...")
#     payload = {
#         "model": "llava:7b",
#         "prompt": prompt,
#     }
#     if image_path:
#         payload["image"] = image_path

#     response = requests.post(LLM_ENDPOINT, json=payload, stream=True)
#     full_response = ""
#     for line in response.iter_lines():
#         if line:
#             try:
#                 part = json.loads(line)
#                 full_response += part.get("response", "")
#             except json.JSONDecodeError:
#                 pass
#     print(f"🤖 Bot: {full_response.strip()}")
#     return full_response.strip()


# def speak(text):
#     if not text.strip():
#         print("⚠️ Nothing to say — empty response.")
#         return
#     print("🔊 Speaking...")
#     tts.tts_to_file(text=text, file_path=RESPONSE_AUDIO)
#     os.system(f"aplay {RESPONSE_AUDIO}")


# # === Function to take picture ===
# def take_picture():
#     print("📸 Opening camera...")
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Failed to open camera")
#         return None
#     ret, frame = cap.read()
#     if ret:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filepath = os.path.join(CAPTURE_FOLDER, f"image_{timestamp}.jpg")
#         cv2.imwrite(filepath, frame)
#         print(f"✅ Image saved to {filepath}")
#         cap.release()
#         return filepath
#     else:
#         print("❌ Failed to capture image")
#         cap.release()
#         return None

# # === Main Loop ===
# # === Main Loop ===
# while True:
#     record_audio(AUDIO_FILENAME, DURATION, SAMPLE_RATE)
#     text = transcribe_audio(AUDIO_FILENAME)

#     if "what do you see" in text:
#         print("👁️ Command recognized: Taking picture!")
#         image_path = take_picture()
#         if image_path:
#             response = query_llm("Describe this image", image_path=image_path)
#             speak(response)
#     else:
#         response = query_llm(text)
#         speak(response)







import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import os
import cv2
import json
import requests
from faster_whisper import WhisperModel
from TTS.api import TTS

# === Settings ===
DURATION = 5  # seconds
SAMPLE_RATE = 16000
AUDIO_FILENAME = "temp_audio.wav"
BASE_DIR = "mcp_ai"
CAPTURE_FOLDER = os.path.join(BASE_DIR, "captured_images")
RESPONSE_AUDIO = "response.wav"
LLM_ENDPOINT = "http://localhost:11434/api/chat"
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

# === Prepare folders ===
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# === Load Models ===
print("Loading Whisper...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

print("Loading TTS...")
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)

# === Audio Recorder ===
def record_audio(filename, duration, sample_rate):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, sample_rate)
    print("✅ Audio recorded")

# === Audio Transcriber ===
def transcribe_audio(filename):
    segments, _ = whisper_model.transcribe(filename)
    full_text = " ".join([seg.text.lower() for seg in segments])
    print(f"🧠 Heard: {full_text}")
    return full_text

# === LLM Query ===
def query_llm(prompt, image_path=None):
    print("🤖 Generating response...")

    payload = {
        "model": "llava:7b",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    if image_path:
        payload["image"] = image_path

    try:
        response = requests.post(LLM_ENDPOINT, json=payload, stream=True)
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    part = json.loads(line)
                    full_response += part.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
        print(f"🤖 Bot: {full_response.strip()}")
        return full_response.strip()

    except requests.exceptions.RequestException as e:
        print(f"❌ Error talking to LLM: {e}")
        return "Sorry, I couldn't connect to the AI model."

# === Text-to-Speech ===
def speak(text):
    if not text.strip():
        print("⚠️ Nothing to say — empty response.")
        return
    print("🔊 Speaking...")
    tts.tts_to_file(text=text, file_path=RESPONSE_AUDIO)
    os.system(f"aplay {RESPONSE_AUDIO}")

# === Take Picture ===
def take_picture():
    print("📸 Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return None

    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(CAPTURE_FOLDER, f"image_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"✅ Image saved to {filepath}")
        cap.release()
        return filepath
    else:
        print("❌ Failed to capture image")
        cap.release()
        return None

# === Main Loop ===
print("🎉 Voice Assistant Started! Say something like 'what do you see' or just talk.")
try:
    while True:
        record_audio(AUDIO_FILENAME, DURATION, SAMPLE_RATE)
        text = transcribe_audio(AUDIO_FILENAME)

        if not text.strip():
            print("⚠️ No speech detected. Try again.")
            continue

        if "what do you see" in text:
            print("👁️ Command recognized: Taking picture!")
            image_path = take_picture()
            if image_path:
                response = query_llm("Describe this image", image_path=image_path)
                speak(response)
        else:
            response = query_llm(text)
            speak(response)

        print("-" * 50)

except KeyboardInterrupt:
    print("\n👋 Exiting. Goodbye!")
