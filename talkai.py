import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import requests
import json
from faster_whisper import WhisperModel
from TTS.api import TTS

# Configs
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
WHISPER_MODEL_SIZE = "small"
AUDIO_PATH = "user_input.wav"
RESPONSE_AUDIO = "response.wav"
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
LLM_ENDPOINT = "http://localhost:11434/api/generate"

# Init TTS and Whisper
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

def record_audio(filename, duration=RECORD_SECONDS, fs=SAMPLE_RATE):
    print(f"\n🎙️ Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio_data, fs)
    print("✅ Recording saved.")

def transcribe_audio(filename):
    print("📝 Transcribing...")
    segments, _ = whisper_model.transcribe(filename, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    print(f"🗣️ You said: {text}")
    return text.strip()

def query_llm(prompt):
    print("🤖 Generating response...")
    response = requests.post(LLM_ENDPOINT, json={"model": "llava:7b", "prompt": prompt}, stream=True)
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                part = json.loads(line)
                full_response += part.get("response", "")
            except json.JSONDecodeError:
                pass
    print(f"🤖 Bot: {full_response.strip()}")
    return full_response.strip()

def speak(text):
    print("🔊 Speaking...")
    tts.tts_to_file(text=text, file_path=RESPONSE_AUDIO)
    os.system(f"aplay {RESPONSE_AUDIO}")  

# Main loop
print("🎉 Real-time voice assistant started. Press Ctrl+C to exit.")
try:
    while True:
        record_audio(AUDIO_PATH)
        user_input = transcribe_audio(AUDIO_PATH)
        if not user_input:
            print("⚠️ Nothing detected. Try speaking again.")
            continue
        response = query_llm(user_input)
        speak(response)
        print("-" * 50)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n👋 Exiting. Goodbye!")
