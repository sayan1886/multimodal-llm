#!/usr/bin/env python3
# llm-tools/run_agent.py

import os
import soundfile as sf
from pathlib import Path
from agent import ToolAgent

# Optional: auto-generate sample audio if missing
def generate_sample_audio(path: str):
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("Please install gTTS (`pip install gtts`) to auto-generate sample audio.")

    text = "A cute robot holding a red balloon in a sunny park."
    tts = gTTS(text)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tts.save(path)
    print(f"[INFO] Sample audio generated at: {path}")


def main():
    backend = os.getenv("IMAGE_BACKEND", "flux").lower()
    print(f"\n===== DEMO: LLM + Tools (Whisper + Image Backend: {backend}) =====\n")

    # Initialize the ToolAgent (LLM + Whisper + Image backends)
    agent = ToolAgent()

    # Path to the shared audio file
    audio_path = "./multimodal-dataset/sample_audio.wav"
    if not os.path.exists(audio_path):
        print("[INFO] Audio file not found. Generating a sample audio...")
        generate_sample_audio(audio_path)

    # ------------------------------------------------------
    # DEMO 1: Speech-to-Text (Whisper)
    # ------------------------------------------------------
    print("DEMO 1: Speech Transcription using Whisper")
    try:
        audio, sr = sf.read(audio_path)
        result_asr = agent.run("transcribe this audio", audio=audio)
        print("ASR Output:", result_asr)
    except Exception as e:
        print(f"ASR Demo Failed! Make sure '{audio_path}' exists and is valid.")
        print("Error:", e)

    print("\n" + "-"*60 + "\n")

    # ------------------------------------------------------
    # DEMO 2: Image Generation (selectable backend)
    # ------------------------------------------------------
    print(f"DEMO 2: Image Generation (backend: {backend})")
    try:
        # Use transcription from audio as prompt
        prompt = result_asr
        result_img = agent.run(f"generate image: {prompt}")
        print(f"{backend} Output saved at:", result_img)
    except RuntimeError as e:
        print(f"{backend} Demo Failed! Check your API key or endpoint configuration.")
        print("Error:", e)
    except Exception as e:
        print(f"{backend} Demo Failed!")
        print("Error:", e)

    print("\n" + "-"*60 + "\n")

    # ------------------------------------------------------
    # DEMO 3: Normal Text LLM Response
    # ------------------------------------------------------
    print("DEMO 3: Normal LLM Text Response")
    try:
        llm_input = f"Describe the scene: {result_asr}"
        result_text = agent.run(llm_input)
        print("LLM Output:", result_text)
    except Exception as e:
        print("LLM Demo Failed!")
        print("Error:", e)

    print("\n===== END OF LLM TOOLS DEMO =====\n")


if __name__ == "__main__":
    main()
