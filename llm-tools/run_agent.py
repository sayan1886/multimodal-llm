#!/usr/bin/env python3
# llm-tools/run_agent.py (underscore version)

import os
import soundfile as sf
from agent import ToolAgent


def main():

    backend = os.getenv("IMAGE_BACKEND", "flux")
    print(f"\n===== DEMO: LLM + Tools (Whisper + Image Backend: {backend}) =====\n")

    # Initialize the ToolAgent (GPT-2 + Whisper + Image backends)
    agent = ToolAgent()

    # ------------------------------------------------------
    # DEMO 1: Speech-to-Text (Whisper)
    # ------------------------------------------------------
    print("DEMO 1: Speech Transcription using Whisper")
    try:
        audio_path = "./multimodal_dataset/sample_audio.wav"
        audio, sr = sf.read(audio_path)
        result_asr = agent.run("transcribe this audio", audio=audio)
        print("ASR Output:", result_asr)
    except Exception as e:
        print(f"ASR Demo Failed! Make sure '{audio_path}' exists.")
        print("Error:", e)

    print("\n" + "-"*60 + "\n")

    # ------------------------------------------------------
    # DEMO 2: Image Generation (selectable backend)
    # ------------------------------------------------------
    print(f"DEMO 2: Image Generation (backend: {backend})")
    try:
        prompt = "Generate image of a cute robot holding a red balloon"
        result_img = agent.run(prompt)
        print(f"{backend} Output:", result_img)
    except Exception as e:
        print(f"{backend} Demo Failed! Check your API key or endpoint configuration.")
        print("Error:", e)

    print("\n" + "-"*60 + "\n")

    # ------------------------------------------------------
    # DEMO 3: Normal Text LLM Response
    # ------------------------------------------------------
    print("DEMO 3: Normal LLM Text Response")
    result_text = agent.run("Explain what machine learning is")
    print("LLM Output:", result_text)

    print("\n===== END OF LLM TOOLS DEMO =====\n")


if __name__ == "__main__":
    main()
