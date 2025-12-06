# llm-tools/whisper.py

from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperASR:
    def __init__(self, model="openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model)

    def transcribe(self, audio_array, sr=16000):
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
        ids = self.model.generate(inputs["input_features"])
        return self.processor.decode(ids[0])
