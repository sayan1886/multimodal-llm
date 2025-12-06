# llm-tools/agent.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from .tools_whisper import WhisperASR
from .tools_flux import generate_flux

class ToolAgent:
    def __init__(self, model="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = AutoModelForCausalLM.from_pretrained(model)
        self.asr = WhisperASR()

    def decide(self, instruction):
        text = instruction.lower()
        if "transcribe" in text or "audio" in text:
            return "asr"
        if "generate image" in text or "draw" in text:
            return "flux"
        return "llm"

    def run(self, instruction, audio=None):
        tool = self.decide(instruction)

        if tool == "asr":
            return self.asr.transcribe(audio)

        elif tool == "flux":
            return generate_flux(instruction)

        else:
            inputs = self.tokenizer(instruction, return_tensors="pt")
            result = self.llm.generate(**inputs, max_new_tokens=80)
            return self.tokenizer.decode(result[0])
