# llm-tools/agent.py

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load sibling modules dynamically so this file can be executed
# both as a package member and as a top-level module (script).
import importlib.util
import os

_HERE = os.path.dirname(__file__)

def _load_module_from_sibling(name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_mod_whisper = _load_module_from_sibling("whisper", "whisper.py")
WhisperASR = _mod_whisper.WhisperASR

_mod_flux = _load_module_from_sibling("flux", "flux.py")
generate_flux = _mod_flux.generate_flux

_mod_sd = _load_module_from_sibling("stable_diffusion", "stable_diffusion.py")
generate_sd = _mod_sd.generate_sd

_mod_nb = _load_module_from_sibling("nano_banana", "nano_banana.py")
generate_nb = _mod_nb.generate_nano_banana

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
            # Allow switching image backend via IMAGE_BACKEND env var.
            # Supported values: 'flux' (default), 'stable-diffusion'
            backend = os.getenv("IMAGE_BACKEND", "flux").lower()
            if backend in ("stable-diffusion", "stable_diffusion", "sd"):
                return generate_sd(instruction)
            if backend in ("nano-banana", "nano_banana", "nano"):
                return generate_nb(instruction)
            else:
                return generate_flux(instruction)

        else:
            inputs = self.tokenizer(instruction, return_tensors="pt")
            result = self.llm.generate(**inputs, max_new_tokens=80)
            return self.tokenizer.decode(result[0])
