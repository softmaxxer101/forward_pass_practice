import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer



print("Loading tokenizer and reference model...")

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct" # USE official model name from huggingface

tok = AutoTokenizer.from_pretrained(MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"



hf_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)

sd = hf_model.state_dict()
