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




layers= 32
q_h= 15
kv_h= 5
hd=64


def rope(x, position)


def rms_norm(x, )

def attention(x, wq, wk, wv, wo, cache_k, cache_v):
    B,T,D = x.shape


    q= wq(x)
    k= wk(x)
    v= wv(x)

    q= q.reshape(B, T, q_h, hd).transpose(1,2)
    k= k.reshape(B, T, kv_h, hd).transpose(1,2)
    v= v.reshape(B, T, kv_h, hd).transpose(1,2)

    if cache_k is not NOne:
        k=torch.cat([cache_k, k], dim=2)
        v=torch.cat([cache_v, v], dim=2)

        new_k= k
        new_v= v

        #some interleave function for GQA


    scores= q@ k.transpose(-2, -1)
    scores = scores / (hd ** 0.5)

    mask= torch.triu(torch.ones(T,T, device=x.device), diagonal=1).bool()
    scores= scores.masked_fill(mask, float("inf"))
    attention= F.softmax(scores, dim=-1)
    out= attention @ v
    out= out.transpose(1,2).reshape(B,T,D)
    return wo(out) , cache_k, cache_v



def mlp(x, layer):
    #some mlp layer










def transformer_layer(x, layer, cache_k, cache_v):
    norm= rms_norm(x,)
    attention, cache_k, cache_v= attention()
    x= x + attention
    #mlp norm
norm= rms_norm(x,)
x= x+ mlp(norm, layer)

return x, cache_k, cache_v


def forward()

def generate()  #prompt nikaloooo bhaisahab
