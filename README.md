Practicing manual forward passes on random small huggingface language models.

Use this to convert weights of a downloaded huggingface model from safetensors to JSON readable format in the current directory.



python -c "
import json
path = '/home/alis/.cache/huggingface/hub/models--gpt2/blobs/248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
with open(path, 'rb') as f:
    n = int.from_bytes(f.read(8), 'little')
    print(json.dumps(json.loads(f.read(n)), indent=2))
" > current_model_weights.json
