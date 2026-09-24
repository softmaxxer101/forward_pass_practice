from transformers import pipeline, set_seed

# Load the model. It will automatically download the weights.
generator = pipeline('text-generation', model='gpt2', device_map="auto")
set_seed(42)

# Generate text
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
print(result[0]['generated_text'])
