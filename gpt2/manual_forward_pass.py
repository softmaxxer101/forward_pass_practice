import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "openai-community/gpt2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use FP32 first so numerical comparison is easier.
# Change to torch.float16 only after your outputs match.
model_dtype = torch.float32

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def create_gpt2_attention_mask(
    batch_size,
    sequence_length,
    device,
    dtype,
    attention_mask=None,
):
    """
    Returns an additive attention mask with shape:

        [batch_size, 1, sequence_length, sequence_length]

    Allowed positions contain 0.
    Blocked positions contain the minimum value for the dtype.
    """

    future_token_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=device,
    ).triu diagonal=1

    blocked_positions = future_token_mask.view(
        1,
        1,
        sequence_length,
        sequence_length,
    ).expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )

    if attention_mask is not None:
        padding_positions = attention_mask[:, None, None, :] == 0
        blocked_positions = blocked_positions | padding_positions

    additive_mask = torch.zeros(
        batch_size,
        1,
        sequence_length,
        sequence_length,
        dtype=dtype,
        device=device,
    )

    additive_mask = additive_mask.masked_fill(
        blocked_positions,
        torch.finfo(dtype).min,
    )

    return additive_mask


# transformer block
#
# Implement this class yourself.
#
# It must contain these learned submodules:
#
#   self.ln_1
#   self.qkv
#   self.attn_proj
#   self.ln_2
#   self.fc
#   self.proj
#
# Expected shapes for GPT-2 small:
#
#   ln_1: LayerNorm(768)
#   qkv: 768 -> 2304
#   attn_proj: 768 -> 768
#   ln_2: LayerNorm(768)
#   fc: 768 -> 3072
#   proj: 3072 -> 768
#
# Its forward method must accept:
#
#   hidden_states:       [B, T, 768]
#   additive_attn_mask:  [B, 1, T, T]
#
# It must return:
#
#   hidden_states:       [B, T, 768]
#
# Use GPT-2 pre-layer normalization:
#
#   x = x + attention(layer_norm_1(x))
#   x = x + mlp(layer_norm_2(x))
#
# Do not place LayerNorm after the residual addition.
#
class TransformerBlock(nn.Module):
    # transformer block

    def __init__(self, config, layer_index):
        super().__init__()

        raise NotImplementedError(
            "Implement TransformerBlock before running this file."
        )


class GPT2Manual(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.n_embd
        self.num_layers = config.n_layer
        self.max_positions = getattr(
            config,
            "n_positions",
            config.n_ctx,
        )

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.position_embedding = nn.Embedding(
            self.max_positions,
            config.n_embd,
        )

        self.embedding_dropout = nn.Dropout(
            config.embd_pdrop,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    layer_index=index,
                )
                for index in range(config.n_layer)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.n_embd,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, sequence_length]"
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_positions:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"GPT-2 limit of {self.max_positions}"
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.long,
                device=input_ids.device,
            )

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(position_ids)

        hidden_states = token_vectors + position_vectors
        hidden_states = self.embedding_dropout(hidden_states)

        additive_attn_mask = create_gpt2_attention_mask(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            attention_mask=attention_mask,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                additive_attn_mask=additive_attn_mask,
            )

        hidden_states = self.final_layer_norm(hidden_states)

        # GPT-2 ties the output language-model head to token embeddings.
        logits = F.linear(
            hidden_states,
            self.token_embedding.weight,
        )

        if labels is None:
            return logits

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shifted_logits.view(-1, self.vocab_size),
            shifted_labels.view(-1),
            ignore_index=-100,
        )

        return logits, loss


def copy_exact_parameter(destination, source, name):
    source = source.to(
        device=destination.device,
        dtype=destination.dtype,
    )

    if destination.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for {name}: "
            f"destination={tuple(destination.shape)}, "
            f"source={tuple(source.shape)}"
        )

    destination.copy_(source)


def copy_huggingface_linear(
    destination_linear,
    state_dict,
    weight_name,
    bias_name,
):
    """
    Hugging Face GPT-2 Conv1D weights are stored as:

        [input_features, output_features]

    PyTorch Linear weights are stored as:

        [output_features, input_features]

    Therefore, the Hugging Face weight is transposed.
    """

    source_weight = state_dict[weight_name].transpose(0, 1)

    copy_exact_parameter(
        destination_linear.weight,
        source_weight,
        weight_name,
    )

    copy_exact_parameter(
        destination_linear.bias,
        state_dict[bias_name],
        bias_name,
    )


@torch.no_grad()
def load_gpt2_weights(manual_model, state_dict):
    copy_exact_parameter(
        manual_model.token_embedding.weight,
        state_dict["transformer.wte.weight"],
        "transformer.wte.weight",
    )

    copy_exact_parameter(
        manual_model.position_embedding.weight,
        state_dict["transformer.wpe.weight"],
        "transformer.wpe.weight",
    )

    copy_exact_parameter(
        manual_model.final_layer_norm.weight,
        state_dict["transformer.ln_f.weight"],
        "transformer.ln_f.weight",
    )

    copy_exact_parameter(
        manual_model.final_layer_norm.bias,
        state_dict["transformer.ln_f.bias"],
        "transformer.ln_f.bias",
    )

    for layer_index, block in enumerate(manual_model.blocks):

        prefix = f"transformer.h.{layer_index}."

        copy_exact_parameter(
            block.ln_1.weight,
            state_dict[prefix + "ln_1.weight"],
            prefix + "ln_1.weight",
        )

        copy_exact_parameter(
            block.ln_1.bias,
            state_dict[prefix + "ln_1.bias"],
            prefix + "ln_1.bias",
        )

        copy_huggingface_linear(
            destination_linear=block.qkv,
            state_dict=state_dict,
            weight_name=prefix + "attn.c_attn.weight",
            bias_name=prefix + "attn.c_attn.bias",
        )

        copy_huggingface_linear(
            destination_linear=block.attn_proj,
            state_dict=state_dict,
            weight_name=prefix + "attn.c_proj.weight",
            bias_name=prefix + "attn.c_proj.bias",
        )

        copy_exact_parameter(
            block.ln_2.weight,
            state_dict[prefix + "ln_2.weight"],
            prefix + "ln_2.weight",
        )

        copy_exact_parameter(
            block.ln_2.bias,
            state_dict[prefix + "ln_2.bias"],
            prefix + "ln_2.bias",
        )

        copy_huggingface_linear(
            destination_linear=block.fc,
            state_dict=state_dict,
            weight_name=prefix + "mlp.c_fc.weight",
            bias_name=prefix + "mlp.c_fc.bias",
        )

        copy_huggingface_linear(
            destination_linear=block.proj,
            state_dict=state_dict,
            weight_name=prefix + "mlp.c_proj.weight",
            bias_name=prefix + "mlp.c_proj.bias",
        )


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids,
    tokenizer,
    max_new_tokens=30,
):
    model.eval()

    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):

        model_input_ids = generated_ids[
            :,
            -model.max_positions:,
        ]

        model_attention_mask = torch.ones_like(
            model_input_ids,
        )

        logits = model(
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
        )

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token,
            ],
            dim=1,
        )

        if tokenizer.eos_token_id is not None:
            if torch.all(
                next_token == tokenizer.eos_token_id
            ):
                break

    return generated_ids


def main():

    print("Loading GPT-2 reference model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
    )

    reference_model = reference_model.to(device)
    reference_model.eval()

    print("Loading Hugging Face state dictionary...")

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in reference_model.state_dict().items()
    }

    print("Creating manual model...")

    manual_model = GPT2Manual(
        reference_model.config,
    )

    manual_model = manual_model.to(
        device=device,
        dtype=model_dtype,
    )

    load_gpt2_weights(
        manual_model,
        state_dict,
    )

    manual_model.eval()

    prompt = "Hello, I'm a language model,"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print("Comparing tokenization...")
    print("Input IDs:", input_ids.tolist())

    print("Comparing logits...")

    with torch.inference_mode():

        reference_outputs = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        reference_logits = reference_outputs.logits

        manual_logits = manual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    difference = (
        reference_logits.float()
        - manual_logits.float()
    ).abs()

    maximum_difference = difference.max().item()
    mean_difference = difference.mean().item()

    reference_next_token = reference_logits[
        :, -1, :
    ].argmax(dim=-1)

    manual_next_token = manual_logits[
        :, -1, :
    ].argmax(dim=-1)

    print("Maximum logit difference:", maximum_difference)
    print("Mean logit difference:", mean_difference)
    print(
        "Reference next token:",
        reference_next_token.tolist(),
    )
    print(
        "Manual next token:",
        manual_next_token.tolist(),
    )

    print("Reference next token text:")
    print(
        tokenizer.decode(
            reference_next_token,
            skip_special_tokens=True,
        )
    )

    print("Manual next token text:")
    print(
        tokenizer.decode(
            manual_next_token,
            skip_special_tokens=True,
        )
    )

    print("Manual greedy generation:")

    generated_ids = greedy_generate(
        model=manual_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        max_new_tokens=30,
    )

    print(
        tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
    )


if __name__ == "__main__":
    main()
