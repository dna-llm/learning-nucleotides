from kan_gpt.model import GPT as KAN_GPT
from models import (
    AE,
    VAE,
    MistralDenseFormerForCausalLM,
    MultiresTransformer,
    MultiresTransformerConfig,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def format_param_count(num_params: float) -> str:
    suffixes = {
        1_000_000_000: "B",
        1_000_000: "M",
        1_000: "K",
    }

    for divisor, suffix in sorted(suffixes.items(), reverse=True):
        if num_params >= divisor:
            base_value = num_params / divisor
            rounded_base_value = round(base_value, 1) if base_value < 10 else round(base_value)
            formatted_value = f"{rounded_base_value}{suffix}"
            # Handle edge case where rounding up crosses the threshold to the next suffix
            if formatted_value.endswith("1000K"):
                return f"{num_params / 1_000_000:.1f}M".replace(".0", "")
            elif formatted_value.endswith("1000M"):
                return f"{num_params / 1_000_000_000:.1f}B".replace(".0", "")

            return formatted_value.replace(".0", "")

    return str(int(num_params))


def load_tokenizer(tokenizer_name: str | None = None) -> AutoTokenizer | None:
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_pythia(config_name: str, **kwargs) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(config_name)
    config.vocab_size = 8  # tokenizer.vocab_size
    model = AutoModelForCausalLM.from_config(config)

    return model


def load_denseformer(config_name: str, **kwargs) -> MistralDenseFormerForCausalLM:
    base = AutoConfig.from_pretrained(config_name)
    denseformer_config = "winglian/mistral-denseformer-7b"
    custom = AutoConfig.from_pretrained(denseformer_config, trust_remote_code=True)
    # Transfer settings from base to custom config
    settings_to_transfer = [
        "vocab_size",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "tie_word_embeddings",
        "transformers_version",
        "use_cache",
    ]
    for attr in settings_to_transfer:
        setattr(custom, attr, getattr(base, attr))

    custom.rms_norm_eps = base.rotary_pct
    custom.rope_theta = base.rotary_emb_base
    custom.num_key_value_heads = custom.num_attention_heads
    custom.vocab_size = 8

    return MistralDenseFormerForCausalLM(custom)


def load_evo(config_name: str, tokenizer_name: str, **kwargs) -> AutoModelForCausalLM:
    config_p = AutoConfig.from_pretrained(config_name)
    evo_config = "togethercomputer/evo-1-8k-base"
    config = AutoConfig.from_pretrained(evo_config, trust_remote_code=True, revision="1.1_fix")
    config.vocab_size = 8
    config.auto_map["AutoTokenizer"] = tokenizer_name
    config.max_seqlen = 2048
    config.bos_token_id = config_p.bos_token_id
    config.eos_token_id = config_p.eos_token_id
    config.hidden_size = config_p.hidden_size
    config.num_filters = config_p.hidden_size
    config.inner_mlp_size = 1024
    config.tie_embeddings = False
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    return model


def load_wavelet(**kwargs):
    config = MultiresTransformerConfig(
        n_tokens=8,
        d_model=128,
        n_layers=256,
        kernel_size=2,
        depth=4,
        dropout=0.1,
        d_mem=1024,
        indep_res_init=True,
        tree_select="fading",
        hinit=None,  # "coif1"
        max_seqlen=2048,
        d_input=6,
        nr_logistic_mix=3,
    )
    model = MultiresTransformer(config)

    return model


def load_kan(max_seq_length: int, **kwargs):
    config = KAN_GPT.get_default_config()
    config.model_type = "gpt-mini"
    config.vocab_size = 8
    config.block_size = max_seq_length
    model = KAN_GPT(config)

    return model


def load_ae(**kwargs):
    model = AE(
        sequence_length=1000,
        linear_sizes=[1024, 512, 256, 128, 64],
        activation="leakyrelu",
    )

    return model


def load_vae(**kwargs):
    model = VAE(
        sequence_length=1000,
        linear_sizes=[1024, 512, 256, 128, 64],
        activation="leakyrelu",
    )

    return model
