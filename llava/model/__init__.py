# try:
#     from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
#     from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
#     from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
# except:
#     pass
# LLaVA model registry — explicit imports with proper fallbacks

# --- Original LLaMA (LLaVA-LLaMA) ---
try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
except Exception as e:
    print(f"[WARN] Could not import LlavaLlamaForCausalLM: {e}")
    LlavaLlamaForCausalLM, LlavaConfig = None, None

# --- MPT variant ---
try:
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
except Exception as e:
    print(f"[WARN] Could not import LlavaMptForCausalLM: {e}")
    LlavaMptForCausalLM, LlavaMptConfig = None, None

# --- Mistral variant ---
try:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
except Exception as e:
    print(f"[WARN] Could not import LlavaMistralForCausalLM: {e}")
    LlavaMistralForCausalLM, LlavaMistralConfig = None, None

# --- Spatial LLaMA-3 variant (your custom model) ---
try:
    from .language_model.llava_llama_spatial import LlavaLlamaSpatialForCausalLM, LlavaSpatialConfig
except Exception as e:
    print(f"[WARN] Could not import LlavaLlamaSpatialForCausalLM: {e}")
    LlavaLlamaSpatialForCausalLM, LlavaSpatialConfig = None, None

__all__ = [
    "LlavaLlamaForCausalLM",
    "LlavaMptForCausalLM",
    "LlavaMistralForCausalLM",
    "LlavaLlamaSpatialForCausalLM",
    "LlavaConfig",
    "LlavaMptConfig",
    "LlavaMistralConfig",
    "LlavaSpatialConfig",
]
