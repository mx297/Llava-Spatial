import torch
from types import SimpleNamespace
import transformers

# Core LLaVA imports (match your repo layout)
from llava.model.language_model.llava_llama_spatial import LlavaLlamaSpatialForCausalLM
#from llava import conversation as conversation_lib

# --- Choose your model + towers (replace with the actual IDs/paths you use) ---
BASE_LLM = "lmsys/vicuna-7b-v1.5"             # example base LLM
VISION_TOWER = "google/siglip-so400m-patch14-384" # example vision tower used by LLaVA
SPATIAL_TOWER = "vggt"      # <-- your spatial tower checkpoint
FUSION_BLOCK  = "cross_attn_spatial_fusion"        # <-- the fusion block name your code expects

def build_model(device="cuda", use_bf16=True):
    # 1) Load the spatial variant of the model
    model_args = SimpleNamespace(
        # --- standard vision/projector fields your repo uses ---
        model_name_or_path=BASE_LLM,
        vision_tower=VISION_TOWER,
        mm_vision_select_layer=-1,
        pretrain_mm_mlp_adapter=None,
        mm_projector_type="linear",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=True,
        mm_patch_merge_type="flat",
        mm_vision_select_feature="patch",
        tune_mm_mlp_adapter = True,

        # --- spatial / fusion specific (NEW) ---
        spatial_tower=SPATIAL_TOWER,
        spatial_tower_select_feature="all",  # "patch_tokens", "camera_tokens", or "all"
        spatial_tower_select_layer=-1,
        spatial_feature_dim=768,            # set if your spatial tower needs this explicitly
        tune_spatial_tower=False,

        fusion_block=FUSION_BLOCK,
        tune_fusion_block=True,
    )
    model = LlavaLlamaSpatialForCausalLM.from_pretrained(
        BASE_LLM,
        attn_implementation="flash_attention_2",  # or "sdpa"/None, depending on your setup
        torch_dtype=(torch.bfloat16 if use_bf16 else torch.float16),
    )

    # 2) Prepare a lightweight object with the attributes initialize_*() expect.
    #    You can also import your ModelArguments dataclass and instantiate that instead.

    # 3) Initialize the vision modules (this sets up the vision tower + projector)
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=None
    )

    # Put the vision tower on device + dtype
    vision_tower = model.get_vision_tower()
    vision_tower.to(
        dtype=(torch.bfloat16 if use_bf16 else torch.float16),
        device=device
    )

    # 4) Initialize the spatial tower (optional but requested)
    if hasattr(model.get_model(), "initialize_spatial_tower") and model_args.spatial_tower is not None:
        model.get_model().initialize_spatial_tower(model_args, fsdp=None)
        spatial_tower = model.get_model().get_spatial_tower()
        if spatial_tower is not None:
            spatial_tower.to(
                dtype=(torch.bfloat16 if use_bf16 else torch.float16),
                device=device
            )

    # 5) Initialize the fusion block (optional but requested)
    if hasattr(model.get_model(), "initialize_fusion_block") and model_args.fusion_block is not None:
        model.get_model().initialize_fusion_block(model_args, fsdp=None)
        fusion_block = model.get_model().get_fusion_block()
        if fusion_block is not None:
            fusion_block.to(
                dtype=(torch.bfloat16 if use_bf16 else torch.float16),
                device=device
            )
    
    return model

model = build_model()
print(model)