# ===============================================================
# AsyncVLA Inference - VLA only
# ===============================================================
#
# Input: Language instruction
# Output: Hidden action states
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/", '../lerobot'
])

import time
from typing import Optional, Tuple, Type, Dict

import zenoh

from functools import lru_cache

import io
import traceback

import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.models.small_head import Proj_Actiontokens

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

from transformers.modeling_outputs import CausalLMOutputWithPast

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

# vla, pose_proj, action_proj, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, vla, lan_inst_prompt, img, action_proj, device_id, num_patches, action_tokenizer, processor):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.img = img
        self.vla = vla
        self.action_proj = action_proj
        self.device_id = device_id
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate # What would changing the tick rate do?
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                actions = self.tick()
                start_time = time.time()
                return actions

    # Run AsyncVLA inference at 3Hz
    def tick(self):
        return self.run_asyncvla()

    # ----------------------------
    # AsyncVLA Inference
    # ----------------------------
    def run_asyncvla(self):
        # Run forward pass
        actions = self.run_forward_pass(
            vla=self.vla.eval(),
            action_proj=self.action_proj.eval(),
            noisy_action_projector=None,
            current_image_PIL=self.img,
            lan_inst=self.lan_inst_prompt,
            device_id=self.device_id,
            use_diffusion=False,
            use_film=False, # this must be true if we're using image+language conditioning
            num_patches=self.num_patches,
        )
        self.count_id += 1

        return actions.detach().to(torch.float32).cpu().numpy()

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            # duplicate observation image. Model expects both observation and goal image, but masks out the goal if modality is set to 7.
            stacked = torch.stack(pixel_values)
            pixel_values = torch.cat([stacked, stacked], dim=1)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        
        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(
        self,
        inst_obj,
        actions,
        current_image_PIL,
        prompt_builder,
        action_tokenizer,
        base_tokenizer,
        image_transform,
        predict_stop_token=True,
    ):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        dataset_name = "lelan"

        return dict(
            pixel_values_current=image_transform(current_image_PIL),
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for AsyncVLA
    # ----------------------------
    def data_transformer_asyncvla(self, current_image_PIL, lan_inst, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions

        batch_data = self.transform_datatype(
            lan_inst,
            actions,
            current_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        return batch
        
    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(
            self,
            vla,
            action_proj,
            current_image_PIL,
            lan_inst,
            device_id,
            num_patches,
            use_diffusion=False,
            use_film=False, # this must be true if we're using image+language conditioning
            noisy_action_projector=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        
        # Prepare the token format the VLA expects
        batch = self.data_transformer_asyncvla(
            current_image_PIL,
            lan_inst,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )

        # Modify if you want to use difussion for inference
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        modality_id = torch.as_tensor([7], dtype=torch.float32, device=device_id)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id), # images
                modality_id=modality_id.to(torch.bfloat16),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                noisy_actions=noisy_actions if use_diffusion else None, # for diffusion
                noisy_action_projector=noisy_action_projector if use_diffusion else None, # for diffusion
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None, # for diffusion
                use_film=use_film,
            )

        # To determine the action-related hidden states later on
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
         
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]              
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        # Predict actions from the action portion of hidden states
        with torch.no_grad():
            projected_actions = action_proj.predict_action(
                actions_hidden_states.detach(),
                modality_id.to(torch.bfloat16),
            )
        print(f"inst='{lan_inst}' proj_mean={projected_actions.mean().item():.4f}")

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return projected_actions


# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    # Whether to load trained weights on top of the base VLA under vla_path.
    # init_module loads pose_projector, action_head, and action_proj from checkpoint files.
    # define_model loads shead (edge adapter) from checkpoint files
    resume: bool = True
    vla_path: str = "./AsyncVLA_release"
    # Training step index in checkpoint filenames
    resume_step: Optional[int] = 750000

    # In OpenVLA, policies can output actions in different ways: discrete bins (token prediction), direction regression or diffusion.
    use_l1_regression: bool = True # Not used
    use_diffusion: bool = False # Not used
    use_film: bool = False # Not used

    # Number of RGB images the vision backbone will process per step.
    num_images_in_input: int = 2

    # Whether to use LoRA to fine-tune the base VLA. Not used for inference.
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

@lru_cache(maxsize=1)
def define_model(cfg: InferenceConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU setup
    # torch.device: an object representing the hardware where a tensor is or will be allocated
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache() # release unused cached memory

    # These constants are determined during training and cannot be changed without retraining the model
    # Establish the size of action and pose vectors
    print(
        "Detected constants:\n"
        # Short trajectory in (dx, dy dtheta) - 8 waypoints in the plane
        # Your controller (pd_controller) picks one waypoint and turns into wheel-style commands
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n" # number of future action steps the policy predicts in one chunk (8)
        # (x, y, cos(theta), sin(theta)) - robot pose
        f"\tACTION_DIM: {ACTION_DIM}\n" # size of one control vectoer per timestep (4)
        "\tGoal-pose conditioning disabled in this script"
    )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    # Load OpenVLA config
    AutoConfig.register("openvla", OpenVLAConfig)
    # Image processors: turns PIL images into pixel_values tensors -> handles resizing, cropping, and normalisation required by the specific model
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    # Processor: combines image processor and tokenizer -> outputs a BatchFeature with input_ids, attention_mask, and pixel_values
    # Processor uses class implementation from local prismatic pkg, but the configs are from AsyncVLA_release
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    # Transformers library designed for multimodal tasks
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    
    # Load processor and VLA
    # Load the multimodal preprocessor for this checkpoint
    # trust_remote_code=True: allow repo specific classes
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    # Load the processor: vision encoder + language model that maps images + texts to action
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16, # Efficient way of storing floats
        low_cpu_mem_usage=True, # Avoid huge RAM spikes
    ).to(device_id) #            trust_remote_code=True,
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    # Turns the VLA's hidden states into a latent feature vector of dimension 1024 -> to be consumed by the edge adapter (shead)
    action_proj = init_module(
        Proj_Actiontokens,
        "action_proj",
        cfg,
        device_id,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": 1024},
        to_bf16=True,
    )
    
    # Get number of vision patches
    # Compute how many non-text tokens are prepended before language tokens. So we can later locate the text/action-token region correctly.
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()

    # Create Action Tokenizer
    # Encode action chunks into tokens so then the positions of the action tokens can be determined from the VLA output
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_proj, device_id, NUM_PATCHES, action_tokenizer, processor

class InferenceHandler:
    def __init__(self, vla, action_proj, device_id, num_patches, action_tokenizer, processor, action_pub):
        self.vla = vla
        self.action_proj = action_proj
        self.device_id = device_id
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.action_pub = action_pub
        self.img = None
        self.lan_inst = None
        self.frame_id = None

    def process_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = TF.to_tensor(img)
        processed_tensor = TF.resize(img_tensor, (96, 96)).unsqueeze(0)
        return transform(processed_tensor).to(self.device_id).to(torch.bfloat16)

    def img_callback(self, msg):
        payload = json.loads(msg.payload.to_bytes().decode("utf-8"))
        self.frame_id = payload.get("frame_id")
        img_bytes = payload["curr_img"].encode("latin-1")
        self.img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if not hasattr(self, '_saved'):
            self.img.save('/tmp/vla_input.jpg')
            self._saved = True
            print("saved input image")
        self.maybe_run()

    def inst_callback(self, msg):
        self.lan_inst = msg.payload.to_bytes().decode("utf-8")
        print(f"instruction received: '{self.lan_inst}'")
        self.maybe_run()
    
    def maybe_run(self):
        if self.img is None or self.lan_inst is None:
            return
        print("before inference.run")
        inference = Inference(
            vla=self.vla,
            lan_inst_prompt=self.lan_inst,
            img=self.img,
            action_proj=self.action_proj,
            device_id=self.device_id,
            num_patches=self.num_patches,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        actions = inference.run()
        print("after inference.run")
        payload = {
            "t_vla": time.time(),
            "frame_id": self.frame_id,
            "dtype": str(actions.dtype),
            "shape": list(actions.shape),
            "data": actions.reshape(-1).tolist(),
        }
        self.action_pub.put(json.dumps(payload).encode("utf-8"))
        print("action sent")

# ===============================================================
# Main Entry
# ===============================================================

def main():
    cfg = InferenceConfig()
    vla, action_proj, device_id, num_patches, action_tokenizer, processor = define_model(cfg)

    z_conf = zenoh.Config()
    z_conf.insert_json5("mode", '"client"')
    z_conf.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')
    z_conf.insert_json5("scouting/multicast/enabled", "false")
    with zenoh.open(z_conf) as z_session:
        action_publisher = z_session.declare_publisher("vla/actions")

        inference_handler = InferenceHandler(vla, action_proj, device_id, num_patches, action_tokenizer, processor, action_publisher)

        inst_subscriber = z_session.declare_subscriber("robot/instruction", inference_handler.inst_callback)
        img_subscriber = z_session.declare_subscriber("camera/img_compressed", inference_handler.img_callback)

        print("VLA ready. Listening on Zenoh for camera/img_compressed and robot/instruction...", flush=True)
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
