# ===============================================================
# AsyncVLA Inference
# ===============================================================
# 
# Sample inference code for AsyncVLA
# if you want to control the robot, you need to update the current state such as pose and image in "run_asyncvla"
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/", '../lerobot'
])

import time, math, json
import yaml
from typing import Optional, Tuple, Type, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens

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

def delta_to_pose(delta):
    """
    Fully differentiable, no inplace ops.
    delta: [N, T, 4]
    """
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    N, T = dx.shape

    # Allocate list, then stack (safe)
    poses = []

    # Initial pose
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]

    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    # Loop for t >= 1
    for t in range(1, T):
        ct = torch.cos(theta)
        st = torch.sin(theta)

        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]

        # NEW tensors — NOT inplace updates
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]

        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    # Stack at the end
    return torch.stack(poses, dim=1)

def clip_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def _sync_time_ms(device):
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize(device)
    return time.perf_counter() * 1000.0


# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, action_tokenizer, processor):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.datastore_path_image = save_dir
    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate # What would changing the tick rate do?
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                self.tick()
                start_time = time.time()
                break

    # Run AsyncVLA inference at 3Hz
    def tick(self):
        self.run_asyncvla()

    # ----------------------------
    # AsyncVLA Inference
    # ----------------------------
    def run_asyncvla(self):
        # Setup values for computing the goal pose in the robot's local frame
        thres_dist = 30.0 # Max distance (m) for the relative goal vector, to avoid huge far-away goals
        metric_waypoint_spacing = 0.1 # Distance (m) between waypoints on the generated trajectory

        # Load current GPS & heading - If we are not using pose goal / GPS conditioning, we don't need these values
        # ---
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass
        # ---

        # Local goal position
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius
        """
        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(self.goal_compass - cur_compass),
            np.sin(self.goal_compass - cur_compass)
        ])
        """
        # Hardcoded goal pose for testing
        # x = 10.0m, y = -100.0 , yaw = -90.0 degrees
        yaw_ang = -90.0
        goal_pose_loc_norm = np.array([
            1.0 / metric_waypoint_spacing,
            -10.0 / metric_waypoint_spacing,
            np.cos(yaw_ang/180.0*3.1415),
            np.sin(yaw_ang/180.0*3.1415)
        ])
        print("Goal pose on the robot coordinate for 1st image: [x, y], yaw=", goal_pose_loc_norm[0:2]*metric_waypoint_spacing, yaw_ang)
        # --- Up to this point, it is only used for goal pose conditioning - NOT RELEVANT FOR OUR PROJECT
        
        # Load current image
        current_image_path = "./inference/past.png"
        current_image_PIL = Image.open(current_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        next_image_path = "./inference/cur.png"
        next_image_PIL = Image.open(next_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        
        # Language instruction
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"

        # Run forward pass
        actions_list, modality_id = self.run_forward_pass(
            vla=vla.eval(),
            action_head=action_head.eval(),
            action_proj=action_proj.eval(),
            shead=shead.eval(),
            noisy_action_projector=None,
            pose_projector=pose_projector.eval(),
            current_image_PIL=current_image_PIL,
            lan_inst=lan_inst,
            goal_pose_loc_norm=goal_pose_loc_norm,
            metric_waypoint_spacing=metric_waypoint_spacing,
            action_tokenizer=self.action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False, # this must be true if we're using image+language conditioning
            num_patches=NUM_PATCHES,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
            idrun=self.count_id,
        )
        self.count_id += 1

        # Save behavior
        self.save_robot_behavior(
            current_image_PIL, next_image_PIL, goal_pose_loc_norm, actions_list, metric_waypoint_spacing, modality_id.cpu().numpy()
        )

    # ----------------------------
    # Save Robot Behavior Visualization
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, actions_list, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq_1 = actions_list[0][0, :, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv_1 = -actions_list[0][0, :, 1]           
        x_seq_2 = actions_list[1][0, :, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv_2 = -actions_list[1][0, :, 1]        
                
        ax_graph_pos.plot(np.insert(y_seq_inv_1, 0, 0.0), np.insert(x_seq_1, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue', label="1st image for edge adapter")
        ax_graph_pos.plot(np.insert(y_seq_inv_2, 0, 0.0), np.insert(x_seq_2, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='red', label="2nd image for edge adapter")
        
        # Mask annotation
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("1st egocentric image", fontsize=18)
        ax_goal.set_title("2nd Egocentric image", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15) 
        ax_graph_pos.tick_params(axis='y', labelsize=15) 
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  
        else:                           
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_xlim(-6.0, 6.0)
        ax_graph_pos.set_ylim(-0.1, 12.0)
        ax_graph_pos.legend()                
        ax_graph_pos.set_title("Normalized generated 2D trajectories from AsyncVLA", fontsize=18)
        
        save_path = os.path.join(self.datastore_path_image, "visualization_asyncvla.jpg")
        plt.savefig(save_path)

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
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
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])
        
        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
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

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for AsyncVLA
    # ----------------------------
    def data_transformer_asyncvla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # PD Controller
    # - This is not actually a PD controller, it just converts waypoints into wheel commands ensuring we stay within velocity limits
    # ----------------------------
    def pd_controller(self, actions, metric_waypoint_spacing):
        waypoints = actions.float().cpu().numpy()

        # Select waypoint - midpoint of the trajectory
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing # convert to meters
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
        DT = 1 / 3 # control interval

        # Aready at target
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT # IMPLEMENT CLIP_ANGLE
        
        # Target is directly to the side
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        # Move forward proportionally to forward error dx
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        # Clip velocities to be within limits
        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # Velocity limitation - set this depending on known robot's velocity limits
        maxv, maxw = 0.3, 0.3

        # Linear velocity is within limit
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                # If angular velocity is over the limit, reduce v proportionally to preserve turn shape
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        # Linear velocity over the limit, reduce v and w proportionally to preserve turn shape
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        
        return linear_vel_value_limit, angular_vel_value_limit
        
    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, action_proj, shead, noisy_action_projector, pose_projector,
                         current_image_PIL, lan_inst, goal_pose_loc_norm, metric_waypoint_spacing, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        
        # Prepare the token format the VLA expects
        batch = self.data_transformer_asyncvla(
            current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )

        # Modify if you want to use difussion for inference
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # Determine modality
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)

        #For Edge adapter - USE THESE IF YOU ARE GETTING ACTUAL IMAGES
        #img_cur = transform(c_image).to(device_id).to(torch.bfloat16)
        #img_past = transform(p_image).to(device_id).to(torch.bfloat16)        
        #img_cur = transform(batch["c_image"]).to(device_id).to(torch.bfloat16)
        #img_past = transform(batch["p_image"]).to(device_id).to(torch.bfloat16)

        t0 = _sync_time_ms(device_id)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id), # images
                modality_id=modality_id.to(torch.bfloat16).to(device_id), # which modality is being used
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id), # for goal pose conditioning
                proprio_projector=pose_projector, # for goal pose conditioning
                noisy_actions=noisy_actions if use_diffusion else None, # for diffusion
                noisy_action_projector=noisy_action_projector if use_diffusion else None, # for diffusion
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None, # for diffusion
                use_film=use_film,
            )

        t1 = _sync_time_ms(device_id)

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
            projected_actions = action_proj.predict_action(actions_hidden_states.detach(), modality_id.to(torch.bfloat16).to(device_id))

        t2 = _sync_time_ms(device_id)

        past_image_path = "./inference/past.png" 
        past_image_PIL = Image.open(past_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        p_image=TF.resize(to_tensor(past_image_PIL), (96, 96)).unsqueeze(0)   
        img_past = transform(p_image).to(device_id).to(torch.bfloat16)    

        t3 = _sync_time_ms(device_id)
        
        predicted_actions_list = []
        for i in range(2):
            # Load current image        
            if i==0:
                current_image_path = "./inference/past.png"
            elif i==1:
                current_image_path = "./inference/cur.png"
            current_image_PIL = Image.open(current_image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
            c_image=TF.resize(to_tensor(current_image_PIL), (96, 96)).unsqueeze(0)
            img_cur = transform(c_image).to(device_id).to(torch.bfloat16)

            # Predict action - based on past, and current images
            with torch.no_grad():
                predicted_dactions = shead(img_cur, img_past, projected_actions)
                predicted_actions = delta_to_pose(predicted_dactions)

            # Use a controller to convert the waypoints into wheel commands
            linear_vel, angular_vel = self.pd_controller(predicted_actions.cpu(), metric_waypoint_spacing)
            print("action pose chunk", predicted_actions)
            print("linear velocity", linear_vel)
            print("angular velocity", angular_vel)

            predicted_actions_list.append(predicted_actions.cpu().float())
        
        t4 = _sync_time_ms(device_id)

        print(
            f"VLA Inference time = {t1 - t0:.2f} ms | "
            f"action_proj time = {t2 - t1:.2f} ms | "
            f"image load time = {t3 - t2:.2f} ms | "
            f"shead_loop time={t4 - t3:.2f} ms | "
            f"total_inference time = {t4 - t0:.2f} ms"
        )

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions_list, modality_id



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
        f"\tPOSE_DIM: {POSE_DIM}\n" # Dimension of the robot's pose passed into the model (4)
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}" # How actions and pose signals are normalised
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

    # If applicable, instantiate proprio projector
    # Allos the VLA condition on the robot's pose
    # A small NN that maps a proprio vector (POSE_DIM) to the same embedding dimension as the LLM
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},
    )

    # Turns the VLA's hidden states into continuous actions
    # NOT USED CURRENTLY
    action_head = init_module(
        L1RegressionActionHead_idcat, # A small MLP trained to predict actions from transformer hidden states
        "action_head",
        cfg,
        device_id,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    # Turns the VLA's hidden states into a latent feature vector of dimension 1024 -> to be consumed by the edge adapter (shead)
    action_proj = init_module(
        Proj_Actiontokens,
        "action_proj",
        cfg,
        device_id,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": 1024},
        to_bf16=True,
    )
    
    #defining edge adapter
    with open("./config_nav/dataset_config.yaml", "r") as f:        
        config = yaml.safe_load(f)      
    
    # Small fusion head that combines:
    # - current image features
    # - past image features
    # - VLA's hidden states
    # then outputs trajectory pose chunks
    shead = Edge_adapter(
        obs_encoding_size=config["obs_encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    
    # Load pre-trained weights for the shead if available
    if cfg.resume and os.path.exists(os.path.join(cfg.vla_path, f"shead--{cfg.resume_step}_checkpoint.pt")):
        checkpoint_path_shead = os.path.join(cfg.vla_path, f"shead--{cfg.resume_step}_checkpoint.pt")
        print("Loading shead model from ", checkpoint_path_shead)
        latest_checkpoint_shead = torch.load(checkpoint_path_shead, map_location="cpu")
        state_dict = latest_checkpoint_shead
        if any(k.startswith("module.") for k in state_dict.keys()):
            print("Detected DDP-style checkpoint, removing 'module.' prefix...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = shead.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    
    shead.to(torch.bfloat16).to(device=device_id)
    
    # Get number of vision patches
    # Compute how many non-text tokens are prepended before language tokens. So we can later locate the text/action-token region correctly.
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    # Encode action chunks into tokens so then the positions of the action tokens can be determined from the VLA output
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, shead, action_proj, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    # select modality
    pose_goal = False
    satellite = False
    image_goal = False
    lan_prompt = True

    # Goal definitions
    lan_inst_prompt = "move toward blue trash bin"
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi
    goal_image_PIL = Image.open("./inference/goal.png").convert("RGB").resize((224, 224), Image.BILINEAR)

    # Define models (VLA, action_head, pose_projector, processor, etc.)
    cfg = InferenceConfig()
    vla, action_head, pose_projector, shead, action_proj, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)

    # Run inference
    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )
    inference.run()
