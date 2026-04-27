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

import zenoh

import time, json
import yaml
from typing import Optional, Tuple
import io
import struct

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.models.small_head import Edge_adapter


transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================================================
# Utility Functions
# ===============================================================
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

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, projected_actions, shead, device_id, past_img, curr_img):
        self.projected_actions = projected_actions
        self.shead = shead
        self.device_id = device_id
        self.past_img = past_img
        self.curr_img = curr_img
        self.linear, self.angular = 0.0, 0.0

    # ----------------------------
    # Action Head Inference
    # ----------------------------
    def run_shead(self) -> Tuple[float, float]:
        metric_waypoint_spacing = 0.1 # Distance (m) between waypoints on the generated trajectory

        with torch.no_grad():
            predicted_dactions = self.shead(self.curr_img, self.past_img, self.projected_actions)
            predicted_actions = delta_to_pose(predicted_dactions)
        
        linear_vel, angular_vel = self.pd_controller(predicted_actions.cpu(), metric_waypoint_spacing)

        print("action pose chunk", predicted_actions)
        print("linear velocity", linear_vel)
        print("angular velocity", angular_vel)

        return float(linear_vel), float(angular_vel)

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


def define_model(cfg: InferenceConfig) -> None:
    # GPU setup
    # torch.device: an object representing the hardware where a tensor is or will be allocated
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache() # release unused cached memory
    
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

    return shead, device_id


class InferenceHandler:
    def __init__(self, shead, device_id, cmd_pub):
        self.shead = shead
        self.device_id = device_id
        self.curr_actions = None
        self.cmd_pub = cmd_pub

    def action_callback(self, msg):
        payload = json.loads(msg.payload.decode("utf-8"))

        dtype_str = payload.get("dtype", "float32")
        np_dtype = np.dtype(dtype_str)
        shape = tuple(payload["shape"])
        data = np.array(payload["data"], dtype=np_dtype).reshape(shape)

        self.curr_actions = (
            torch.from_numpy(data)
            .to(self.device_id)
            .to(torch.bfloat16)
        )
    
    def process_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = TF.to_tensor(img)
        processed_tensor = TF.resize(img_tensor, (96, 96)).unsqueeze(0)
        return transform(processed_tensor).to(self.device_id).to(torch.bfloat16)

    def img_callback(self, msg):
        if self.curr_actions is None:
            return

        payload = json.loads(msg.payload.decode("utf-8"))

        past_jpeg_bytes = payload["past_img"].encode("latin-1")
        curr_jpeg_bytes = payload["curr_img"].encode("latin-1")

        past_img = self.process_image(past_jpeg_bytes)
        curr_img = self.process_image(curr_jpeg_bytes)

        inference = Inference(
            projected_actions=self.curr_actions,
            shead=self.shead,
            device_id=self.device_id,
            past_img=past_img,
            curr_img=curr_img
        )
        lin_x, ang_z = inference.run_shead()
        
        cmd_vel_payload = struct.pack("ddd", time.time(), float(lin_x), float(ang_z))
        self.cmd_pub.put(cmd_vel_payload)

# ===============================================================
# Main Entry
# ===============================================================
def main():
    cfg = InferenceConfig()
    shead, device_id = define_model(cfg)

    z_conf = zenoh.Config()
    z_conf.insert_json5(
        "connect/endpoints", '["tcp/127.0.0.1:7447", "tcp/127.0.0.1:7448"]'
    )
    with zenoh.open(z_conf) as z_session:
        cmd_vel_publisher = z_session.declare_publisher("/vla/cmd_vel")

        inference_handler = InferenceHandler(shead, device_id, cmd_vel_publisher)

        action_subscriber = z_session.declare_subscriber("/vla/actions", inference_handler.action_callback)
        img_subscriber = z_session.declare_subscriber("/camera/img_compressed", inference_handler.img_callback)

        while True:
            time.sleep(1)
    

if __name__ == "__main__":
    main()