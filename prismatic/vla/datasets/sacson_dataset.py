import os
import io
import pickle
import yaml
import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from torchvision.transforms.functional import to_tensor

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

def img_path_to_data_front(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """Load image and convert to tensor."""
    #print(image_resize_size)
    return TF.to_tensor(Image.open(path).resize((224, 224), Image.Resampling.LANCZOS))


def img_path_to_data_front_PIL(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> Image.Image:
    """Load image and return PIL.Image."""
    #print(image_resize_size)
    return Image.open(path).resize((224, 224), Image.Resampling.LANCZOS)


class SACSoN_Dataset_rand(Dataset):
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        data_split_type: str,
        data_image_folder: str,
        data_pickle_folder: str,
        predict_stop_token: bool = True,
        context_type: str = "temporal",
        normalize: bool = True,
        backside: bool = False,
        aug_seq: bool = False,
        only_front: bool = False,
    ):
        """Main LeLaN Dataset class."""
        # General config
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        self.data_image_folder = data_image_folder
        self.data_pickle_folder = data_pickle_folder
        self.image_size = image_size
        self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        assert context_type in {"temporal", "randomized", "randomized_temporal"}, \
            "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.normalize = normalize
        self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        self.only_front = only_front

        # Load dataset configuration
        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config, f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = sorted(all_data_config.keys())
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]

        # Tokenizers and prompt builder
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform

        # Action parameters
        self.num_action_params = 3 if self.learn_angle else 2

        # Load dataset indices
        self._load_split_index()
        #self._get_augdata()
        self._build_caches_front()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    # ----------------------------------------
    # Dataset loading / caching
    # ----------------------------------------
    def _load_split_index(self):
        if self.dataset_name == "sacson":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            odom_path = []   
            ped_flag = [] 
            ped_list = [] 
                                                      
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            folder_lst_dataset = folder_lst                 
            for folder in folder_lst_dataset:
                subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")                                    
                for subfolder in subfolder_lst:
                    #print(self.data_image_folder + folder + "/" + subfolder + "/" + subfolder + "_" + folder + "_fisheye.txt")
                    with open(self.data_image_folder + folder + "/" + subfolder + "/" + subfolder + "_" + folder + "_fisheye.txt", "r") as f:
                        fisheye = f.read().splitlines() #.replace("fisheye", "fisheye_small")                            
                    with open(self.data_image_folder + folder + "/" + subfolder + "/" + subfolder + "_" + folder + "_odom.txt", "r") as f:
                        odom = f.read().splitlines()   
                    with open(self.data_image_folder + folder + "/" + subfolder + "/" + subfolder + "_" + folder + "_pedlist_update.txt", "r") as f:
                        ped = f.read().splitlines()   
                    num_min = min(len(fisheye), len(odom), len(ped))
                                                                
                    image_path += fisheye[0:num_min]
                    odom_path += odom[0:num_min]             
                    ped_flag += ped[0:num_min]

        self.image_path = image_path
        self.odom_path = odom_path
        self.ped_flag = ped_flag

    def _get_augdata(self):
        self.aug_data_list = []
        for path in self.pickle_path:
            if os.path.getsize(path) > 0:
                with open(path, "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(f"Empty pickle: {path}")
                aug_data = []
            self.aug_data_list.append(aug_data)
    
    def _build_caches_front(self):
        """Build LMDB cache for faster image loading."""
        cache_file = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb")
        if not os.path.exists(cache_file):
            with lmdb.open(cache_file, map_size=2**40) as env:
                with env.begin(write=True) as txn:
                    for img_path in self.image_path:
                        #print(self.data_image_folder + img_path)
                        with open(self.data_image_folder + img_path, "rb") as f:
                            txn.put((self.data_image_folder + img_path).encode(), f.read())
        self._image_cache_path = cache_file
        self._image_cache = None

    def _get_image_cache(self):
        if self._image_cache is None:
            self._image_cache = lmdb.open(
                self._image_cache_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048
            )
        return self._image_cache

    def _load_image_front(self, path):
        """Load image from LMDB."""
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buffer = txn.get(path.encode())
            return img_path_to_data_front(io.BytesIO(buffer), self.image_size)
        except TypeError:
            print(f"Failed to load image {path}")
            return None

    def _load_image_front_PIL(self, path):
        """Load image as PIL from LMDB."""
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buffer = txn.get(path.encode())
            return img_path_to_data_front_PIL(io.BytesIO(buffer), self.image_size)
        except TypeError:
            print(f"Failed to load image {path}")
            return None

    # ----------------------------------------
    # Helper functions
    # ----------------------------------------
    def _resize_norm(self, image, size):
        return TF.resize(image, size)

    def _sample_negative(self):
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]

    # ----------------------------------------
    # Dataset API
    # ----------------------------------------
    def __len__(self):
        return len(self.image_path)-50

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        flag_data = 0
        iv = i
        rand_dist = random.random() 

        while flag_data == 0:
            # random delay
            dfps = 12/3                             
            goal_id = random.randint(0,50*dfps)              
            #goal_id = 50*dfps
            lt = random.randint(0, min(iv, 9*dfps))    
            #lt = 0
            ped_flag_1 = float(self.ped_flag[iv].split()[0])
            ped_dist_1 = float(self.ped_flag[iv].split()[1])            
            ped_flag_2 = float(self.ped_flag[iv-lt].split()[0])
            ped_dist_2 = float(self.ped_flag[iv-lt].split()[1])
                                    
            if (ped_flag_1 > 0.5 or ped_flag_2 > 0.5) and (ped_dist_1 < 4.0 or ped_dist_1 < 4.0):            
                flag_data = 1
            else:
                iv = random.randint(0, len(self.image_path)-50-1)
                continue
            
            image_fullsize_PIL = self._load_image_front_PIL(self.data_image_folder + self.image_path[iv-lt])                            
            image_fullsize = self._load_image_front(self.data_image_folder + self.image_path[iv])                        
            #print(image_fullsize.size())
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                if iv-ih > 0:                   
                    context_image.append(self._load_image_front(self.data_image_folder + self.image_path[iv-ih]))             
                else:
                    context_image.append(self._load_image_front(self.data_image_folder + self.image_path[0]))          
            
            for ih in range(self.context_size + 1):
                if context_image[ih] is None:
                    iv = random.randint(0, len(self.image_path)-50-1)

            try:
                goal_image_full_8 = self._load_image_front(self.data_image_folder + self.image_path[min(iv + int(dfps*8), len(self.image_path)-1)])
            except:
                goal_image_full_8 = self._load_image_front(self.data_image_folder + self.image_path[iv])  
    
            try:
                gimage_fullsize_PIL = self._load_image_front_PIL(self.data_image_folder + self.image_path[int(min(iv + goal_id - lt, len(self.image_path)-1))])  
                goal_odoms_txt = [self.odom_path[int(min(iv + goal_id - lt, len(self.image_path)-1))].split()]            
            except:
                goal_id = 0
                gimage_fullsize_PIL = self._load_image_front_PIL(self.data_image_folder + self.image_path[int(iv + goal_id - lt)])
                goal_odoms_txt = [self.odom_path[int(min(iv + goal_id - lt, len(self.image_path)-1))].split()]     
                
            cimage_fullsize_PIL = self._load_image_front_PIL(self.data_image_folder + self.image_path[iv])
            distance = int(goal_id/dfps)
                                     
            odoms_txt = [self.odom_path[iv + int(dfps*i)].split() for i in range(9)] + goal_odoms_txt
            odoms_lt_txt = [self.odom_path[iv -lt + int(dfps*i)].split() for i in range(9)] + goal_odoms_txt                    
            odoms_xy = [[float(odom_list[0]), float(odom_list[1])] for odom_list in odoms_txt]
            odoms_yaw = [float(odom_list[5]) for odom_list in odoms_txt]    
            odoms_lt_xy = [[float(odom_lt_list[0]), float(odom_lt_list[1])] for odom_lt_list in odoms_lt_txt]
            odoms_lt_yaw = [float(odom_lt_list[5]) for odom_lt_list in odoms_lt_txt]                                        
            waypoints = to_local_coords(np.array(odoms_xy[1:]), np.array(odoms_xy[0]), np.array(odoms_yaw[0]))
            waypoints_lt = to_local_coords(np.array(odoms_lt_xy[1:]), np.array(odoms_lt_xy[0]), np.array(odoms_lt_yaw[0]))            
            norm_param = 0.125
            actions_all = [[waypoints[i].tolist()[0]/norm_param, waypoints[i].tolist()[1]/norm_param, np.cos(odoms_yaw[i+1] - odoms_yaw[0]), np.sin(odoms_yaw[i+1] - odoms_yaw[0])] for i in range(9)]
            actions_lt_all = [[waypoints_lt[i].tolist()[0]/norm_param, waypoints_lt[i].tolist()[1]/norm_param, np.cos(odoms_lt_yaw[i+1] - odoms_lt_yaw[0]), np.sin(odoms_lt_yaw[i+1] - odoms_lt_yaw[0])] for i in range(9)]
                        
            actions = np.array(actions_all[0:-1])
            actions_lt = np.array(actions_lt_all[0:-1])
                        
            dist = np.sqrt((actions[7,0]-actions_lt[7,0])**2 + (actions[7,1]-actions_lt[7,1])**2)
            if rand_dist > 0.9 and dist > 4.0:
                flag_data = 1
            elif not rand_dist > 0.9:
                flag_data = 1
            else:
                flag_data = 0
                iv = random.randint(0, len(self.image_path)-50-1)  
            
            if np.max(np.abs(actions)) > 15.0:
                flag_data = 0
                iv = random.randint(0, len(self.image_path)-50-1)

            goal_pose_cos_sin = np.array(actions_lt_all[-1])            
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())  
        
        image_obs_list = [] 
        if self.only_front:
            for ih in range(self.context_size + 1):
                image_obs_list.append(self._resize_norm(context_image[ih][:, 0:224, 0:224], self.image_size))    
            goal_image_full_8 = self._resize_norm(goal_image_full_8[:, 0:224, 0:224], self.image_size)   
            PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
            cropped_image_fullsize_PIL = image_fullsize_PIL.crop(PILbox).resize(self.image_size_clip) 
            cropped_cimage_fullsize_PIL = cimage_fullsize_PIL.crop(PILbox).resize(self.image_size_clip)             
            cropped_gimage_fullsize_PIL = gimage_fullsize_PIL.crop(PILbox).resize(self.image_size_clip) 
        else:
            for ih in range(self.context_size + 1):     
                image_obs_list.append(self._resize_norm(context_image[ih][:, 0:224, 0:224], self.image_size))    
            goal_image_full_8 = self._resize_norm(goal_image_full_8[:, 0:224, 0:224], self.image_size)       
            PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
            cropped_image_fullsize_PIL = image_fullsize_PIL.crop(PILbox).resize(self.image_size_clip)            
            cropped_cimage_fullsize_PIL = cimage_fullsize_PIL.crop(PILbox).resize(self.image_size_clip)              
            cropped_gimage_fullsize_PIL = gimage_fullsize_PIL.crop(PILbox).resize(self.image_size_clip)                               
                                             
        image_obs = torch.cat(image_obs_list[::-1])      
        if random.random() > 0.5:       
            image_obs_r = torch.flip(image_obs, [2])
            goal_image_full_8_r = torch.flip(goal_image_full_8, [2])            
            actions[:,1] = -actions[:,1]
            actions[:,3] = -actions[:,3]  
            actions_lt[:,1] = -actions_lt[:,1]
            actions_lt[:,3] = -actions_lt[:,3]              
            goal_pose_cos_sin[1] = -goal_pose_cos_sin[1]
            goal_pose_cos_sin[3] = -goal_pose_cos_sin[3]             
            cropped_image_fullsize_PIL_r = cropped_image_fullsize_PIL.transpose(Image.FLIP_LEFT_RIGHT)      
            cropped_cimage_fullsize_PIL_r = cropped_cimage_fullsize_PIL.transpose(Image.FLIP_LEFT_RIGHT)                     
            cropped_gimage_fullsize_PIL_r = cropped_gimage_fullsize_PIL.transpose(Image.FLIP_LEFT_RIGHT)    
        else:
            image_obs_r = image_obs
            goal_image_full_8_r = goal_image_full_8            
            cropped_image_fullsize_PIL_r = cropped_image_fullsize_PIL
            cropped_cimage_fullsize_PIL_r = cropped_cimage_fullsize_PIL            
            cropped_gimage_fullsize_PIL_r = cropped_gimage_fullsize_PIL
            actions = actions 
            actions_lt = actions_lt             
            goal_pose_cos_sin = goal_pose_cos_sin 

        # Set the available modality id for each dataset 
        # 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"
        modality_list = [4, 5, 6]   
        if distance <= 20:
            modality_id = random.choice(modality_list)
        else:
            modality_id = random.choice(modality_list[0:2]) #tdisntace is long --> no image only

        ### Adapting OpenVLA stle ###
        #actions = nomad_traj_norm
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
                
        conversation = [
            {"from": "human", "value": f"No language instruction"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder("openvla")
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)     
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)   
        
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)    
        pixel_values = self.image_transform(cropped_image_fullsize_PIL_r)
        pixel_values_g = self.image_transform(cropped_gimage_fullsize_PIL_r)
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!     
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
            
        action_chunk_tokens = self.base_tokenizer(action_chunk_string, add_special_tokens=False).input_ids
        action_chunk_len_x = len(action_chunk_tokens)
        dataset_name = "sacson"         
            
        #action select 1.0: raw action, 0.0: MBRA synthetic action            
        action_select_mask = torch.tensor(1.0)            
                        
        return dict(
            pixel_values=pixel_values, 
            pixel_values_goal=pixel_values_g, 
            input_ids=input_ids, 
            labels=labels, 
            dataset_name=dataset_name, 
            modality_id=modality_id,
            actions=torch.as_tensor(actions),  
            action_select_mask = action_select_mask,
            goal_pose=goal_pose_cos_sin, 
            delay_pose=goal_pose_cos_sin, #tempolarily delay_pose = goal_pose       
            obj_pose_norm=goal_pose_cos_sin[0:2], 
            img_PIL=cropped_image_fullsize_PIL_r,
            gimg_PIL=cropped_gimage_fullsize_PIL_r,
            p_image=TF.resize(to_tensor(cropped_image_fullsize_PIL_r), (96, 96)),    
            c_image=TF.resize(to_tensor(cropped_cimage_fullsize_PIL_r), (96, 96)),                               
            cur_image = image_obs_r, 
            goal_image_8=goal_image_full_8_r, 
            temp_dist=distance,
            lan_prompt="No language instruction"
        )          
                   
