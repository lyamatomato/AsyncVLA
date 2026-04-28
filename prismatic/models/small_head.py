import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import MultiLayerDecoder_trans
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, STOP_INDEX

class Edge_adapter(nn.Module):
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super(Edge_adapter, self).__init__()
        self.obs_encoding_size = obs_encoding_size

        # For small head model
        self.cat_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # context
        self.num_cat_features = self.cat_encoder._fc.in_features
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3) # context
        self.num_obs_features = self.obs_encoder._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        if self.num_cat_features != self.obs_encoding_size:
            self.compress_cat_enc = nn.Linear(self.num_cat_features, self.obs_encoding_size)
        else:
            self.compress_cat_enc = nn.Identity()

        self.decoder = MultiLayerDecoder_trans(
            embed_dim=self.obs_encoding_size,
            seq_len=8+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(self.obs_encoding_size, 256),   
            nn.ReLU(),            
            nn.Linear(256, 128),  
            nn.ReLU(),            
            nn.Linear(128, 64),  
            nn.ReLU(),            
            nn.Linear(64, 8 * 4),         
        )
    def forward(self, obs_img: torch.tensor, past_img: torch.tensor, vla_feature: torch.tensor) -> torch.Tensor:

        # For small head model
        batch_size = obs_img.shape[0]
        cat_img = torch.cat((obs_img, past_img), dim=1)
        cat_encoding = self.cat_encoder.extract_features(cat_img)
        cat_encoding = self.cat_encoder._avg_pooling(cat_encoding)
        if self.cat_encoder._global_params.include_top:
            cat_encoding = cat_encoding.flatten(start_dim=1)
            cat_encoding = self.cat_encoder._dropout(cat_encoding)
        cat_encoding = self.compress_cat_enc(cat_encoding)            

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)            

        tokens = torch.cat((vla_feature, obs_encoding.unsqueeze(1), cat_encoding.unsqueeze(1)), dim=1)
        tokens = self.decoder(tokens)[:,-2:-1,:]

        x = tokens.reshape(tokens.shape[0], -1)
        action_pred = self.action_predictor(x).reshape(batch_size, NUM_ACTIONS_CHUNK, -1)

        return action_pred 

class Edge_adapter_v0(nn.Module):
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super(Edge_adapter_v0, self).__init__()
        self.obs_encoding_size = obs_encoding_size

        # For small head model
        self.cat_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # context
        self.num_cat_features = self.cat_encoder._fc.in_features
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3) # context
        self.num_obs_features = self.obs_encoder._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        if self.num_cat_features != self.obs_encoding_size:
            self.compress_cat_enc = nn.Linear(self.num_cat_features, self.obs_encoding_size)
        else:
            self.compress_cat_enc = nn.Identity()

        self.decoder = MultiLayerDecoder_trans(
            embed_dim=self.obs_encoding_size,
            seq_len=8+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(self.obs_encoding_size+1, 256),   
            nn.ReLU(),            
            nn.Linear(256, 128),  
            nn.ReLU(),            
            nn.Linear(128, 64),  
            nn.ReLU(),            
            nn.Linear(64, 8 * 4),         
        )
    def forward(self, obs_img: torch.tensor, past_img: torch.tensor, vla_feature: torch.tensor, taskid: torch.tensor) -> torch.Tensor:

        # For small head model
        batch_size = obs_img.shape[0]
        cat_img = torch.cat((obs_img, past_img), dim=1)
        cat_encoding = self.cat_encoder.extract_features(cat_img)
        cat_encoding = self.cat_encoder._avg_pooling(cat_encoding)
        if self.cat_encoder._global_params.include_top:
            cat_encoding = cat_encoding.flatten(start_dim=1)
            cat_encoding = self.cat_encoder._dropout(cat_encoding)
        cat_encoding = self.compress_cat_enc(cat_encoding)            

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)            

        tokens = torch.cat((vla_feature, obs_encoding.unsqueeze(1), cat_encoding.unsqueeze(1)), dim=1)
        tokens = self.decoder(tokens)[:,-2:-1,:]
        
        x = torch.cat((tokens.reshape(tokens.shape[0], -1), taskid.unsqueeze(1)), dim=1)
        action_pred = self.action_predictor(x).reshape(batch_size, NUM_ACTIONS_CHUNK, -1)

        return action_pred 

        
class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x        

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)     
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)    
        x = self.fc2(x)  # shape: (batch_size, output_dim)    
        return x

class MLPResNet_idcat(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, taskid):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = torch.cat((x, taskid.unsqueeze(1).unsqueeze(2).repeat(1, NUM_ACTIONS_CHUNK, 1)), axis=2)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim) 
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)    
        x = self.fc2(x)  # shape: (batch_size, output_dim)       
        return x

class Proj_Actiontokens(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        # Split-mode projection path: no explicit task-id conditioning.
        self.model = MLPResNet_idcat(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states, taskid):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states, taskid)
        return action
