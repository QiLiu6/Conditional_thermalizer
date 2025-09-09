from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import os
import math
import pickle

import sys

import Flow_matching.Models.misc as misc
## Adapted from the beautiful repo at https://github.com/pdearena/pdearena/blob/main/pdearena/modules/twod_unet.py

class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        rollout_step_embedding = None,
        ratio_embedding = None,
    ):
        '''
        Inputs:
            in_channels (int)              : Number of input channels.
            out_channels (int)             : Number of output channels.
            activation (str)               : Activation function to use.
            norm (bool)                    : Whether to use normalization.
            n_groups (int)                 : Number of groups for group normalization.
            rollout_step_embedding (int) : Number of dimensions to embed rollout time in
            ratio_embedding (int)           : Number of dimensions to embed the ratio in
        '''
        super().__init__()
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding_mode="circular")
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if rollout_step_embedding:
            self.rollout_step_embedding = rollout_step_embedding
            self.t_dense1 = nn.Linear(rollout_step_embedding, out_channels)
            self.t_dense2 = nn.Linear(rollout_step_embedding, out_channels)
        else:
            self.rollout_step_embedding = None
            
        if ratio_embedding:
            self.ratio_embedding = ratio_embedding
            self.s_dense1 = nn.Linear(ratio_embedding, out_channels)
            self.s_dense2 = nn.Linear(ratio_embedding, out_channels)
        else:
            self.ratio_embedding = None

        
    def forward(self, x: torch.Tensor, t = None, s = None):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        if self.rollout_step_embedding:
            h += self.t_dense1(t)[:, :, None, None]
            h = self.activation(h)
        if self.ratio_embedding:
            h += self.s_dense1(s)[:, :, None, None]
            h = self.activation(h)
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        if self.rollout_step_embedding:
            h += self.t_dense2(t)[:, :, None, None]
            h = self.activation(h)
        if self.ratio_embedding:
            h += self.s_dense2(s)[:, :, None, None]
            h = self.activation(h)
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        rollout_step_embedding = None,
        ratio_embedding = None,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation = activation, norm = norm, ratio_embedding = ratio_embedding, rollout_step_embedding = rollout_step_embedding)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t=None, s=None):
        x = self.res(x,t,s)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False, ratio_embedding = None, rollout_step_embedding = None):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm, ratio_embedding = ratio_embedding, rollout_step_embedding = rollout_step_embedding)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm, ratio_embedding = ratio_embedding, rollout_step_embedding = rollout_step_embedding)

    def forward(self, x: torch.Tensor, t = None, s = None):
        x = self.res1(x,t,s)
        x = self.attn(x)
        x = self.res2(x,t,s)
        return x

class UpBlock(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        rollout_step_embedding = None,
        ratio_embedding = None,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, activation=activation, norm=norm, ratio_embedding = ratio_embedding, rollout_step_embedding = rollout_step_embedding)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t = None, s=None):
        x = self.res(x, t, s)
        x = self.attn(x)
        return x

class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t = None, s = None):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1), padding_mode="circular")

    def forward(self, x: torch.Tensor, t = None, s = None):
        return self.conv(x)


class ModernUnet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Config keys:
        n_input_scalar_components (int): Number of scalar components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
    """

    def __init__(self,config) -> None:
        super().__init__()
        self.config=config
        self.config["model_type"]="ModernUnet"
        self.n_input_scalar_components = self.config["input_channels"]
        self.n_output_scalar_components = self.config["output_channels"]
        self.hidden_channels = self.config["hidden_channels"]
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(self.config["activation"], None)
        n_resolutions = len(self.config["dim_mults"])
        n_channels = self.config["hidden_channels"]
        
        insize = self.n_input_scalar_components
        # Project image into feature map
        self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.normBool=self.config["norm"]
        ## Define remaining stuff
        self.config["mid_attn"]=False
        self.mid_attn=self.config["mid_attn"]
        self.config["n_blocks"]=2
        self.is_attn=(False, False, False, False)
        
        self.rollout_step_embedding = self.config.get("rollout_step_embedding")
        if self.rollout_step_embedding:
            self.t_mlp1=nn.Linear(self.rollout_step_embedding,self.rollout_step_embedding)
            self.t_mlp2=nn.Linear(self.rollout_step_embedding,self.rollout_step_embedding)

        self.ratio_embedding = self.config.get("ratio_embedding")
        if self.ratio_embedding:
            self.s_mlp1=nn.Linear(self.ratio_embedding,self.ratio_embedding)
            self.s_mlp2=nn.Linear(self.ratio_embedding,self.ratio_embedding)
      
        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * self.config["dim_mults"][i]
            # Add `n_blocks`
            for _ in range(self.config["n_blocks"]):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        rollout_step_embedding=self.rollout_step_embedding,
                        ratio_embedding=self.ratio_embedding,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, has_attn=self.mid_attn, activation=self.config["activation"],
                                  norm=self.normBool, ratio_embedding=self.ratio_embedding,rollout_step_embedding=self.rollout_step_embedding)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(self.config["n_blocks"]):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        ratio_embedding=self.ratio_embedding,
                        rollout_step_embedding=self.rollout_step_embedding
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // self.config["dim_mults"][i]
            up.append(UpBlock(in_channels, out_channels, has_attn=self.is_attn[i], activation=self.config["activation"],
                              norm=self.normBool, ratio_embedding=self.ratio_embedding, rollout_step_embedding=self.rollout_step_embedding))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if self.normBool:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.n_output_scalar_components
        #
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                            padding=(1, 1), padding_mode="circular")

    def forward(self, x: torch.Tensor, t = None, s = None):
        x = self.image_proj(x)
        if self.ratio_embedding:
            s = misc.get_timestep_embedding(s, self.ratio_embedding)
            s = self.activation(self.s_mlp1(s))
            s = self.activation(self.s_mlp2(s))
        if self.rollout_step_embedding:
            t = misc.get_timestep_embedding(t, self.rollout_step_embedding)
            t = self.activation(self.t_mlp1(t))
            t = self.activation(self.t_mlp2(t))

        h = [x]
        for m in self.down:
            x = m(x,t,s)
            h.append(x)

        x = self.middle(x,t,s)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x,t,s)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                skip = h.pop()
                x = torch.cat((x, skip), dim=1)
                #   
                x = m(x,t,s)

        x = self.final(self.activation(self.norm(x)))
        return x

    def step(self, x_t: torch.Tensor, t : torch.Tensor, s_start: torch.Tensor, s_end: torch.Tensor) -> torch.Tensor:
        s_start = s_start.view(1, 1, 1, 1).expand_as(x_t) 
        return x_t + (s_end - s_start) * self(x_t + self(x_t, t, s_start) * (s_end - s_start) / 2, t, s_start + (s_end - s_start) / 2)

    def save_model(self):
        """ Save the model config, and optimised weights and biases. We create a dictionary
        to hold these two sub-dictionaries, and save it as a pickle file """
        if self.config["save_path"] is None:
            print("No save path provided, not saving")
            return
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        with open(save_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved as %s" % save_string)
        return
