from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import os
import math
import pickle
import CT.Models.misc as misc
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
        lag_embedding = None,
        noise_timestep_embedding = None,
    ):
        '''
        Inputs:
            in_channels (int)              : Number of input channels.
            out_channels (int)             : Number of output channels.
            activation (str)               : Activation function to use.
            norm (bool)                    : Whether to use normalization.
            n_groups (int)                 : Number of groups for group normalization.
            lag_embedding (int)           : Number of dimensions to embed the conditional time step in
            noise_timestep_embedding (int) : Number of dimensions to embed the noise timestep in
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
            
        if lag_embedding:
            self.lag_embedding = lag_embedding
            self.lag_dense1 = nn.Linear(lag_embedding, out_channels)
            self.lag_dense2 = nn.Linear(lag_embedding, out_channels)
        else:
            self.lag_embedding = None

        if noise_timestep_embedding:
            self.noise_timestep_embedding = noise_timestep_embedding
            self.noise_dense1 = nn.Linear(noise_timestep_embedding, out_channels)
            self.noise_dense2 = nn.Linear(noise_timestep_embedding, out_channels)
        else:
            self.noise_timestep_embedding = None

    def forward(self, x: torch.Tensor, delta = None, t = None):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        if self.lag_embedding:
            h += self.lag_dense1(delta)[:, :, None, None]
            h = self.activation(h)
        if self.noise_timestep_embedding:
            h += self.noise_dense1(t)[:, :, None, None]
            h = self.activation(h)
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        if self.lag_embedding:
            h += self.lag_dense2(delta)[:, :, None, None]
            h = self.activation(h)
        if self.noise_timestep_embedding:
            h += self.noise_dense2(t)[:, :, None, None]
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
        lag_embedding = None,
        noise_timestep_embedding = None,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation = activation, norm = norm, lag_embedding = lag_embedding, noise_timestep_embedding = noise_timestep_embedding)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, delta=None, t=None):
        x = self.res(x,delta,t)
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

    def __init__(self, n_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False, lag_embedding = None, noise_timestep_embedding = None):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm, lag_embedding = lag_embedding, noise_timestep_embedding = noise_timestep_embedding)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm, lag_embedding = lag_embedding, noise_timestep_embedding = noise_timestep_embedding)

    def forward(self, x: torch.Tensor, delta = None, t = None):
        x = self.res1(x,delta,t)
        x = self.attn(x)
        x = self.res2(x,delta,t)
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
        lag_embedding = None,
        noise_timestep_embedding = None,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, activation=activation, norm=norm, lag_embedding = lag_embedding, noise_timestep_embedding = noise_timestep_embedding)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, delta = None, t=None):
        x = self.res(x, delta, t)
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

    def forward(self, x: torch.Tensor, delta = None, t = None):
        return self.conv(x)

class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1), padding_mode="circular")

    def forward(self, x: torch.Tensor, delta = None, t = None):
        return self.conv(x)
        
class RegressorBlock(nn.Module):
    def __init__(self, mid_channels: int, mid_size: int, mlp_dim: int=256, out_dim: int=1,
                         mlp_max: int=4096, activation: str = "gelu", norm: bool = False):
        """ Block to take the encoded middle layer, run through a 1x1 conv,
            then vectorise to an MLP to produce a scalar output. We will use this
            to try and predict the noise level in the fields.
            
            mid_channels: number of channels in the middle layer - used to define 1x1 conv
            mid_size: image size in the middle layer - used to define MLP layer size
            """
        
        super().__init__()

        ## Set number of channels to compress intermediate layer to
        self.intermediate_filters=round(mlp_max/mid_size**2)
        self.vector_size=(mid_size**2)*self.intermediate_filters
        
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        self.conv1=nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.conv2=nn.Conv2d(mid_channels, self.intermediate_filters, kernel_size=(1, 1), padding_mode="circular")
        self.linear1=nn.Linear(self.vector_size, out_dim)
        self.linear2=nn.Linear(out_dim, out_dim)

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self,x):
        """ Forward pass - run some convolutions, then vectorise to MLP and
            produce scalar output """
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.conv2(self.activation(self.norm2(x)))
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


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
        self.lag_embedding = self.config.get("lag_embedding")
        if self.lag_embedding:
            self.lag_mlp1=nn.Linear(self.lag_embedding,self.lag_embedding)
            self.lag_mlp2=nn.Linear(self.lag_embedding,self.lag_embedding)
        self.noise_timestep_embedding = self.config.get("noise_timestep_embedding")
        if self.noise_timestep_embedding:
            self.noise_timestep_mlp1=nn.Linear(self.noise_timestep_embedding,self.noise_timestep_embedding)
            self.noise_timestep_mlp2=nn.Linear(self.noise_timestep_embedding,self.noise_timestep_embedding)

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
                        lag_embedding=self.lag_embedding,
                        noise_timestep_embedding=self.noise_timestep_embedding
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
                                  norm=self.normBool, lag_embedding=self.lag_embedding,noise_timestep_embedding=self.noise_timestep_embedding)

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
                        lag_embedding=self.lag_embedding,
                        noise_timestep_embedding=self.noise_timestep_embedding
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // self.config["dim_mults"][i]
            up.append(UpBlock(in_channels, out_channels, has_attn=self.is_attn[i], activation=self.config["activation"],
                              norm=self.normBool, lag_embedding=self.lag_embedding, noise_timestep_embedding=self.noise_timestep_embedding))
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

    def forward(self, x: torch.Tensor, delta = None, t = None):
        x = self.image_proj(x)
        if self.lag_embedding:
            delta = misc.get_timestep_embedding(delta, self.lag_embedding)
            delta = self.activation(self.lag_mlp1(delta))
            delta = self.activation(self.lag_mlp2(delta))
        if self.noise_timestep_embedding:
            t = misc.get_timestep_embedding(t, self.noise_timestep_embedding)
            t = self.activation(self.noise_timestep_mlp1(t))
            t = self.activation(self.noise_timestep_mlp2(t))

        h = [x]
        for m in self.down:
            x = m(x,delta,t)
            h.append(x)

        x = self.middle(x,delta,t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x,delta,t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #   
                x = m(x,delta,t)

        x = self.final(self.activation(self.norm(x)))
        return x

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


class ModernUnetRegressor(ModernUnet):
    """ Inherit the Modern Unet, but add a scalar output
        to perform regression, by overriding forward method """
    def __init__(self,config):
        super().__init__(config)
        self.config["model_type"]="ModernUnetRegressor"
        self.mid_pix=int(self.config["image_size"]/(2*(len(self.config["dim_mults"])-1)))
        self.mid_channels=self.config["hidden_channels"]*math.prod(self.config["dim_mults"])
        self.out_features=self.config["timesteps"]
        self.regressor_block=RegressorBlock(self.mid_channels,self.mid_pix,out_dim=self.out_features)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, delta = None, regression_output: bool=False):
        """ Forward pass - add a flag to optionally return the regression output, as this
            is not always needed """
        x = self.image_proj(x)
        if self.lag_embedding:
            delta = misc.get_timestep_embedding(delta, self.lag_embedding)
            delta = self.activation(self.lag_mlp1(delta))
            delta = self.activation(self.lag_mlp2(delta))

        h = [x]
        for m in self.down:
            x = m(x,delta)
            h.append(x)

        x = self.middle(x,delta)
        if regression_output:
            y = self.regressor_block(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x,delta)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x,delta)

        x = self.final(self.activation(self.norm(x)))
        if regression_output:
            return x, y
        else:
            return x

    @torch.no_grad()
    def noise_class_distribution(self, x: torch.Tensor, delta = None):
        """ Forward pass with only a regressor output - skipping the conv upsampling output for the
            score field prediction. Return the full predicted categorical distribution """

        x = self.image_proj(x)
        if self.lag_embedding:
            delta = misc.get_timestep_embedding(delta, self.lag_embedding)
            delta = self.activation(self.lag_mlp1(delta))
            delta = self.activation(self.lag_mlp2(delta))
        for m in self.down:
            x = m(x,delta)
        x = self.middle(x,delta)
        x = self.softmax(self.regressor_block(x))
        return x

    @torch.no_grad()
    def noise_class(self, x: torch.Tensor, delta = None):
        """ Get predicted noise class  """

        x = self.noise_class_distribution(x, delta)
        return x.argmax(1)