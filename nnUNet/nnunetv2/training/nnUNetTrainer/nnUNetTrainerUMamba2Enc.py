from typing import Tuple, Union, List
import pydoc
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.nets.UMamba2Enc_2d import UMamba2Enc_2d
from nnunetv2.nets.UMamba2Enc_3d import UMamba2Enc_3d
# import os
# os.environ['nnUNet_compile'] = '0'

class UMamba2EncTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 150
        self.save_every = 5

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs.get(ri, None) is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None:
            architecture_kwargs["deep_supervision"] = enable_deep_supervision
        
        if len(architecture_kwargs["kernel_sizes"][0]) == 2:
            model = UMamba2Enc_2d(
                input_channels = num_input_channels,
                num_classes = num_output_channels,
                **architecture_kwargs
            )
        elif len(architecture_kwargs["kernel_sizes"][0]) == 3:
            model = UMamba2Enc_3d(
                input_channels = num_input_channels,
                num_classes = num_output_channels,
                **architecture_kwargs
            )
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        model.apply(InitWeights_He(1e-2))
        
        # print("UMamba2Enc: {}".format(model))
        return model