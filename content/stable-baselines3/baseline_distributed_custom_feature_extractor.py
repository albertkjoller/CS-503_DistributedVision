from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from gym.spaces import Space, Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class SiameseCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        actor_features_dim: int = 128,  # feature dimension will be 3 times that
        image_shape: Tuple[int, int] = (160, 120),
        **kwargs
    ):
        super().__init__(observation_space, 3 * actor_features_dim)
        self.actor_observation_space: Space = Box(
            low=0,
            high=255,
            shape=(4, image_shape[0], image_shape[1]),
            dtype=np.uint8
        )

        self.cnn = NatureCNN(observation_space, actor_features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actor1_encoding = self._process_actor_image(observations, 0, 3)
        actor2_encoding = self._process_actor_image(observations, 3, 6)
        actor3_encoding = self._process_actor_image(observations, 6, 9)

        image_encoding = torch.cat(
            [actor1_encoding, actor2_encoding, actor3_encoding],
            dim=1
        )
        return image_encoding

    def _process_actor_image(self, observations, start_index, end_index) -> torch.Tensor:
        actor_image = observations[:, start_index:end_index, :, :]
        occupancy_map = observations[:, -1:, :, :]
        actor_input = torch.cat([actor_image, occupancy_map], dim=1)
        actor_features = self.cnn(actor_input).reshape(observations.shape[0], -1)
        return actor_features


class SiameseResNetCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        actor_features_dim: int = 128,  # feature dimension will be 3 times that
        image_shape: Tuple[int, int] = (160, 120),
        **kwargs
    ):
        super().__init__(observation_space, 3 * actor_features_dim)
        self.resnet = models.resnet18(pretrained=False)

        # We have smaller (and 4 channel) images
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(5, 5), stride=(1, 1), padding=(3, 3), bias=False)
        
        cnn_output_dim = self._get_cnn_output_shape(image_shape)
        num_output_features = actor_features_dim
        print("CNN Output Dimension", cnn_output_dim)
        # Get shape of output of 
        self.resnet.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, num_output_features),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actor1_encoding = self._process_actor_image(observations, 0, 3)
        actor2_encoding = self._process_actor_image(observations, 3, 6)
        actor3_encoding = self._process_actor_image(observations, 6, 9)

        image_encoding = torch.cat(
            [actor1_encoding, actor2_encoding, actor3_encoding],
            dim=1
        )
        return image_encoding

    def _process_actor_image(self, observations, start_index, end_index) -> torch.Tensor:
        actor_image = observations[:, start_index:end_index, :, :]
        occupancy_map = observations[:, -1:, :, :]
        actor_input = torch.cat([actor_image, occupancy_map], dim=1)
        actor_features = self.resnet(actor_input).reshape(observations.shape[0], -1)
        return actor_features

    def _get_cnn_output_shape(self, image_input_shape: Tuple[int, int]) -> int:
        with torch.no_grad():
            input_image = torch.zeros((1, 4, image_input_shape[0], image_input_shape[1]))
            output_image = self.resnet.avgpool(
                self.resnet.layer4(
                    self.resnet.layer3(
                        self.resnet.layer2(
                            self.resnet.layer1(
                                self.resnet.maxpool(
                                    self.resnet.conv1(input_image)
                                )
                            )
                        )
                    )
                )
            )
            output_neurons = len(output_image.reshape(-1))
        
        return output_neurons
