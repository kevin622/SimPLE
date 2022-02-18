import torch
from torch import nn
import torch.nn.functional as F

# World Model
# NN simulated env
# train to mimic real world env
class DeterministicWorldModel(nn.Module):
    def __init__(self, pixel_shape, action_shape):
        super().__init__()
        self.input_frame_dense = nn.Linear(pixel_shape, 96)
        self.downscale_conv1 = nn.Conv2d(96, 192, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv2 = nn.Conv2d(192, 384, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv3 = nn.Conv2d(384, 768, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv4 = nn.Conv2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv5 = nn.Conv2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv6 = nn.Conv2d(768, 768, kernel_size=(4, 4), stride=(2, 2))

        self.action_embedding = nn.Embedding(action_shape, 768)
        



# Policy
# PPO
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
