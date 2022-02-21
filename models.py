import torch
from torch import nn
import torch.nn.functional as F

# World Model
# NN simulated env
# train to mimic real world env
class WorldModel(nn.Module):
    def __init__(self, in_channel_size, action_size):
        super().__init__()
        # Frame Prediction
        self.input_frame_dense = nn.Linear(in_channel_size, 96)
        self.downscale_conv1 = nn.Conv2d(96, 192, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv2 = nn.Conv2d(192, 384, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv3 = nn.Conv2d(384, 768, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv4 = nn.Conv2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv5 = nn.Conv2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.downscale_conv6 = nn.Conv2d(768, 768, kernel_size=(4, 4), stride=(2, 2))

        self.action_embedding = nn.Linear(action_size, 768)
        self.latent_pred_embedding = nn.Embedding()

        self.middle_conv1 = nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1))
        self.middle_conv2 = nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1))

        self.upscale_trans_conv1 = nn.ConvTranspose2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.upscale_trans_conv2 = nn.ConvTranspose2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.upscale_trans_conv3 = nn.ConvTranspose2d(768, 768, kernel_size=(4, 4), stride=(2, 2))
        self.upscale_trans_conv4 = nn.ConvTranspose2d(768, 384, kernel_size=(4, 4), stride=(2, 2))
        self.upscale_trans_conv5 = nn.ConvTranspose2d(384, 192, kernel_size=(4, 4), stride=(2, 2))
        self.upscale_trans_conv5 = nn.ConvTranspose2d(192, 96, kernel_size=(4, 4), stride=(2, 2))

        self.output_frame_dense = nn.Linear(96, 768)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    
    def forward(self, input_frame, action, target_frame=None):
        pass




# Policy
# PPO
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
