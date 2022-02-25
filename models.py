import torch
from torch import nn
import torch.nn.functional as F

from utils import to_tensor

# World Model
# NN simulated env
# train to mimic real world env


class DeterministicModel(nn.Module):

    def __init__(self, in_channel_size, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel_size, 64, 4, 2)
        self.conv2 = nn.Conv2d(64, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 2)
        self.convs = [self.conv1, self.conv2, self.conv3]

        self.dense1 = nn.Linear(64 * 11 * 8 + action_size, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.dense3 = nn.Linear(1024, 64 * 11 * 8)
        self.denses = [self.dense1, self.dense2, self.dense3]

        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 4, 2)
        self.deconv3 = nn.ConvTranspose2d(64, 12, 4, 2)
        self.deconvs = [self.deconv1, self.deconv2, self.deconv3]

        self.reward_dense = nn.Linear(64 * 11 * 8, 1)
        # TODO Don't know how to use this last layer
        self.last_deconv = nn.ConvTranspose2d(12, 3, 1, 1)
        # TODO Should the embedding be done by dense layers? The paper said they are one-hot-encoding the action.
        self.action_size = action_size

    def forward(self, state, action, batch_size):
        x = state
        # Convolutional
        conv_shapes = []
        conv_values = []

        for conv_layer in self.convs:
            conv_shapes.append(x.shape)
            conv_values.append(x)
            x = conv_layer(x)
            x = F.relu(F.dropout(F.layer_norm(x, x.shape), p=0.2))

        # Dense
        action_tensor = F.one_hot(action, num_classes=self.action_size)
        x = torch.cat([x.flatten(1), action_tensor], dim=1)
        for dense_layer in self.denses:
            x = dense_layer(x)
            x = F.relu(F.dropout(F.layer_norm(x, x.shape), p=0.2))

        # Reward
        reward = self.reward_dense(x)

        # Deconvolutional
        x = x.reshape([batch_size, 64, 11, 8])
        for deconv_layer in self.deconvs:
            x = deconv_layer(x, conv_shapes.pop())
            x = F.relu(F.dropout(F.layer_norm(x, x.shape), p=0.2))
            x += conv_values.pop()

        x = self.last_deconv(x)
        output_frame = F.relu(x)
        output_frame = output_frame.reshape([batch_size, 105, 80, 3])

        return output_frame, reward

    def get_output_frame_and_reward(self, stacked_image, action, batch_size, device):
        shape = stacked_image.shape
        stacked_image = stacked_image.reshape([shape[0], shape[1] * shape[4], shape[2], shape[3]])
        output_frame, reward = self(stacked_image, action, batch_size)
        return output_frame, reward

# ------------------------------------------------------------------------
# TODO


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
