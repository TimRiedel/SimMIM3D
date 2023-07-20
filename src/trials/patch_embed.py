import torch
import torch.nn as nn

channel_1 = torch.rand(1, 128, 128, 128)
channel_4 = torch.rand(4, 128, 128, 128)

conv_1 = nn.Conv3d(
    in_channels=1, out_channels=768, kernel_size=(16, 16, 16), stride=(16, 16, 16)
)
conv_4 = nn.Conv3d(
    in_channels=4, out_channels=768, kernel_size=(16, 16, 16), stride=(16, 16, 16)
)

print(conv_1(channel_1).shape)
print(conv_4(channel_4).shape)

param_list_1 = list(param for name, param in conv_1.named_parameters())
print(param_list_1[0].shape)

param_list_4 = list(param for name, param in conv_4.named_parameters())
print(param_list_4[0].shape)