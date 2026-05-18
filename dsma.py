import torch
import torch.nn as nn
import torch.nn.functional as F

class DSMA(nn.Module):

    def __init__(self, channels):
        super(DSMA, self).__init__()
        self.channels = channels
        
        # 1. dw conv（3×3）
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # branch1：7×7（7×1 + 1×7）
        self.branch7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7,1), padding=(3,0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1,7), padding=(0,3), groups=channels)
        )
        # branch2：11×11（11×1 + 1×11）
        self.branch11 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11,1), padding=(5,0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1,11), padding=(0,5), groups=channels)
        )
        # branch3：13×13
        self.branch13 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(13,1), padding=(6,0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1,13), padding=(0,6), groups=channels)
        )
        

        self.channel_mix = nn.Conv2d(channels, channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        local_feat = self.relu(self.bn(self.local_conv(x)))

        multi_scale = self.branch7(local_feat) + self.branch11(local_feat) + self.branch13(local_feat) + local_feat

        att_weight = self.sigmoid(self.channel_mix(multi_scale))

        return x * att_weight


if __name__ == '__main__':
    print("Testing DSMA module...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    channels = 72  
    dsma = DSMA(channels=channels).to(device)
    
    batch_size = 4
    height, width = 13, 13  
    
    input_tensor = torch.randn(batch_size, channels, height, width).to(device)
    print(f"Input shape: {input_tensor.shape}")
    
    output_tensor = dsma(input_tensor)
    print(f"Output shape: {output_tensor.shape}")
    
    assert input_tensor.shape == output_tensor.shape, "Input and output shapes should be the same"
    print("Test passed: Input and output shapes match!")
    
    print(f"Input range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
    print(f"Output range: [{output_tensor.min().item():.4f}, {output_tensor.max().item():.4f}]")
    print(f"Attention weights range: [{(output_tensor / input_tensor).min().item():.4f}, {(output_tensor / input_tensor).max().item():.4f}]")