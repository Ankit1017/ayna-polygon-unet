import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PolygonColorUNet(nn.Module):
    """
    UNet model that takes polygon image and color as input
    and outputs colored polygon image
    """
    def __init__(self, n_colors=10, bilinear=False):
        super(PolygonColorUNet, self).__init__()
        self.n_colors = n_colors
        self.bilinear = bilinear

        # Image encoder path
        self.inc = DoubleConv(1, 64)  # Input: grayscale polygon
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Color embedding
        self.color_embed = nn.Sequential(
            nn.Linear(n_colors, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512)
        )

        # Decoder path with color conditioning
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)  # Output: RGB colored polygon

        # Color conditioning layers (FiLM - Feature-wise Linear Modulation)
        self.film_conv1 = nn.Conv2d(512, 64, 1)
        self.film_conv2 = nn.Conv2d(512, 128, 1)
        self.film_conv3 = nn.Conv2d(512, 256, 1)
        self.film_conv4 = nn.Conv2d(512, 512, 1)

    def apply_film_conditioning(self, features, color_features, film_layer):
        """Apply Feature-wise Linear Modulation"""
        # Reshape color features to match spatial dimensions
        color_spatial = film_layer(color_features)

        # Expand to match feature map size
        _, _, H, W = features.shape
        color_spatial = F.interpolate(color_spatial, size=(H, W), mode='bilinear', align_corners=True)

        # Apply conditioning (gamma * features + beta)
        return features + color_spatial

    def forward(self, x, color_onehot):
        """
        Forward pass
        Args:
            x: Input polygon image (B, 1, H, W)
            color_onehot: One-hot encoded color (B, n_colors)
        Returns:
            Colored polygon image (B, 3, H, W)
        """
        # Encode color
        color_features = self.color_embed(color_onehot)  # (B, 512)
        color_features = color_features.unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)

        # Encoder path
        x1 = self.inc(x)
        x1 = self.apply_film_conditioning(x1, color_features, self.film_conv1)

        x2 = self.down1(x1)
        x2 = self.apply_film_conditioning(x2, color_features, self.film_conv2)

        x3 = self.down2(x2)
        x3 = self.apply_film_conditioning(x3, color_features, self.film_conv3)

        x4 = self.down3(x3)
        x4 = self.apply_film_conditioning(x4, color_features, self.film_conv4)

        x5 = self.down4(x4)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output layer with sigmoid activation for RGB values
        logits = self.outc(x)
        return torch.sigmoid(logits)

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    model = PolygonColorUNet(n_colors=10)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    # Test forward pass
    batch_size = 2
    img = torch.randn(batch_size, 1, 256, 256)
    color = torch.randn(batch_size, 10)

    with torch.no_grad():
        output = model(img, color)
        print(f"Input shape: {img.shape}")
        print(f"Color shape: {color.shape}")
        print(f"Output shape: {output.shape}")
