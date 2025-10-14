
import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out)
        out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_channels, num_classes, num_levels=4, n_channels=128, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_channels
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, n_channels, kernel_size, 1, dilation, (kernel_size-1)*dilation, dropout))
            in_ch = n_channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(n_channels, num_classes))
    def forward(self, x):
        y = self.backbone(x)
        return self.head(y)
