import torch
import torch.nn as nn


class PhosphoPredictorCNN(nn.Module):
    def __init__(self, num_amino_acids, sequence_length, kernel_size=5):
        super(PhosphoPredictorCNN, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for 'same' padding to be perfectly symmetric.")

        self.conv1 = nn.Conv1d(in_channels=num_amino_acids, out_channels=64, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding='same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        l_after_conv1 = sequence_length
        l_after_pool1 = int(torch.floor((torch.tensor(l_after_conv1, dtype=torch.float32) - self.pool1.kernel_size) / self.pool1.stride + 1).item())

        l_after_conv2 = l_after_pool1
        l_after_pool2 = int(torch.floor((torch.tensor(l_after_conv2, dtype=torch.float32) - self.pool2.kernel_size) / self.pool2.stride + 1).item())

        l_after_conv3 = l_after_pool2
        l_after_pool3 = int(torch.floor((torch.tensor(l_after_conv3, dtype=torch.float32) - self.pool3.kernel_size) / self.pool3.stride + 1).item())

        self.flattened_size = 256 * l_after_pool3

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.5)
        self.output_layer = nn.Linear(64, 1)

    def forward(self,x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout_fc(x)

        x = self.output_layer(x)
        return x


