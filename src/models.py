from torch import nn
import torch.nn.functional as F



class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout()
        # Calculate the output size after max pooling
        self.fc1 = nn.Linear(2740, 128)  
        # self.fc1 = nn.Linear(512, 256)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2740)  # Adjust 4384 to match your specific output size after pooling
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(SimpleMLP, self).__init__()
        self.dim_in = dim_in  # Define dim_in as an instance attribute
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, self.dim_in)  # Access dim_in as an instance attribute
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
        

