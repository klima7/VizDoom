import torch
from torch import nn
from torchinfo import summary


class ConvNetwork(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[2], kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[3], kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Dropout(0.2),
        )

    def forward(self, image):
        return self.net(image)


class DQNNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.screen_net = ConvNetwork([3, 16, 32, 64])

        self.neck_net = nn.Sequential(
            nn.Linear(384 + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward(self, data):
        # summary(self.screen_net, input_data=data['screen'], col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        screen_out = self.screen_net(data['screen'])
        # summary(self.automap_net, input_data=data['automap'], col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        # automap_out = self.automap_net(data['automap'])
        neck_in = torch.cat([screen_out, data['variables']], axis=1)
        # summary(self.neck_net, input_data=neck_in, col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        neck_out = self.neck_net(neck_in)
        return neck_out

    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        # automaps = torch.tensor(state['automap'], device=device, dtype=torch.float32).unsqueeze(0)
        variables = torch.tensor(state['variables'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'variables': variables}
        return self.forward(data)[0]
