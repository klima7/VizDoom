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

            nn.Conv2d(channels[3], channels[4], kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Dropout(0.5),
        )

    def forward(self, image):
        return self.net(image)


class NeckNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.l1 = nn.Linear(3072, 1024)
        self.d1 = nn.Dropout(0.5)
        self.r1 = nn.ReLU()

        self.l2 = nn.Linear(1024+2, 512)
        self.d2 = nn.Dropout(0.5)
        self.r2 = nn.ReLU()

        self.l3 = nn.Linear(512, n_actions)

    def forward(self, screen, variables):
        out = self.l1(screen)
        out = self.d1(out)
        out = self.r1(out)

        out = torch.cat([out, variables], dim=1)

        out = self.l2(out)
        out = self.d2(out)
        out = self.r2(out)

        out = self.l3(out)

        return out


class DQNNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.screen_net = ConvNetwork([3, 64, 128, 256, 1024])
        self.neck_net = NeckNetwork(n_actions)

    def forward(self, data):
        # summary(self.screen_net, input_data=data['screen'], col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        screen_out = self.screen_net(data['screen'])
        # summary(self.automap_net, input_data=data['automap'], col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        # automap_out = self.automap_net(data['automap'])
        # neck_in = torch.cat([screen_out, data['variables']], axis=1)
        # summary(self.neck_net, input_data=(screen_out, data['variables']), col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        neck_out = self.neck_net(screen_out, data['variables'])
        return neck_out

    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        # automaps = torch.tensor(state['automap'], device=device, dtype=torch.float32).unsqueeze(0)
        variables = torch.tensor(state['variables'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'variables': variables}
        return self.forward(data)[0]
