import torch
from torch import nn
from torchinfo import summary


class ConvNetwork(nn.Module):

    def __init__(self, screen_size, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),

            nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(),

            nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1),
            nn.BatchNorm2d(channels[5]),
            nn.ReLU(),

            nn.Flatten(),
        )

    def forward(self, image):
        return self.net(image)


class NeckNetwork(nn.Module):

    def __init__(self, n_actions, n_variables):
        super().__init__()

        self.l1 = nn.Linear(1280+n_variables, 256)
        self.r1 = nn.ReLU()

        self.l2 = nn.Linear(256+n_variables, 64)
        self.r2 = nn.ReLU()

        self.l3 = nn.Linear(64+n_variables, n_actions)

    def forward(self, screen, variables):
        out = self.l1(torch.cat([screen, variables], dim=1))
        out = self.r1(out)

        out = self.l2(torch.cat([out, variables], dim=1))
        out = self.r2(out)

        out = self.l3(torch.cat([out, variables], dim=1))
        return out


class DQNNetwork(nn.Module):

    def __init__(self, n_actions, screen_size, n_variables):
        super().__init__()

        self.screen_net = self.screen_net = ConvNetwork(screen_size, [5, 16, 32, 64, 64, 128])
        self.neck_net = NeckNetwork(n_actions, n_variables)

    def forward(self, data):
        # summary(self.screen_net, input_data=data['screen'], col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        screen_out = self.screen_net(data['screen'])
        # summary(self.neck_net, input_data=(screen_out, data['variables']), col_names=['input_size', 'output_size', 'num_params', 'params_percent'])
        neck_out = self.neck_net(screen_out, data['variables'])
        return neck_out

    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        variables = torch.tensor(state['variables'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'variables': variables}
        return self.forward(data)[0]
