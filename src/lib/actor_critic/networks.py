import torch
import torch.nn as nn


class ConvNetwork(nn.Module):

    def __init__(self, screen_size, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3),
            nn.BatchNorm2d(channels[1]),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[2], kernel_size=3),
            nn.BatchNorm2d(channels[2]),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[3], kernel_size=3),
            nn.BatchNorm2d(channels[3]),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Flatten(),
        )

    def forward(self, image):
        return self.net(image)


class Actor(nn.Module):
    def __init__(self, n_actions, screen_size, n_variables):
        super().__init__()

        self.screen_net = ConvNetwork(screen_size, [1, 32, 64, 128])

        self.l1 = nn.Linear(768 + n_variables, 512)
        self.l2 = nn.Linear(512 + n_variables, 256)
        self.l3 = nn.Linear(256 + n_variables, n_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        screen = state['screen']
        variables = state['variables']
        out = self.screen_net(screen)

        out = self.l1(torch.cat([out, variables], dim=1))
        out = self.relu(out)

        out = self.l2(torch.cat([out, variables], dim=1))
        out = self.relu(out)

        out = self.l3(torch.cat([out, variables], dim=1))
        out = self.softmax(out)

        return out

    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        variables = torch.tensor(state['variables'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'variables': variables}
        return self.forward(data)[0]


class Critic(nn.Module):
    def __init__(self, screen_size, n_variables):
        super().__init__()

        self.screen_net = ConvNetwork(screen_size, [1, 32, 64, 128])

        self.l1 = nn.Linear(768 + n_variables, 512)
        self.l2 = nn.Linear(512 + n_variables, 256)
        self.l3 = nn.Linear(256 + n_variables, 1)

        self.relu = nn.ReLU()

    def forward(self, state):
        screen = state['screen']
        variables = state['variables']

        out = self.screen_net(screen)

        out = self.l1(torch.cat([out, variables], dim=1))
        out = self.relu(out)

        out = self.l2(torch.cat([out, variables], dim=1))
        out = self.relu(out)

        out = self.l3(torch.cat([out, variables], dim=1))
        return out

    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        variables = torch.tensor(state['variables'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'variables': variables}
        return self.forward(data)[0]
