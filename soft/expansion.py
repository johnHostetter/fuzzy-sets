from collections import OrderedDict

import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 6, 3, 1, 1, bias=False)),
                    ("conv2", nn.Conv2d(6, 12, 3, 1, 1, bias=False)),
                    ("conv3", nn.Conv2d(12, 24, 3, 1, 1, bias=False)),
                ]
            )
        )
        self.add_layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.ModuleList([])),
                    ("conv2", nn.ModuleList([])),
                    ("conv3", nn.ModuleList([])),
                ]
            )
        )
        self.add_layers_help = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.ModuleList([])),
                    ("conv2", nn.ModuleList([])),
                    ("conv3", nn.ModuleList([])),
                ]
            )
        )

    def expand_layer(self, layer_number, num_filters):
        # Create new conv layer for current layer
        in_channels0 = self.layers[layer_number].in_channels
        if layer_number > 0:
            in_channels0 += sum(
                [
                    add_layer.in_channels
                    for add_layer in self.add_layers_help[layer_number]
                ]
            )
        out_channels0 = num_filters
        kernel_size0 = self.layers[layer_number].kernel_size
        layer0 = nn.Conv2d(
            in_channels0, out_channels0, kernel_size0, stride=1, padding=1, bias=False
        )
        new_idx = len(self.add_layers[layer_number])
        self.add_layers[layer_number].add_module(str(new_idx), layer0)

        # Check if next layer needs modification
        if layer_number < (len(self.layers) - 1):
            in_channels1 = num_filters
            out_channels1 = self.layers[layer_number + 1].out_channels
            kernel_size1 = self.layers[layer_number + 1].kernel_size
            layer1 = nn.Conv2d(
                in_channels1,
                out_channels1,
                kernel_size1,
                stride=1,
                padding=1,
                bias=False,
            )
            new_idx = len(self.add_layers_help[layer_number + 1])
            self.add_layers_help[layer_number + 1].add_module(str(new_idx), layer1)

    def forward(self, x):
        # Iterate all layers
        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            add_layer = self.add_layers[idx]
            add_layer_help = self.add_layers_help[idx]

            # Helper flag
            needs_layer = True

            # Check if "helper" layers are needed .
            # This is the case, if the previous output is bigger now.
            if len(add_layer_help) != 0:
                # Split the tensor so that the original layer and the helper
                # layers can perform the convolution together.
                split_idx = [layer.in_channels] + [
                    l.in_channels for l in add_layer_help
                ]
                x_split = torch.split(x, split_idx, dim=1)

                # Apply the original layer on the first split, since all
                # additional slices are appended at the end.
                x_out = layer(x_split[0])
                needs_layer = False
                # Iterate all helper layers and sum the outputs
                for x_, l in zip(x_split[1:], add_layer_help):
                    x_out = x_out + l(x_)

            # Check if any additional layers are needed
            if len(add_layer) != 0:
                x1 = torch.cat([l(x) for l in add_layer], dim=1)
                if needs_layer:
                    x_out = layer(x)
                    needs_layer = False
                x_out = torch.cat((x_out, x1), dim=1)

            # Check if no op was performed yet
            if needs_layer:
                x_out = layer(x)

            x = x_out

        return x


model = MyModel()
x = torch.randn(1, 3, 4, 4)
output = model(x)
print(output.shape)

model.expand_layer(layer_number=0, num_filters=2)
output = model(x)
print(output.shape)

model.expand_layer(layer_number=1, num_filters=2)
output = model(x)
print(output.shape)

model.expand_layer(layer_number=2, num_filters=2)
output = model(x)
print(output.shape)

# Add another layer
model.expand_layer(layer_number=2, num_filters=2)
output = model(x)
print(output.shape)
