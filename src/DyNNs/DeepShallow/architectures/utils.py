import torch.nn as nn
import torch


# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        # red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size)  # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels * input_size * input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size / red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels * red_input_size * red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size / 4)
    else:
        return -1
