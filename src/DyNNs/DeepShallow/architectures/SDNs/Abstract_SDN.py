import torch.nn as nn


class Abstract_SDN(nn.Module):
    def __init__(self, params, train_func, test_func):
        super().__init__()
        self.train_func = train_func
        self.test_func = test_func
        self.params = params

    def adaptive_flops(self):
        pass

    def batch_adaptive_forward(self, x):
        pred_labels, confidences = [], []
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                output = nn.Softmax()(output)
                max_confidence, pred_y = output.max(1)
                confidences.append(max_confidence)
                pred_labels.append(pred_y)
        output = self.end_layers(fwd)
        output = nn.Softmax()(output)
        max_confidence, pred_y = output.max(1)
        confidences.append(max_confidence)
        pred_labels.append(pred_y)
        return confidences, pred_labels

    def single_adaptive_forward(self, x, threshold):
        fwd = self.init_conv(x)
        cnt = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            cnt += 1
            if is_output:
                output = nn.Softmax()(output)
                max_confidence, pred_y = output.max(1)
                if max_confidence > threshold:
                    return cnt
        output = self.end_layers(fwd)
        cnt += 1
        output = nn.Softmax()(output)
        max_confidence, pred_y = output.max(1)
        return cnt