import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(LinearModel, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x, req_feat=False):
        feature = self.backbone(x)
        out = self.linear(feature)
        if req_feat:
            return feature, out
        else:
            return out

    def update_encoder(self, backbone):
        self.backbone = backbone
