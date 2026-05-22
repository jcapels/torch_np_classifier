import torch
import torch.nn as nn
import torch.nn.functional as F


class NPClassifierDNN(nn.Module):
    def __init__(self, num_categories=730, original=False):
        super(NPClassifierDNN, self).__init__()
        self.original = original
        if not self.original:
            self.dense1 = nn.Linear(6144, 2048)
            self.bn1 = nn.BatchNorm1d(2048)
            self.dense2 = nn.Linear(2048, 3072)
            self.bn2 = nn.BatchNorm1d(3072)
            self.dense3 = nn.Linear(3072, 1536)
            self.bn3 = nn.BatchNorm1d(1536)
            self.dense4 = nn.Linear(1536, 1536)
            self.dropout = nn.Dropout(0.2)
            self.output = nn.Linear(1536, num_categories)
        else:
            self.dense1 = nn.Linear(6144, 3072)
            self.bn1 = nn.BatchNorm1d(3072)
            self.dense2 = nn.Linear(3072, 2304)
            self.bn2 = nn.BatchNorm1d(2304)
            self.dense3 = nn.Linear(2304, 1152)
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(1152, num_categories)

    def forward(self, x, return_embedding=False):
        if not self.original:
            x = F.relu(self.dense1(x[0]))
            x = self.bn1(x)
            x = F.relu(self.dense2(x))
            x = self.bn2(x)
            x = F.relu(self.dense3(x))
            x = self.bn3(x)
            x = self.dense4(x)
            x = F.relu(x)
            embedding = torch.clone(x)
            x = self.dropout(embedding)
            x = self.output(x)
        else:
            x = F.relu(self.dense1(x[0]))
            x = self.bn1(x)
            x = F.relu(self.dense2(x))
            x = self.bn2(x)
            x = F.relu(self.dense3(x))
            embedding = torch.clone(x)
            x = self.dropout(x)
            x = self.output(x)

        x = torch.sigmoid(x)
        if return_embedding:
            return x, embedding
        return x
