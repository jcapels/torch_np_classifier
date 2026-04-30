import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plants_sm.models.lightning_model import InternalLightningModule


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


class NPClassifier(InternalLightningModule):

    def __init__(self, classification_neurons=730, metric=None, scheduler=False, original=False) -> None:
        self.classification_neurons = classification_neurons
        self.scheduler = scheduler
        self.return_embedding = False
        self._contructor_parameters = {}
        self.original = original
        super().__init__(metric=metric)
        self._create_model()

    def _update_constructor_parameters(self):
        self._contructor_parameters.update({
            'classification_neurons': self.classification_neurons,
            'metric': self.metric,
            'scheduler': self.scheduler,
            'original': self.original,
        })

    def _create_model(self):
        self.np_classifier_model = NPClassifierDNN(self.classification_neurons, original=self.original)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.np_classifier_model.parameters()}], lr=0.00001)
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        return self.np_classifier_model(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)

    def predict_step(self, batch):
        if len(batch) == 2:
            inputs, target = batch
        else:
            inputs = batch
        if not isinstance(inputs, list):
            inputs = [inputs]
        return self.np_classifier_model(inputs, return_embedding=self.return_embedding)
