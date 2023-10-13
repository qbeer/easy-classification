import torch
from torch import nn
import timm
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import wandb
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class BackboneClassifier(pl.LightningModule):
    def __init__(self, backbone, n_classes, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.backbone = backbone

        self.linear_head = nn.Sequential(
            nn.Linear(self.backbone.as_sequential()[-4].num_features, n_classes)
        )

        self.accuracy = MulticlassAccuracy(num_classes=n_classes)
        self.auroc = MulticlassAUROC(num_classes=n_classes)
        
        self.val_accuracy = MulticlassAccuracy(num_classes=n_classes)
        self.val_auroc = MulticlassAUROC(num_classes=n_classes)
        
        self.n_classes = n_classes
        
        self.y_buffer = []
        self.y_hat_buffer = []
        self.y_hat_prob_buffer = []

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        pred = self.linear_head(embedding)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).softmax(-1)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        
        acc = self.accuracy(y_hat, y)
        roc = self.auroc(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log('train_accuracy', acc, on_step=True, on_epoch=False)
        self.log('train_auroc', roc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).softmax(-1)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        
        acc = self.val_accuracy(y_hat, y)
        roc = self.val_auroc(y_hat, y)

        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        self.log('valid_accuracy', acc, on_step=False, on_epoch=True)
        self.log('valid_auroc', roc, on_step=False, on_epoch=True)
        
        self.y_buffer.extend(y.cpu().numpy())
        self.y_hat_buffer.extend(y_hat.argmax(-1).cpu().numpy())
        self.y_hat_prob_buffer.extend(y_hat.cpu().numpy())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).softmax(-1)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        cm = confusion_matrix(self.y_buffer, self.y_hat_buffer, labels=np.arange(self.n_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(self.n_classes))
        disp.plot(include_values=False)
        
        fig = disp.figure_
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        del fig
        
        cfm_img = Image.fromarray(data)
        self.logger.log_image('valid_confusion_matrix', [cfm_img], caption=['Validation Confusion Matrix'])
        
        
        # plot ROC curves for each class
        y_probs = np.array(self.y_hat_prob_buffer)
        y_true = np.array(self.y_buffer)
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            ax.plot(fpr, tpr, label=f'Class {i}')
            
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        del fig
        
        img = Image.fromarray(data)
        self.logger.log_image('valid_roc_curves', [img], caption=['Validation ROC curves'])
        
        self.y_buffer = []
        self.y_hat_buffer = []
        self.y_hat_prob_buffer = []

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)