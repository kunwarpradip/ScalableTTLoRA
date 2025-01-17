import time
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F

''' CustomLightningModule class is a custom implementation of a PyTorch Lightning module for training models'''
class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, data, model_learning_rate):
        super().__init__()
        self.model_learning_rate = model_learning_rate
        self.model = model
        self.data = data
        self.val_f1= torchmetrics.F1Score(task="multiclass",num_classes=2, average = 'micro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass",num_classes=2, average = 'micro')

        if self.data == "mnli":
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)

        if self.data == "stsb":
            self.val_acc = torchmetrics.SpearmanCorrCoef()
            self.test_acc = torchmetrics.SpearmanCorrCoef()

        if self.data == "mrpc" or self.data == "qqp":
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)

        if self.data == "cola":
            self.val_acc = torchmetrics.MatthewsCorrCoef(task="multiclass",num_classes=2)
            self.test_acc = torchmetrics.MatthewsCorrCoef(task="multiclass",num_classes=2)
        else:
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],labels=batch["label"])
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training


    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],labels=batch["label"])
        self.log("val_loss", outputs["loss"], prog_bar=True)
        logits = outputs["logits"]
        if self.data == "stsb":
            predicted_labels = logits.squeeze(-1) 
        else:
            predicted_labels = torch.argmax(logits, 1)
        if self.data == "mrpc" or "qqp":
            acc=self.val_acc(predicted_labels, batch["label"])
            f1=self.val_f1(predicted_labels, batch["label"])
            final=(acc+f1)/2
            self.log("val_acc", final, prog_bar=True)
        else:
            self.val_acc(predicted_labels, batch["label"])
            self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],labels=batch["label"])
        logits = outputs["logits"]
        if self.data == "stsb":
            predicted_labels = logits.squeeze(-1) 
        else:
            predicted_labels = torch.argmax(logits, 1)
        if self.data == "mrpc" or "qqp":
            acc = self.test_acc(predicted_labels, batch["label"])
            f1 = self.test_f1(predicted_labels, batch["label"])
            final=(acc+f1)/2
            self.log("accuracy", final, prog_bar=True)
        else:
            self.test_acc(predicted_labels, batch["label"])
            self.log("accuracy", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_learning_rate)
        return optimizer
