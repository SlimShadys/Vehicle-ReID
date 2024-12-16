import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import distances as py_distances
from pytorch_metric_learning import losses as py_losses
from pytorch_metric_learning import miners as py_miners

from reid.losses.supcon_loss import SupConLoss
from reid.losses.triplet_loss import TripletLoss, TripletLossRPTM

class LossBuilder(nn.Module):
    def __init__(self, loss_type="TripletLoss", alpha=0.9, k=10, margin=0.3, label_smoothing=0.1, apply_MALW=False, batch_size=20, use_amp=False) -> torch.nn.Module:
        super(LossBuilder, self).__init__()

        self.iteration = 0
        self.margin = margin                    # Margin for contrastive loss
        self.label_smoothing = label_smoothing  # Label smoothing factor
        self.apply_MALW = apply_MALW            # Apply MALW algorithm
        self.loss_type = loss_type              # Loss type (TripletLoss, TripletLossRPTM, SupConLoss, SupConLoss_Pytorch, TripletMarginLoss_Pytorch)
        self.batch_size = batch_size            # Batch size
        self.use_amp = use_amp                  # Use Automatic Mixed Precision

        ## ============ ID Loss ============ ##
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # Cross Entropy Loss

        ## =========== Metric Loss =========== ##
        if self.loss_type == "TripletLoss":  # Triplet Loss
            self.triplet_loss = TripletLoss(self.margin, self.use_amp)
        elif self.loss_type == "TripletLossRPTM":
            self.triplet_loss = TripletLossRPTM(self.margin, self.use_amp)
        elif self.loss_type == "SupConLoss":  # Supervised Contrastive Loss from SupConLoss
            self.triplet_loss = SupConLoss(num_ids=6, views=6, contrast_mode='all')
        elif self.loss_type == "SupConLoss_Pytorch":  # Supervised Contrastive Loss from Pytorch Metric Learning
            self.distance = py_distances.CosineSimilarity()
            self.miner = py_miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="semihard")
            self.triplet_loss = py_losses.SupConLoss()
        elif self.loss_type == "TripletMarginLoss_Pytorch":  # Triplet Margin Loss from Pytorch Metric Learning
            self.distance = py_distances.LpDistance()
            self.miner = py_miners.BatchHardMiner(distance=self.distance)
            self.triplet_loss = py_losses.TripletMarginLoss(margin=self.margin)
        else:
            raise ValueError("Invalid loss type. Choose between 'TripletLoss', 'SupConLoss', 'SupConLoss_Pytorch' or 'TripletMarginLoss_Pytorch'")

        if (self.apply_MALW):
            self.alpha = alpha
            self.k = k
            self.ratio_ID = 1.0
            self.ratio_metric = 1.0

            self.ID_losses = []
            self.metric_losses = []

    # ================================================
    # Apply the MALW algorithm (Mean Adaptive Loss Weighting)
    #
    # Paper: https://arxiv.org/pdf/2104.10850
    # Github: https://github.com/cybercore-co-ltd/track2_aicity_2021/blob/master/lib/layers/build.py#L42-L70
    # ================================================
    def forward(self, embeddings: torch.Tensor, pred_ids: torch.Tensor, target_ids: torch.Tensor, normalize: bool = False, use_rptm: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # If we are using RPTM, we only take the first (batch_size) elements
        if use_rptm:
            pred_ids = pred_ids[0:self.batch_size]
            target_ids_ce = target_ids[0:self.batch_size]
        else:
            target_ids_ce = target_ids

        # == Calculate the ID Loss (CrossEntropy)
        ID_loss = self.cross_entropy(pred_ids, target_ids_ce)

        # == Calculate the Metric Loss (Triplet Loss)
        if self.loss_type == "SupConLoss_Pytorch" or self.loss_type == "TripletMarginLoss_Pytorch":
            # In this case, we need to pass the target_ids to the miner so that it can mine the triplets
            metric_loss = self.triplet_loss(embeddings, target_ids, self.miner(embeddings, target_ids)).mean()
        elif self.loss_type == "SupConLoss":
            metric_loss = self.triplet_loss(embeddings, target_ids)
        elif self.loss_type == "TripletLoss" or self.loss_type == "TripletLossRPTM":
            metric_loss, _, _ = self.triplet_loss(embeddings, target_ids)
        else:
            raise ValueError("Invalid loss type. Choose between 'TripletLoss', 'TripletLossRPTM', 'SupConLoss', 'SupConLoss_Pytorch' or 'TripletMarginLoss_Pytorch'")

        # Normalize embeddings for the Metric Loss
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if self.apply_MALW:
            # Apply MALW algorithm
            self.ID_losses.append(ID_loss.detach().cpu())
            self.metric_losses.append(metric_loss.detach().cpu())

            # Update the weights every k iterations
            if (len(self.ID_losses) % self.k == 0) and len(self.ID_losses) > 0:
                ID_std = np.array(self.ID_losses).std()
                metric_std = np.array(self.metric_losses).std()

                if ID_std > metric_std:
                    self.ratio_ID = (self.alpha * self.ratio_ID) + ((1 - (ID_std - metric_std) / ID_std) * 0.1)

                self.ID_losses = []
                self.metric_losses = []
            else:
                pass  # Keep the previous ratio

            # Calculate the weighted sum of losses
            ID_loss = self.ratio_ID * ID_loss
            metric_loss = self.ratio_metric * metric_loss

        self.iteration += 1

        return ID_loss, metric_loss

    def get_weights(self):
        return self.ratio_ID, self.ratio_metric