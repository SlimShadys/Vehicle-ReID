import numpy as np
import torch
import torch.nn as nn
from losses.supcon_loss import SupConLoss
from losses.triplet_loss import TripletLoss
from pytorch_metric_learning import distances as py_distances
from pytorch_metric_learning import losses as py_losses
from pytorch_metric_learning import miners as py_miners

class LossBuilder(nn.Module):
    def __init__(self, alpha=0.9, k=10, margin=0.3, label_smoothing=0.1, apply_MALW=False, **kwargs) -> torch.nn.Module:
        super(LossBuilder, self).__init__()
        
        self.iteration = 0
        self.margin = margin                    # Margin for contrastive loss
        self.label_smoothing = label_smoothing  # Label smoothing factor
        self.apply_MALW = apply_MALW            # Apply MALW algorithm

        # ID Loss -> Cross Entropy Loss
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metric Loss -> Triplet Loss
        #self.triplet_loss = TripletLoss(self.margin, mining_method='batch_hard')
        self.triplet_loss = TripletLoss(self.margin)
        
        # Metric Loss -> Supervised Contrastive Loss
        # self.distance = py_distances.CosineSimilarity()
        # self.miner = py_miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="semihard")
        # self.triplet_loss = py_distances.SupConLoss()
        
        # Metric Loss -> TripletMarginLoss
        # self.distance = py_distances.LpDistance()
        # self.miner = py_miners.BatchHardMiner(distance=self.distance)
        # self.triplet_loss = py_distances.TripletMarginLoss(margin=self.margin)

        if(self.apply_MALW):
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
    def forward(self, embeddings: torch.Tensor, pred_ids: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # == Calculate the cross entropy loss (ID Loss)
        ID_loss = self.cross_entropy(pred_ids, target_ids)

        # == Calculate the Supervised Contrastive Loss (Metric Loss)
        metric_loss, _, _ = self.triplet_loss(embeddings, target_ids)
        
        # ONLY TO BE USED WITH MINER + SupConLoss / TripletMarginLoss from Pytorch Metric Learning
        #(a1, p, n) = self.miner(embeddings, target_ids)
        #metric_loss = self.triplet_loss(embeddings, target_ids, (a1, p, n)).mean()

        # embeddings = nn.functional.normalize(embeddings, p=2, dim=1) # Normalize embeddings for the Metric Loss

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
                    
                self.ID_losses.clear()
                self.metric_losses.clear()
            else:
                pass # Keep the previous ratio
                
            # Calculate the weighted sum of losses
            ID_loss = self.ratio_ID * ID_loss
            metric_loss = self.ratio_metric * metric_loss

        self.iteration += 1

        return ID_loss, metric_loss

    def get_weights(self):
        return self.ratio_ID, self.ratio_metric