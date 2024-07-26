import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_metric_learning import losses, miners, distances

class LossBuilder(nn.Module):
    def __init__(self, alpha=0.9, k=10, margin=0.3, label_smoothing=0.1, num_classes=1000, **kwargs):
        super(LossBuilder, self).__init__()
        
        self.num_classes = num_classes
        self.iteration = 0
        self.margin = margin  # Margin for contrastive loss

        # ID Loss -> Cross Entropy Loss
        #self.cross_entropy = nn.CrossEntropyLoss()
        self.cross_entropy = CrossEntropyLabelSmooth(num_classes=self.num_classes, epsilon=label_smoothing, use_gpu=True)

        # Metric Loss -> Supervised Contrastive Loss / Contrastive Loss
        self.distance = distances.CosineSimilarity()
        self.miner = miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="semihard")
        self.supconloss = losses.SupConLoss()
        #self.triplet_loss = ContrastiveLoss(self.margin) # ContrastiveLoss(self.margin) / SupConLoss()

        self.apply_MALW = True

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
    # ================================================
    def forward(self, embeddings, pred_ids, target_ids):
        # == Calculate the cross entropy loss (ID Loss)
        ID_loss = self.cross_entropy(pred_ids, target_ids).mean()

        # == Calculate the Supervised Contrastive Loss (Metric Loss)
        embeddings = F.normalize(embeddings, p=2, dim=1) # Normalize embeddings for the Metric Loss
        hard_pairs = self.miner(embeddings, target_ids)
        metric_loss = self.supconloss(embeddings, target_ids, hard_pairs).mean()

        if self.apply_MALW:
            # Apply MALW algorithm
            self.ID_losses.append(ID_loss.detach().cpu())
            self.metric_losses.append(metric_loss.detach().cpu())
            
            # Update the weights every k iterations
            if (self.iteration % self.k == 0) and len(self.ID_losses) > 0:
                ID_std = np.array(self.ID_losses).mean()
                metric_std = np.array(self.metric_losses).mean()
                
                self.ID_losses.clear()
                self.metric_losses.clear()

                if ID_std > metric_std:
                    self.ratio_ID = (self.alpha * self.ratio_ID) + ((1 - (ID_std - metric_std) / ID_std) * 0.1)
            else:
                pass # Keep the previous ratio
                
            # Calculate the weighted sum of losses
            ID_loss = self.ratio_ID * ID_loss
            metric_loss = self.ratio_metric * metric_loss

        self.iteration += 1

        return ID_loss, metric_loss

    def get_weights(self):
        return self.ratio_ID, self.ratio_metric
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        loss = list()
        n = inputs.size(0)

        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            neg_loss = 0.0

            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)

            pos_loss = torch.sum(-pos_pair_ + 1)

            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, num_ids=4, views=8, temperature=0.07, base_temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        self.num_ids = num_ids
        self.views = views

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, features: torch.Tensor, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # Normalize the features through an L2 normalization as per ReadME
        features = F.normalize(features, dim=1)

        # susu
        features = features.view(self.num_ids, self.views, -1)
        labels = labels.view(self.num_ids, self.views)[:,0]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss