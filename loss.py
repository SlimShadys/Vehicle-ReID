import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import distances, losses, miners
from utils import euclidean_dist, normalize

class LossBuilder(nn.Module):
    def __init__(self, alpha=0.9, k=10, margin=0.3, label_smoothing=0.1, apply_MALW=False, **kwargs) -> torch.nn.Module:
        super(LossBuilder, self).__init__()
        
        self.iteration = 0
        self.margin = margin                    # Margin for contrastive loss
        self.label_smoothing = label_smoothing  # Label smoothing factor

        # ID Loss -> Cross Entropy Loss
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Metric Loss -> Supervised Contrastive Loss
        # self.distance = distances.CosineSimilarity()
        # self.miner = miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="semihard")
        # self.triplet_loss = losses.SupConLoss()
        
        # Metric Loss -> TripletMarginLoss
        # self.distance = distances.LpDistance()
        # self.miner = miners.BatchHardMiner(distance=self.distance)
        # self.triplet_loss = losses.TripletMarginLoss(margin=self.margin)
        
        self.triplet_loss = TripletLoss(self.margin)
                
        self.apply_MALW = apply_MALW

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
        # embeddings = nn.functional.normalize(embeddings, p=2, dim=1) # Normalize embeddings for the Metric Loss
        
        #(a1, p, n) = self.miner(embeddings, target_ids)
        #metric_loss = self.triplet_loss(embeddings, target_ids, (a1, p, n)).mean()
        
        metric_loss = self.triplet_loss(embeddings, target_ids)
        
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

    def forward(self, features: torch.Tensor, labels = None, mask = None):
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

        # # Normalize the features through an L2 normalization as per ReadME
        # features = F.normalize(features, dim=1)

        # susu
        features = features.view(self.num_ids, self.views, -1)
        labels = labels.view(self.num_ids, self.views)[:, 0]
        
        if(labels.shape[0] != features.shape[0]):
            raise ValueError('Num of labels does not match num of features')

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

        # Compute mean of log-likelihood over positive
        # Modified to handle edge cases when there is no positive pair for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`
        Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        """

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        
        N = dist_mat.size(0)
        dist_ap, dist_an = [], []

        # Shape [N, N]
        mask = labels.expand(N, N).eq(labels.expand(N, N).t())
        
        # 'dist_ap' means distance(anchor, positive)
        # 'dist_an' means distance(anchor, negative)
        # Distances will have shape [N, 1]
        for i in range(N):
            negative_mask = dist_mat[i][mask[i] == 0]
            positive_mask = dist_mat[i][mask[i]]
            
            if(len(positive_mask) == 0):
                dist_ap.append(torch.tensor([0.0]))
                print('WARNING: No positive pair found for anchor point')
            elif(len(negative_mask) == 0):
                dist_an.append(torch.tensor([0.0]))
                print('WARNING: No negative pair found for anchor point')
            else:
                dist_ap.append(positive_mask.max().unsqueeze(0))
                dist_an.append(negative_mask.min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap) # Shape: [N]
        dist_an = torch.cat(dist_an) # Shape: [N]
        
        # Retrieve the indices of the positive and negative pairs | To be tested
        if return_inds:
            p_inds = torch.argmax(dist_mat + mask.float() * 1e6, dim=1)
            n_inds = torch.argmin(dist_mat + mask.float() * 1e6, dim=1)
            return dist_ap, dist_an, p_inds, n_inds
        else:
            return dist_ap, dist_an, None, None

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            
        # Compute L2 distance between features and find hard positive and negative samples
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, _, _ = self.hard_example_mining(dist_mat, labels, return_inds=False)
        
        # Compute ranking hinge loss
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
            
        return loss, dist_ap, dist_an