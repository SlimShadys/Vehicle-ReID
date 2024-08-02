import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import euclidean_dist, normalize

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

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, _, _ = self.hard_example_mining(dist_mat, labels, return_inds=False)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`(?)
        Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
        """

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_inds:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels)
                .copy_(torch.arange(0, N).long())
                .unsqueeze(0).expand(N, N))
            # shape [N, 1]
            p_inds = torch.gather(
                ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
            n_inds = torch.gather(
                ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
        else:
            p_inds = None
            n_inds = None
        return dist_ap, dist_an, p_inds, n_inds

# class TripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#     def __init__(self, margin=None, mining_method='batch_hard'):
#         self.margin = margin
#         self.mining_method = mining_method
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()

#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
            
#         # Compute L2 distance between features and find hard positive and negative samples
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dist_ap, dist_an, _, _ = self.hard_example_mining(dist_mat, labels, self.mining_method, return_inds=False)
        
#         # Compute ranking hinge loss
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
            
#         return loss, dist_ap, dist_an
    
#     def hard_example_mining(self, dist_mat: torch.Tensor, labels: torch.Tensor, mining_method='batch_hard', return_inds=False):
#         """For each anchor, find the hardest positive and negative sample.
#         Args:
#         dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#         labels: pytorch LongTensor, with shape [N]
#         mining_method: str, mining method for hard negative examples; choices: 'batch_hard', 'batch_sample', 'batch_soft'
#         return_inds: whether to return the indices. Save time if `False`(?)
#         Returns:
#         dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#         dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#         p_inds: pytorch LongTensor, with shape [N];
#             indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#         n_inds: pytorch LongTensor, with shape [N];
#             indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#         NOTE: Only consider the case in which all labels have same num of samples,
#         thus we can cope with all anchors in parallel.
#         """
#         assert len(dist_mat.size()) == 2
#         assert dist_mat.size(0) == dist_mat.size(1)

#         N = dist_mat.size(0)
#         device = dist_mat.device
#         dist_ap, dist_an = [], []
#         relative_p_inds, relative_n_inds = [], []

#         # Shape [N, N]
#         is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#         is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
#         # For each anchor, find the hardest positive and negative sample based on the mining method
#         # 'dist_ap' means distance(anchor, positive)
#         # 'dist_an' means distance(anchor, negative)
#         # Distances will have shape [N, 1]
#         for i in range(N):
#             positive_mask = dist_mat[i][is_pos[i]]
#             negative_mask = dist_mat[i][is_neg[i]]

#             # Batch hard mining | Select the hardest positive and negative samples
#             if mining_method == 'batch_hard':
#                 if len(positive_mask) == 0:
#                     dist_ap.append(torch.tensor([0.0], device=device))
#                     relative_p_inds.append(torch.tensor([-1], device=device))
#                     print('WARNING: No positive pair found for anchor point')
#                 else:
#                     max_pos_pair = positive_mask.max()
#                     dist_ap.append(max_pos_pair.unsqueeze(0))
#                     relative_p_inds.append(torch.where(dist_mat[i] == max_pos_pair)[0][0].unsqueeze(0))

#                 if len(negative_mask) == 0:
#                     dist_an.append(torch.tensor([0.0], device=device))
#                     relative_n_inds.append(torch.tensor([-1], device=device))
#                     print('WARNING: No negative pair found for anchor point')
#                 else:
#                     min_neg_pair = negative_mask.min()
#                     dist_an.append(min_neg_pair.unsqueeze(0))
#                     relative_n_inds.append(torch.where(dist_mat[i] == min_neg_pair)[0][0].unsqueeze(0))

#             # Batch sample mining | Randomly sample positive and negative samples
#             elif mining_method == 'batch_sample':
#                 if len(positive_mask) == 0:
#                     dist_ap.append(torch.tensor([0.0], device=device))
#                     relative_p_inds.append(torch.tensor([-1], device=device))
#                     print('WARNING: No positive pair found for anchor point')
#                 else:
#                     sampled_pos_idx = torch.multinomial(F.softmax(positive_mask, dim=0), num_samples=1)
#                     dist_ap.append(positive_mask[sampled_pos_idx])
#                     relative_p_inds.append(torch.where(is_pos[i])[0][sampled_pos_idx])

#                 if len(negative_mask) == 0:
#                     dist_an.append(torch.tensor([0.0], device=device))
#                     relative_n_inds.append(torch.tensor([-1], device=device))
#                     print('WARNING: No negative pair found for anchor point')
#                 else:
#                     sampled_neg_idx = torch.multinomial(F.softmin(negative_mask, dim=0), num_samples=1)
#                     dist_an.append(negative_mask[sampled_neg_idx])
#                     relative_n_inds.append(torch.where(is_neg[i])[0][sampled_neg_idx])

#             # Batch soft mining | Softly select positive and negative samples
#             elif mining_method == 'batch_soft':
#                 if len(positive_mask) == 0:
#                     dist_ap.append(torch.tensor([0.0], device=device))
#                     relative_p_inds.append(torch.tensor([-1], device=device))
#                     print('WARNING: No positive pair found for anchor point')
#                 else:
#                     weight_ap = torch.exp(positive_mask) / torch.exp(positive_mask).sum()
#                     dist_ap.append((weight_ap * positive_mask).sum().unsqueeze(0))
#                     relative_p_inds.append(torch.argmax(weight_ap).unsqueeze(0))

#                 if len(negative_mask) == 0:
#                     dist_an.append(torch.tensor([0.0], device=device))
#                     relative_n_inds.append(torch.tensor([-1], device=device))
#                     print('WARNING: No negative pair found for anchor point')
#                 else:
#                     weight_an = torch.exp(-negative_mask) / torch.exp(-negative_mask).sum()
#                     dist_an.append((weight_an * negative_mask).sum().unsqueeze(0))
#                     relative_n_inds.append(torch.argmax(weight_an).unsqueeze(0))

#         # Stack the distances and indices
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
#         relative_p_inds = torch.cat(relative_p_inds)
#         relative_n_inds = torch.cat(relative_n_inds)

#         if return_inds:
#             return dist_ap, dist_an, relative_p_inds, relative_n_inds
#         else:
#             return dist_ap, dist_an, None, None