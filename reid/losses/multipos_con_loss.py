import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    Simplified version for single GPU usage
    """

    def __init__(self, temperature=0.1, device='cuda'):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_batch_size = None
        self.device = device

    def set_temperature(self, temp=0.1):
        self.temperature = temp
    
    # feats shape: [B, D]
    # labels shape: [B]
    def forward(self, feats, labels):

        feats = F.normalize(feats, dim=-1, p=2)
        batch_size = feats.size(0)

        # compute the mask based on labels
        if batch_size != self.last_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                          labels.contiguous().view(1, -1)).float().to(self.device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
                0
            )

            self.last_batch_size = batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # Stabilize the logits by subtracting the maximum value
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        
        # compute cross-entropy loss
        logits = F.log_softmax(logits, dim=-1)
        loss = torch.sum(p * logits, dim=-1)
        loss = -loss.mean()

        return loss, None, None
