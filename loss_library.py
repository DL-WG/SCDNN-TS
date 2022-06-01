import torch
import torch.nn as nn
import torch.nn.functional as F

class DSCLoss(torch.nn.Module):

    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs)       
        batch_size = inputs.shape[0]

        #flatten label and prediction tensors
        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        # log_prob = F.log_softmax(input_tensor, dim=-1)

        # prob = torch.exp(input_tensor)
        # return F.cross_entropy(
        #     ((1 - prob) ** self.gamma) * input_tensor, 
        #     target_tensor, 
        #     weight=self.weight,
        #     reduction = self.reduction
        # )
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )