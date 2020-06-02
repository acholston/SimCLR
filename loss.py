import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ContLoss(nn.Module):
    def __init__(self, batch_size, temperature, metric):
        super(ContLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.create_mask()

        self.criterion = nn.CrossEntropyLoss()
        self.metric = metric

    def create_mask(self):
        mask = torch.ones((self.batch_size*2, self.batch_size*2)) - torch.eye(self.batch_size*2, self.batch_size*2)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask.long().to(device)

    def forward(self, z):
        if self.metric == 'cosine':
            similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        else:
            similarity = torch.dot(z, z.T) / self.temperature

        #Isolate same instance - Don't want match
        similarity_i = torch.diag(similarity, self.batch_size)
        similarity_j = torch.diag(similarity, -self.batch_size)

        #find matches and negatives
        pos = torch.cat((similarity_i, similarity_j), dim=0).view(self.batch_size*2, -1)
        neg = similarity[self.mask].view(self.batch_size*2, -1)

        #drive all of these to zero
        labels = torch.zeros(self.batch_size*2).long().to(device)
        logits = torch.cat([pos, neg], dim=1)
        loss = self.criterion(logits, labels)
        return loss
