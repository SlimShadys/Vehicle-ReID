import torch
from tqdm import tqdm
import numpy as np
from evaluate import eval_func
from utils import euclidean_dist

class Trainer:
    def __init__(self, model, dataloaders, val_interval, loss_fn, epochs, batch_size, learning_rate, device='cuda'):
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader, self.num_query = dataloaders['val']
        self.val_interval = val_interval
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.running_loss = []
        self.running_id_loss = []
        self.running_metric_loss = []

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def empty_lists(self):
        self.running_loss = []
        self.running_id_loss = []
        self.running_metric_loss = []

    def run(self):
        for epoch in range(self.epochs):
            self.empty_lists() # Empty the lists for each epoch
                     
            self.train(epoch)
            
            if(epoch > 0 and epoch % self.val_interval == 0):
                self.validate()

            print(f"Epoch {epoch+1}/{self.epochs}\n"
                f"\t - ID Loss: {np.mean(self.running_id_loss):.4f}\n"
                f"\t - Metric Loss: {np.mean(self.running_metric_loss):.4f}\n"
                f"\t - Loss (ID + Metric): {np.mean(self.running_loss):.4f}")
      
            # Save the model
            torch.save(self.model.state_dict(), f'model_ep-{epoch+1}_loss-{np.mean(self.running_loss):.4f}.pth')
      
    def train(self, epoch):
        self.model.train()
           
        for i, (img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}"):
            img, car_id = img.to(self.device), car_id.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass on Resnet
            # Up to the last Linear layer for the ID prediction (ID loss)
            # Up to the last Conv layer for the Embeddings (Metric loss)
            embeddings, pred_ids = self.model(img, training=True)
            
            ID_loss, metric_loss = self.loss_fn(embeddings, pred_ids, car_id)
            loss = ID_loss + metric_loss
            loss.backward()
            self.optimizer.step()
            
            self.running_loss.append(loss.item())
            self.running_id_loss.append(ID_loss.item())
            self.running_metric_loss.append(metric_loss.item())
        return

    def validate(self):
        self.model.eval()
        num_query = self.num_query
        features, car_ids, cam_ids, paths = [], [], [], []
        
        with torch.no_grad():
            for i, (img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Running Validation"):
                img, car_id, cam_id = img.to(self.device), car_id.to(self.device), cam_id.to(self.device)
                
                feat = self.model(img, training=False).detach().cpu()
                features.append(feat)
                car_ids.append(car_id)
                cam_ids.append(cam_id)
                paths.extend(img_path)
                
        features = torch.cat(features, dim=0)
        car_ids = torch.cat(car_ids, dim=0)
        cam_ids = torch.cat(cam_ids, dim=0)

        query_feat = features[:num_query]
        query_pid = car_ids[:num_query]
        query_camid = cam_ids[:num_query]
        query_path = np.array(paths[:num_query])

        gallery_feat = features[num_query:]
        gallery_pid = car_ids[num_query:]
        gallery_camid = cam_ids[num_query:]
        gallery_path = np.array(paths[num_query:])
        
        distmat = euclidean_dist(query_feat, gallery_feat)

        cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(), 
                             query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(), max_rank=5)
        
        CMC_RANKS = [1, 3, 5]
        for r in CMC_RANKS:
            print('\t - CMC Rank-{}: {:.2%}'.format(r, cmc[r-1]))
        print('\t - mAP: {:.2%}'.format(mAP))
        print('\t ----------------------')