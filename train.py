import os
from typing import Optional

import numpy as np
import torch
from misc.utils import (euclidean_dist, eval_func, load_model, re_ranking,
                        save_model)
from tqdm import tqdm
from training.scheduler import WarmupDecayLR

class Trainer:
    def __init__(self, model: torch.nn.Module, dataloaders, val_interval, loss_fn: torch.Tensor, train_configs, val_configs, device='cuda'):
        self.model = model
        self.device = device
        self.train_loader = dataloaders['train']
        self.val_loader, self.num_query = dataloaders['val']
        self.val_interval = val_interval
        self.loss_fn = loss_fn
        
        # Training configurations
        self.epochs = train_configs['epochs']
        self.learning_rate = train_configs['learning_rate']
        self.log_interval = train_configs['log_interval']
        self.output_dir = train_configs['output_dir']
        self.load_checkpoint = train_configs['load_checkpoint']
        self.use_warmup = train_configs['use_warmup']
        
        # Validation configurations
        self.re_ranking = val_configs['re_ranking']
        
        self.start_epoch = 0
        self.running_loss = []
        self.running_id_loss = []
        self.running_metric_loss = []
        self.acc_list = []

        if (self.learning_rate is not None):
            # Update the weights decay and learning rate for the model
            model_params = []
            for key, value in self.model.named_parameters():
                if not value.requires_grad:
                    continue
                lr = self.learning_rate
                weight_decay = train_configs['weight_decay']
                if "bias" in key:
                    lr = self.learning_rate * train_configs['bias_lr_factor']
                    weight_decay = train_configs['weight_decay_bias']
                model_params += [{"params": [value],
                                  "lr": lr, "weight_decay": weight_decay}]

            # Create the optimizer
            self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)

            if (self.use_warmup):
                # For Learning Rate Warm-up and Decay
                # We first warm-up the LR up for a few epochs, so that it reaches the desired LR
                # Then, we decay the LR using the milestones or a smooth decay
                self.scheduler_warmup = WarmupDecayLR(
                    optimizer=self.optimizer,
                    milestones=train_configs['steps'],
                    warmup_epochs=train_configs['warmup_epochs'],
                    warmup_gamma=train_configs['warmup_gamma'],
                    decay_method=train_configs['decay_method']
                )
            else:
                self.scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=train_configs['steps'], gamma=train_configs['warmup_gamma'])
            
            # Check if load checkpoint is not False. If it is not False, check if the string is not empty, then load the model
            if(self.load_checkpoint != False and self.load_checkpoint != ''):
                self.model, self.optimizer, self.scheduler_warmup, self.start_epoch = load_model(
                    self.load_checkpoint, self.model, self.optimizer, self.scheduler_warmup, self.device)
                
    def empty_lists(self):
        self.running_loss = []
        self.running_id_loss = []
        self.running_metric_loss = []
        self.acc_list = []

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.empty_lists()  # Empty the lists for each epoch

            # Train the model
            self.train(epoch)

            # Update learning rate
            self.scheduler_warmup.step()

            # Log the current learning rate
            opt_lr = self.scheduler_warmup.optimizer.param_groups[0]['lr']

            if (epoch > 0 and epoch % self.val_interval == 0):
                self.validate(epoch, save_results=True)

            print(f"Epoch {epoch}/{self.epochs} | LR: {opt_lr:.2e}\n"
                  f"\t - ID Loss: {np.mean(self.running_id_loss):.4f}\n"
                  f"\t - Metric Loss: {np.mean(self.running_metric_loss):.4f}\n"
                  f"\t - Loss (ID + Metric): {np.mean(self.running_loss):.4f}\n"
                  f"\t - Accuracy: {np.mean(self.acc_list):.4f}\n")

            # Save the model every epoch
            save_model({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler_warmup.state_dict(),
                'loss': np.mean(self.running_loss),
            }, self.output_dir, model_path=f'model_ep-{epoch}_loss-{np.mean(self.running_loss):.4f}.pth')

    def train(self, epoch):
        self.model.train()

        for i, (img_path, img, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}/{self.epochs}"):
            img, car_id = img.to(self.device), car_id.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass on Resnet
            # Up to the last Linear layer for the ID prediction (ID loss)
            # Up to the last Conv layer for the Embeddings (Metric loss)
            embeddings, pred_ids = self.model(img, training=True)

            # Loss
            ID_loss, metric_loss = self.loss_fn(embeddings, pred_ids, car_id, normalize=False)
            loss = ID_loss + metric_loss

            # Accuracy
            accuracy = (torch.max(pred_ids, 1)[1] == car_id).float().mean()

            loss.backward()
            self.optimizer.step()

            self.running_loss.append(loss.detach().cpu().item())
            self.running_id_loss.append(ID_loss.detach().cpu().item())
            self.running_metric_loss.append(metric_loss.detach().cpu().item())
            self.acc_list.append(accuracy.detach().cpu().item())

            if (i % self.log_interval == 0):
                print(f"\tIteration[{i}/{len(self.train_loader)}] "
                      f"Loss: {np.mean(self.running_loss):.4f} | "
                      f"ID Loss: {np.mean(self.running_id_loss):.4f} | "
                      f"Metric Loss: {np.mean(self.running_metric_loss):.4f} | "
                      f"Accuracy: {np.mean(self.acc_list):.4f} | "
                      f"LR: {self.scheduler_warmup.get_lr()[0]:.2e}")

    def validate(self, epoch: Optional[int] = 0, save_results=False):
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

        CMC_RANKS = [1, 3, 5, 10]
        cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(),
                                query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(), max_rank=max(CMC_RANKS), re_rank=False)

        for r in CMC_RANKS:
            print('\t - CMC Rank-{}: {:.2%}'.format(r, cmc[r-1]))
        print('\t - mAP: {:.2%}'.format(mAP))
        print('\t ----------------------')

        if(self.re_ranking):
            distmat_re = re_ranking(query_feat, gallery_feat, k1=80, k2=15, lambda_value=0.2)
            
            # if dataset == 'vehicleid':
            #     cmc_re, mAP_re = evaluate_vid(distmat_re, q_pids, g_pids, q_camids, g_camids, 50)
            # else:
            cmc_re, mAP_re, _ = eval_func(distmat_re, query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(),
                                    query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(), max_rank=max(CMC_RANKS), re_rank=True)
            print('-- Re-Ranked Results --')
            for r in CMC_RANKS:
                print('\t - CMC Rank-{}: {:.2%}'.format(r, cmc_re[r-1]))
            print('\t - mAP: {:.2%}'.format(mAP_re))
            print('\t ----------------------')

        if (save_results):
            # Append the results to a file
            with open(os.path.join(self.output_dir, 'results-val.txt'), 'a') as f:
                if(self.re_ranking):
                    f.write(f"Epoch {epoch}\n"
                            f"\t - CMC Rank-1: {cmc[CMC_RANKS[0]-1]:.2%} ({cmc_re[CMC_RANKS[0]-1]:.2%})\n"
                            f"\t - CMC Rank-3: {cmc[CMC_RANKS[1]-1]:.2%} ({cmc_re[CMC_RANKS[1]-1]:.2%})\n"
                            f"\t - CMC Rank-5: {cmc[CMC_RANKS[2]-1]:.2%} ({cmc_re[CMC_RANKS[2]-1]:.2%})\n"
                            f"\t - CMC Rank-10: {cmc[CMC_RANKS[3]-1]:.2%} ({cmc_re[CMC_RANKS[3]-1]:.2%})\n"
                            f"\t - mAP: {mAP:.2%} ({mAP_re:.2%})\n"
                            f"\t ----------------------\n")
                else:
                    f.write(f"Epoch {epoch}\n"
                            f"\t - CMC Rank-1: {cmc[CMC_RANKS[0]-1]:.2%}\n"
                            f"\t - CMC Rank-3: {cmc[CMC_RANKS[1]-1]:.2%}\n"
                            f"\t - CMC Rank-5: {cmc[CMC_RANKS[2]-1]:.2%}\n"
                            f"\t - CMC Rank-10: {cmc[CMC_RANKS[3]-1]:.2%}\n"
                            f"\t - mAP: {mAP:.2%}\n"
                            f"\t ----------------------\n")