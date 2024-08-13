import os
import random
from typing import Optional

import numpy as np
import numpy.ma as ma
import torch
from misc.utils import (euclidean_dist, eval_func, load_model, re_ranking,
                        read_image, save_model, search, strint)
from tqdm import tqdm
from training.scheduler import WarmupDecayLR

class Trainer:
    def __init__(self, model: torch.nn.Module, dataloaders, val_interval, loss_fn: torch.Tensor, configs, device='cuda'):
        self.model = model
        self.device = device
        self.train_loader = dataloaders['train']
        self.val_loader, self.num_query = dataloaders['val']
        self.dataset = dataloaders['dataset']
        self.transforms = dataloaders['transform']
        self.val_interval = val_interval
        self.loss_fn = loss_fn
        self.train_configs, self.val_configs, self.loss_configs, self.augmentation_configs = configs
        
        # Training configurations
        self.epochs = self.train_configs['epochs']
        self.learning_rate = self.train_configs['learning_rate']
        self.log_interval = self.train_configs['log_interval']
        self.output_dir = self.train_configs['output_dir']
        self.load_checkpoint = self.train_configs['load_checkpoint']
        self.use_warmup = self.train_configs['use_warmup']
        self.batch_size = self.train_configs['batch_size']
        self.use_amp = self.train_configs['use_amp']
        
        # Loss configs
        self.use_rptm, self.rptm_type = self.loss_configs['use_rptm']

        # Optimizer configurations
        self.weight_decay = self.train_configs['weight_decay']
        self.weight_decay_bias = self.train_configs['weight_decay_bias']
        self.bias_lr_factor = self.train_configs['bias_lr_factor']
        self.optimizer_name = self.train_configs['optimizer']

        # Validation configurations
        self.re_ranking = self.val_configs['re_ranking']
        
        # Augmentation configurations
        self.img_height = self.augmentation_configs['height'] + self.augmentation_configs['padding'] * 2
        self.img_width = self.augmentation_configs['width'] + self.augmentation_configs['padding'] * 2
        
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
                weight_decay = self.weight_decay
                if "bias" in key:
                    lr = self.learning_rate * self.bias_lr_factor
                    weight_decay = self.weight_decay_bias
                model_params += [{"params": [value],
                                  "lr": lr, "weight_decay": weight_decay}]

            # Create the optimizer
            if(self.optimizer_name == 'adam'):
                self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)
            elif(self.optimizer_name == 'sgd'):
                self.optimizer = torch.optim.SGD(model_params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay, dampening=0.0, nesterov=True)
            else:
                raise ValueError("Invalid optimizer. Choose between 'adam' or 'sgd'")

            if (self.use_warmup):
                # For Learning Rate Warm-up and Decay
                # We first warm-up the LR up for a few epochs, so that it reaches the desired LR
                # Then, we decay the LR using the milestones or a smooth decay
                self.scheduler = WarmupDecayLR(
                    optimizer=self.optimizer,
                    milestones=self.train_configs['steps'],
                    warmup_epochs=self.train_configs['warmup_epochs'],
                    warmup_gamma=self.train_configs['warmup_gamma'],
                    decay_method=self.train_configs['decay_method']
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.train_configs['steps'], gamma=self.train_configs['warmup_gamma'])
            
            # Check if load checkpoint is not False. If it is not False, check if the string is not empty, then load the model
            if(self.load_checkpoint != False and self.load_checkpoint != ''):
                self.model, self.optimizer, self.scheduler, self.start_epoch = load_model(
                    self.load_checkpoint, self.model, self.optimizer, self.scheduler, self.device)
                
    def empty_lists(self):
        self.running_loss = []
        self.running_id_loss = []
        self.running_metric_loss = []
        self.acc_list = []

    def retrieve_image(self, dic):
        img_path = dic[0]
        img = read_image(img_path)
        transforms = self.transforms.get_train_transform()
        img = transforms(img).to(self.device)
        return img

    def run(self):
        if(self.use_rptm):
            self.gms = self.dataset.gms
            self.pidx = self.dataset.pidx
            folders = []
            self.split_dir = os.path.join(self.dataset.data_path, 'splits')
            for fld in os.listdir(self.split_dir):
                folders.append(fld)
            data_index = search(self.split_dir)
        
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                
        for epoch in range(self.start_epoch, self.epochs):
            self.empty_lists()  # Empty the lists for each epoch

            # Train the model
            if self.use_rptm:
                self.train_rptm(epoch, scaler, data_index) # Training with RPTM
            else:
                self.train(epoch, scaler) # Standard training 

            # Update learning rate
            self.scheduler.step()

            # Log the current learning rate
            opt_lr = self.scheduler.optimizer.param_groups[0]['lr']

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
                'scheduler': self.scheduler.state_dict(),
                'loss': np.mean(self.running_loss),
            }, self.output_dir, model_path=f'model_ep-{epoch}_loss-{np.mean(self.running_loss):.4f}.pth')

    def train_rptm(self, epoch, scaler: torch.cuda.amp.GradScaler, data_index):
        self.model.train()

        for batch_idx, (img_path, img, folders, indices, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}/{self.epochs}"):
            trainX = torch.zeros((self.batch_size * 3, 3, self.img_height, self.img_height), dtype=torch.float32)
            trainY = torch.zeros((self.batch_size * 3), dtype=torch.int64)

            for j in range(self.batch_size):
                labelx = str(folders[j])
                indexx = int(indices[j])
                cidx = int(car_id[j])
                if indexx > len(self.gms[labelx]) - 1:
                    indexx = len(self.gms[labelx]) - 1
                a = self.gms[labelx][indexx]
                
                if self.rptm_type == 'min':
                    threshold = np.arange(10)
                elif self.rptm_type == 'mean':
                    threshold = np.arange(np.amax(self.gms[labelx][indexx]) // 2)
                elif self.rptm_type == 'max':
                    threshold = np.arange(np.amax(self.gms[labelx][indexx]))

                minpos = np.argmin(ma.masked_where(a == threshold, a))
                while True:
                    neg_label = random.choice(range(1, 770))
                    if (neg_label is not int(labelx)) and (os.path.isdir(os.path.join(self.split_dir, strint(neg_label, self.dataset.dataset_name))) is True):
                        break
                negative_label = strint(neg_label, self.dataset.dataset_name)
                neg_cid = self.pidx[negative_label]
                neg_index = random.choice(range(0, len(self.gms[negative_label])))

                pos_dic = self.dataset.train[data_index[cidx][1] + minpos]
                neg_dic = self.dataset.train[data_index[neg_cid][1] + neg_index]
                               
                trainX[j] = img[j]
                trainX[j + self.batch_size] = self.retrieve_image(pos_dic)
                trainX[j + (self.batch_size * 2)] = self.retrieve_image(neg_dic)
                trainY[j] = cidx
                trainY[j + self.batch_size] = pos_dic[2]
                trainY[j + (self.batch_size * 2)] = neg_dic[2]
            
            self.optimizer.zero_grad()
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                embeddings, pred_ids = self.model(trainX, training=True)
                ID_loss, metric_loss = self.loss_fn(embeddings, pred_ids, trainY, normalize=False, use_rptm=True)
            
            loss = ID_loss + metric_loss

            # Accuracy
            pred_ids = pred_ids.detach().cpu()
            
            if(self.use_rptm):
                car_id = car_id[0:self.batch_size]
                pred_ids = pred_ids[0:self.batch_size]
                
            accuracy = (torch.max(pred_ids, 1)[1] == car_id).float().mean()

            # Using AMP for mixed precision training
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # loss.backward()
            # self.optimizer.step()

            self.running_loss.append(loss.detach().cpu().item())
            self.running_id_loss.append(ID_loss.detach().cpu().item())
            self.running_metric_loss.append(metric_loss.detach().cpu().item())
            self.acc_list.append(accuracy.detach().cpu().item())
            
            # Get scheduler class, in such a way that if we are dealing with WarmupDecayLR, we can get the learning rate directly
            # otherwise, we can call the get_lr() method (in this case we are using MultiStepLR)
            # This is needed to silence the warning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
            if self.scheduler.__class__ == torch.optim.lr_scheduler.MultiStepLR:
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.scheduler.get_lr()[0]

            if (batch_idx % self.log_interval == 0):
                print(f"\tIteration[{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {np.mean(self.running_loss):.4f} | "
                      f"ID Loss: {np.mean(self.running_id_loss):.4f} | "
                      f"Metric Loss: {np.mean(self.running_metric_loss):.4f} | "
                      f"Accuracy: {np.mean(self.acc_list):.4f} | "
                      f"LR: {lr:.2e}")

    def train(self, epoch, scaler: torch.cuda.amp.GradScaler):
        self.model.train()

        for batch_idx, (img_path, img, folders, indices, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}/{self.epochs}"):
            img, car_id = img.to(self.device), car_id.to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                # Forward pass on the Network
                # Up to the last Linear layer for the ID prediction (ID loss)
                # Up to the last Conv layer for the Embeddings (Metric loss)
                embeddings, pred_ids = self.model(img, training=True)

                # Loss
                ID_loss, metric_loss = self.loss_fn(embeddings, pred_ids, car_id, normalize=False, use_rptm=True)
                
            loss = ID_loss + metric_loss

            # Accuracy
            accuracy = (torch.max(pred_ids, 1)[1] == car_id).float().mean()

            # Using AMP for mixed precision training
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # loss.backward()
            # self.optimizer.step()

            self.running_loss.append(loss.detach().cpu().item())
            self.running_id_loss.append(ID_loss.detach().cpu().item())
            self.running_metric_loss.append(metric_loss.detach().cpu().item())
            self.acc_list.append(accuracy.detach().cpu().item())

            if (batch_idx % self.log_interval == 0):
                print(f"\tIteration[{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {np.mean(self.running_loss):.4f} | "
                      f"ID Loss: {np.mean(self.running_id_loss):.4f} | "
                      f"Metric Loss: {np.mean(self.running_metric_loss):.4f} | "
                      f"Accuracy: {np.mean(self.acc_list):.4f} | "
                      f"LR: {self.scheduler.get_lr()[0]:.2e}")
            break
        
    def validate(self, epoch: Optional[int] = 0, save_results=False):
        self.model.eval()
        num_query = self.num_query
        features, car_ids, cam_ids, paths = [], [], [], []

        with torch.no_grad():
            for i, (img_path, img, folders, indices, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Running Validation"):
                img, car_id, cam_id = img.to(self.device), car_id.to(self.device), cam_id.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
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

        distmat = euclidean_dist(query_feat, gallery_feat, self.use_amp, train=False)

        CMC_RANKS = [1, 3, 5, 10]
        cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(),
                                query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(), max_rank=max(CMC_RANKS), re_rank=False)

        if(self.re_ranking):
            distmat_re = re_ranking(query_feat, gallery_feat, k1=80, k2=15, lambda_value=0.2)
            
            # We have to use the evaluate_vid function only for Vehicle ID dataset
            if self.dataset.dataset_name == 'vehicle_id':
                raise NotImplementedError("The evaluate_vid function is not implemented yet for the Vehicle ID dataset")
                #cmc_re, mAP_re = evaluate_vid(distmat_re, q_pids, g_pids, q_camids, g_camids, 50)
            else:
                cmc_re, mAP_re, _ = eval_func(distmat_re, query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(),
                                        query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(), max_rank=max(CMC_RANKS), re_rank=True)
                
        # Output the results to the console, depending on the re-ranking
        print(f"Epoch {epoch}\n")
        if self.re_ranking:
            for r in CMC_RANKS:
                print('\t - CMC Rank-{}: {:.2%} ({:.2%})'.format(r, cmc[r-1], cmc_re[r-1]))
            print('\t - mAP: {:.2%} ({:.2%})'.format(mAP, mAP_re))
        else:
            for r in CMC_RANKS:
                print('\t - CMC Rank-{}: {:.2%}'.format(r, cmc[r-1]))
            print('\t - mAP: {:.2%}'.format(mAP))
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