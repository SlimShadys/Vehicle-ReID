import os
import random
from typing import Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.utils import (euclidean_dist, eval_func, load_model, re_ranking,
                        read_image, save_model, search, strint,
                        visualize_ranked_results)
from reid.datasets.transforms import Transformations
from reid.datasets.vehicle_id import VehicleID
from reid.datasets.veri_776 import Veri776
from reid.datasets.veri_wild import VeriWild
from reid.datasets.vru import VRU
from reid.loss import LossBuilder
from reid.models.color_model import SVM, EfficientNet
from reid.training.scheduler import WarmupDecayLR

class Trainer:
    def __init__(
        self,
        model: Union[torch.nn.Module, Union[EfficientNet, SVM]],
        dataloaders: Dict[str, Union[
                                        Optional[DataLoader],
                                        Dict[str, Tuple[DataLoader, int]], 
                                        Type[Union[VehicleID, Veri776, VeriWild, VRU]], 
                                        Transformations
                                    ]
                        ],
        #dataloaders,
        val_interval: int,
        loss_fn: LossBuilder,
        configs,
        device: str = 'cuda'
    ):
        self.model, self.car_classifier = model
        self.device = device
        self.train_loader = dataloaders['train']
        self.val_loader, self.num_query = dataloaders['val']
        self.dataset = dataloaders['dataset']
        self.transforms = dataloaders['transform']
        self.val_interval = val_interval
        self.loss_fn = loss_fn
        self.misc_configs, self.augmentation_configs, self.loss_configs, self.train_configs, self.val_configs, self.test_configs = configs

        # Test configurations (Whether we are testing or not)
        self.testing = self.test_configs.TESTING

        # Misc configurations
        self.output_dir = self.misc_configs.OUTPUT_DIR
        self.use_amp = self.misc_configs.USE_AMP

        if (self.testing == False):
            # Training configurations
            self.epochs = self.train_configs.EPOCHS
            self.learning_rate = self.train_configs.LEARNING_RATE
            self.log_interval = self.train_configs.LOG_INTERVAL
            self.load_checkpoint = self.train_configs.LOAD_CHECKPOINT
            self.use_warmup = self.train_configs.USE_WARMUP
            self.batch_size = self.train_configs.BATCH_SIZE

            # Loss configs
            self.use_rptm, self.rptm_type = self.loss_configs.USE_RPTM
            if self.use_rptm:
                self.max_negative_labels = self.dataset.max_negative_labels

            # Optimizer configurations
            self.weight_decay = self.train_configs.WEIGHT_DECAY
            self.weight_decay_bias = self.train_configs.WEIGHT_DECAY_BIAS
            self.bias_lr_factor = self.train_configs.BIAS_LR_FACTOR
            self.optimizer_name = self.train_configs.OPTIMIZER

            # Augmentation configurations
            self.img_height = self.augmentation_configs.HEIGHT + \
                self.augmentation_configs.PADDING * 2
            self.img_width = self.augmentation_configs.WIDTH + \
                self.augmentation_configs.PADDING * 2

            self.start_epoch = 0
            self.running_loss = []
            self.running_id_loss = []
            self.running_metric_loss = []
            self.acc_list = []

        # Validation configurations
        self.re_ranking = self.val_configs.RE_RANKING
        self.visualize_ranks = self.val_configs.VISUALIZE_RANKS

        if (self.testing == False):
            # Update the weights decay and learning rate for the model
            model_params = []
            for key, value in self.model.named_parameters():
                if not value.requires_grad:
                    continue
                else:
                    lr = self.learning_rate
                    weight_decay = self.weight_decay
                    if "bias" in key:
                        lr = self.learning_rate * self.bias_lr_factor
                        weight_decay = self.weight_decay_bias
                    model_params += [{"params": [value],
                                      "lr": lr,
                                      "weight_decay": weight_decay}]

            # Create the optimizer
            if (self.optimizer_name == 'adam'):
                self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)
            elif (self.optimizer_name == 'sgd'):
                self.optimizer = torch.optim.SGD(model_params, lr=self.learning_rate, momentum=0.9, dampening=0.0, nesterov=True)
            else:
                raise ValueError(
                    "Invalid optimizer name in config file. Choose between 'adam' or 'sgd'")

            if (self.use_warmup):
                # For Learning Rate Warm-up and Decay
                # We first warm-up the LR up for a few epochs, so that it reaches the desired LR
                # Then, we decay the LR using the milestones or a smooth decay
                self.scheduler = WarmupDecayLR(
                    optimizer=self.optimizer,
                    milestones=self.train_configs.STEPS,
                    warmup_epochs=self.train_configs.WARMUP_EPOCHS,
                    gamma=self.train_configs.GAMMA,
                    cosine_power=self.train_configs.COSINE_POWER,
                    decay_method=self.train_configs.DECAY_METHOD,
                    min_lr=self.train_configs.MIN_LR
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.train_configs.STEPS, gamma=self.train_configs.GAMMA)

            # Check if load checkpoint is not False. If it is not False, check if the string is not empty, then load the model
            if (self.load_checkpoint != False and self.load_checkpoint != ''):
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
        if (self.use_rptm):
            self.gms = self.dataset.gms
            self.pidx = self.dataset.pidx
            folders = []
            self.split_dir = os.path.join(self.dataset.data_path, 'splits')
            for fld in os.listdir(self.split_dir):
                folders.append(fld)
            data_index = search(self.split_dir)

        # Warning: use `torch.amp.GradScaler('cuda', args...)` instead.
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        for epoch in range(self.start_epoch, self.epochs):
            self.empty_lists()  # Empty the lists for each epoch

            # Train the model
            if self.use_rptm:
                self.train_rptm(epoch, scaler, data_index)  # Training with RPTM
            else:
                self.train(epoch, scaler)                   # Standard training

            # Update learning rate
            self.scheduler.step()

            # Log the current learning rate
            opt_lr = self.scheduler.optimizer.param_groups[0]['lr']

            if (epoch > 0 and epoch % self.val_interval == 0):
                self.validate(epoch, save_results=True, metrics=['reid'])

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
                    neg_label = random.choice(range(1, self.max_negative_labels))
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
            trainX = trainX.to(self.device)
            trainY = trainY.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                embeddings, pred_ids = self.model(trainX, training=True)
                ID_loss, metric_loss = self.loss_fn.forward(embeddings, pred_ids, trainY, normalize=False, use_rptm=True)

            loss = ID_loss + metric_loss

            # Accuracy
            pred_ids = pred_ids.detach().cpu()

            if (self.use_rptm):
                car_id = car_id[0:self.batch_size]
                pred_ids = pred_ids[0:self.batch_size]

            accuracy = (torch.max(pred_ids, 1)[1] == car_id).float().mean()

            # Using AMP for mixed precision training
            if(self.use_amp):
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

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
                ID_loss, metric_loss = self.loss_fn.forward(embeddings, pred_ids, car_id, normalize=False, use_rptm=False)

            loss = ID_loss + metric_loss

            # Accuracy
            accuracy = (torch.max(pred_ids, 1)[1] == car_id).float().mean()

            # Using AMP for mixed precision training
            if(self.use_amp):
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

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

    def validate_color(self, img: torch.Tensor, color_id: torch.Tensor):

        # Initialize the color predictions tensor
        color_predictions_tensor = torch.zeros((img.shape[0], 1), dtype=torch.float32, device=self.device)
        predicted_color_indices = []

        # EfficientNet
        if isinstance(self.car_classifier, EfficientNet):
            prediction = self.car_classifier.predict(img)
            color_predictions = [entry['color'] for entry in prediction]
        # ResNet + SVM
        elif isinstance(self.car_classifier, SVM):
            svm_embedding = self.model(img).detach().cpu().numpy()
            color_predictions = self.car_classifier.predict(svm_embedding)
        else:
            raise ValueError("Invalid color model. Please use either EfficientNet or SVM.")

        # We need to decode the color predictions, which are in a format:
        # [{'color': 'black', 'prob': '0.9999'}, ...] for the Spectrico model
        # or a tensor of shape (batch_size, num_classes) for the EfficientNet model
        # In both cases, we need to convert this to a tensor of indices of the colors
        for i, color in enumerate(color_predictions):
            # Get the index of the color from the dataset and append it to the list
            predicted_index = self.dataset.get_color_index(color)
            predicted_color_indices.append(predicted_index)
            color_predictions_tensor[i][0] = predicted_index
        
        predicted_color_indices = np.array(predicted_color_indices)
        actual_color_indices = color_id.detach().cpu().numpy() # Assuming color_id is a tensor
        
        # We have to remove the items which are -1 in the actual_color_indices
        # as well as the corresponding indices in the predicted_color_indices and color_predictions_tensor
        # Otherwise, we will get an error when calculating the accuracy since the indices will not match
        mask = actual_color_indices != -1
        actual_color_indices = actual_color_indices[mask]
        predicted_color_indices = predicted_color_indices[mask]
        color_predictions_tensor = color_predictions_tensor[mask]
        color_id = color_id[mask]
        
        # Calculate the accuracy between the tensor of predictions and the color_id
        return (torch.max(color_predictions_tensor, 1)[0] == color_id).float().mean(), predicted_color_indices, actual_color_indices

    def validate(self, epoch: Optional[int] = 0, save_results=False, metrics=None): 
        num_query = self.num_query
        features, car_ids, cam_ids, paths = [], [], [], []
        color_accuracies, predictions_colors, ground_truth_colors = [], [], []

        with torch.no_grad():
            for i, (img_path, img, folders, indices, car_id, cam_id, model_id, color_id, type_id, timestamp) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Running Validation"):
                img, car_id, cam_id, color_id = img.to(self.device), car_id.to(self.device), cam_id.to(self.device), color_id.to(self.device)

                if 'reid' in metrics:
                    self.model.eval()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                        feat = self.model(img, training=False).detach().cpu()
                        features.append(feat)

                if 'color' in metrics:
                    # Run the color model
                    color_accuracy, pred_colors, gt_colors = self.validate_color(img, color_id)
                    
                    # Append the color accuracy, predictions, and ground truth colors
                    # Skip this step if the color_accuracy is NaN
                    if (torch.isnan(color_accuracy).item() == False):
                        color_accuracies.append(color_accuracy)
                        predictions_colors.append(pred_colors)
                        ground_truth_colors.append(gt_colors)
                
                # Append the car_id, cam_id, and paths generally without any condition
                car_ids.append(car_id)
                cam_ids.append(cam_id)
                paths.extend(img_path)

        if 'reid' in metrics:
            features = torch.cat(features, dim=0)
            car_ids = torch.cat(car_ids, dim=0)
            cam_ids = torch.cat(cam_ids, dim=0)

            query_feat = features[:num_query]
            query_pid = car_ids[:num_query]
            query_camid = cam_ids[:num_query]
            # query_path = np.array(paths[:num_query])

            gallery_feat = features[num_query:]
            gallery_pid = car_ids[num_query:]
            gallery_camid = cam_ids[num_query:]
            # gallery_path = np.array(paths[num_query:])

        if 'color' in metrics:
            print("Total cars in the validation set: ", len(self.val_loader) * self.val_configs.BATCH_SIZE)
            print("Total cars with color information: ", len(color_accuracies) * self.val_configs.BATCH_SIZE)
            
            color_accuracies = torch.tensor(color_accuracies)
            print(f"Color Accuracy: {torch.mean(color_accuracies).item():.4f}%")

            if self.test_configs.PRINT_CONFUSION_MATRIX:
                # Before returning the accuracy, we would like to have some sort of confusion matrix
                # where we can see the predictions and the actual color_id
                # In this way, we can see if the model is confusing the colors with very slightly different colors
                predictions_colors = np.concatenate([arr for arr in predictions_colors]).tolist()
                ground_truth_colors = np.concatenate([arr for arr in ground_truth_colors]).tolist()
                conf_matrix = confusion_matrix(ground_truth_colors, predictions_colors, labels=list(self.dataset.color_dict.keys()), normalize='true')

                # Plot confusion matrix. Args are given based on the dataset
                if (self.dataset.dataset_name == 'veri-wild'):
                    fontsize = 12
                    figsize = (15, 15)
                    annot_kws_size = 14
                else: # For other datasets (VeRi-776, VRU, VehicleID)
                    fontsize = 9
                    figsize = (10, 12)
                    annot_kws_size = 9

                plt.figure(figsize=figsize)
                sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues",
                            xticklabels=list(self.dataset.color_dict.values()),
                            yticklabels=list(self.dataset.color_dict.values()),
                            cbar_kws={'label': 'Color Scale'},
                            annot_kws={"size": annot_kws_size}) # Increase font size of annotations
                plt.xticks(rotation=45, fontsize=fontsize)  # Rotate x labels
                plt.yticks(rotation=0, fontsize=fontsize)  # Adjust y-axis label rotation and font size
                plt.xlabel('Predicted Color')
                plt.ylabel('True Color')
                plt.title('Confusion Matrix of Color Predictions')
                plt.tight_layout()  # Adjust layout to fit labels better
                plt.show()

        if 'reid' in metrics:
            distmat = euclidean_dist(query_feat, gallery_feat, self.use_amp, train=False)

            CMC_RANKS = [1, 3, 5, 10]
            cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(),
                query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(),
                max_rank=max(CMC_RANKS), dataset_name=self.dataset.dataset_name, re_rank=False)
                        
            if (self.re_ranking):
                distmat_re = re_ranking(query_feat, gallery_feat, k1=80, k2=15, lambda_value=0.2)

                cmc_re, mAP_re, _ = eval_func(distmat_re, query_pid.detach().cpu().numpy(), gallery_pid.detach().cpu().numpy(),
                                                query_camid.detach().cpu().numpy(), gallery_camid.detach().cpu().numpy(),
                                                max_rank=max(CMC_RANKS), dataset_name=self.dataset.dataset_name, re_rank=True)

            # Visualize the ranked results if the flag is set to True
            if (self.visualize_ranks):
                if (self.re_ranking):
                    visualize_ranked_results(distmat=distmat_re, dataset=(self.dataset.query, self.dataset.gallery), topk=max(CMC_RANKS))
                else:
                    visualize_ranked_results(distmat=distmat, dataset=(self.dataset.query, self.dataset.gallery), topk=max(CMC_RANKS))

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
                    if (self.re_ranking):
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