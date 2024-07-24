import torch
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, dataloaders, loss_fn, epochs, batch_size, learning_rate, device='cuda'):
        self.model = model
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['val']
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = []
            running_id_loss = []
            running_metric_loss = []
            
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

                running_loss.append(loss.item())
                running_id_loss.append(ID_loss.item())
                running_metric_loss.append(metric_loss.item())

                if(i % 5 == 0):
                    print(f"Running ID Loss: {np.array(running_id_loss).mean():.4f}, \
                            Running Metric Loss: {np.array(running_metric_loss).mean():.4f}, \
                            Running Loss: {np.array(running_loss).mean():.4f}")

            print(f"Epoch {epoch+1}/{self.epochs}\n \
                  \t - ID Loss: {np.array(running_id_loss):.4f}\n \
                  \t - Metric Loss: {np.array(running_metric_loss):.4f}\n \
                  \t - Loss (ID + Metric): {np.array(running_loss):.4f}")
            
            # Save the model
            torch.save(self.model.state_dict(), 'model_ep-{}_loss-{}.pth'.format(epoch+1, np.array(running_loss)))

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")