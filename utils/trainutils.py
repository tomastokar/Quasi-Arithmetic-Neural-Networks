import time
import torch
from torch.utils.data import DataLoader
from utils.datautils import set_collate_fn



class SetTrainer:
    def __init__(self, 
                 model, 
                 dataset, 
                 eval_dataset=None, 
                 batch_size=32, 
                 lr=1e-3, 
                 weight_decay=0.,                  
                 loss_fn=None,
                 perf_fn=None,
                 collate_fn=None,
                 device = 'cpu',
                 verbosity = 10):
        """
        Args:
            model (torch.nn.Module): PyTorch model to train.
            dataset (torch.utils.data.Dataset): Training dataset.
            eval_dataset (torch.utils.data.Dataset, optional): Evaluation dataset.
            batch_size (int): Number of samples per batch.
            lr (float): Learning rate for optimizer.
            weight_decay (float): Weight decay (L2 regularization).
            loss_fn (callable): Loss function.
        """
        self.device = device
        
        self.model = model
        self.model.to(self.device)
        
        self.train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn if collate_fn is not None else set_collate_fn
        )
        
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset, 
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn if collate_fn is not None else set_collate_fn
            )
        else:
            self.eval_loader = None
            
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.perf_fn = perf_fn
        
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.verbosity = verbosity

    def step(self, batch):
        x, m, y = batch
        
        # Set device
        x = x.to(self.device)
        m = m.to(self.device)
        y = y.to(self.device)
                
        y_pred = self.model(x, m)
        loss = self.loss_fn(y_pred, y)
        
        if self.perf_fn is not None:
            perf = self.perf_fn(y_pred, y)
        else:
            perf = loss.detach()
            
        return loss, perf

    def evaluate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            float: Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0.0
        total_perf = 0.0
        
        num_batches = len(self.eval_loader)

        with torch.no_grad():
            for batch in self.eval_loader:
                loss, perf = self.step(batch)
                total_loss += loss.item()                            
                total_perf += perf.item()

        average_loss = total_loss / num_batches                
        average_perf = total_perf / num_batches
            
        return average_loss, average_perf

    def train(self, num_epochs=10):
        """
        Trains the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of training epochs.
        """
        
        train_losses = []
        eval_losses = []
        
        start = time.time()
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            
            num_batches = len(self.train_loader) 
            total_loss = 0.0   
            total_perf = 0.0
            
            print('\n')         
            for batch_idx, batch in enumerate(self.train_loader):
                loss, perf = self.step(batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss = loss.item()
                perf = perf.item()
                total_loss += loss                        
                total_perf += perf
                
                if batch_idx % self.verbosity == 0:                    
                    print(f"Batch {batch_idx + 1}/{num_batches}; Loss: {loss:.4f}")

            train_loss = total_loss / num_batches
            train_perf = total_perf / num_batches
            train_losses.append(train_loss)      
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss = {train_loss:.4f}")
            print(f"Train Perf = {train_perf:.4f}")         
                        

            # Evaluate if eval dataset is available
            if self.eval_loader:
                eval_loss, eval_perf = self.evaluate()
                eval_losses.append(eval_loss)
                print(f"Eval Loss = {eval_loss:.4f}")                                        
                print(f"Eval Perf = {eval_perf:.4f}")         
                
            print('\nElapsed time: {:1.2f} seconds!'.format(time.time() - start))
        
        return train_losses, eval_losses


class SetTester:
    def __init__(self, 
                 model, 
                 dataset, 
                 loss_fn = None,  
                 perf_fn = None,
                 collate_fn = None,                
                 batch_size=32, 
                 device='cpu'):
        """
        Initialize the ModelTester.

        Args:
            model (torch.nn.Module): The PyTorch model to be tested.
            dataset (torch.utils.data.Dataset): The dataset to evaluate on.
            loss_fn (callable): The loss function.
            batch_size (int, optional): Batch size for testing. Default is 32.
            device (str, optional): The device to use ('cuda' or 'cpu'). If None, automatically selects the available device.
        """        
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        self.dataset = dataset        
        self.batch_size = batch_size        
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn if collate_fn is not None else set_collate_fn
        )
        
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.perf_fn = perf_fn
        

    def step(self, batch):
        x, m, y = batch
        
        # Set device
        x = x.to(self.device)
        m = m.to(self.device)
        y = y.to(self.device)
                
        y_pred = self.model(x, m)
        loss = self.loss_fn(y_pred, y)
        if self.perf_fn is not None:
            perf = self.perf_fn(y_pred, y)
        else:
            perf = loss.detach()
        return loss, perf        

    def test(self):
        """
        Run the model on the dataset and compute the average loss.

        Returns:
            dict: A dictionary containing the average loss and, if applicable, accuracy.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        total_perf = 0.0
        with torch.no_grad():  # Disable gradient computation
            for batch in self.dataloader:
                loss, perf = self.step(batch)
                total_loss += loss.item()
                total_perf += perf.item()
        
        average_loss = total_loss / len(self.dataloader)     
        average_perf = total_perf / len(self.dataloader)     
        return average_loss, average_perf
                        
# class SetModelTrainer:
#     def __init__(self, 
#                  model, 
#                  dataset,
#                  loss_func,
#                  eval_dataset = None,
#                  batch_size = 128,
#                  learning_rate = 1.e-4,
#                  decay = 1e-5):
        
#         self.model = model
#         self.loss_func = loss_func
       
#        # Optimizer
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr = learning_rate,
#             weight_decay = decay
#         )
        
        
        
#     def train(self, steps: int):
#         for step in range(steps):
#             loss = self.step()
#             print('Step {}/{}; Loss: {:1.3f}'.format(step, steps, loss))        
#         return loss
    
#     def sample(self):
#         k = random.randint(2, self.max_k)
#         X = torch.rand(k, self.model.input_dim)
#         y = self.agg_func(X)
#         return X, y
            
#     def step(self):    
#         # Init containers
#         targets, estimates = [], []
        
#         # Populate batch and apply model
#         for _ in range(self.batch_size):
#             X, y = self.sample()
#             y_hat = self.model(X)
#             targets.append(y)
#             estimates.append(y_hat)
            
        
#         # Store targets and estimates
#         targets, estimates = torch.cat(targets), torch.cat(estimates)
        
#         # Calculate loss
#         loss = self.loss_func(estimates, targets)
        
#         # Backpropagate
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()        
        
#         return loss.item()


class ImageClassifierTrainer:
    def __init__(self, model, train_dataset, val_dataset,
                 lr: float = 1e-3, batch_size: int = 64, device: str = None):
        """
        Parameters:
        - model: the MNISTLabelEstimator model
        - train_dataset: PyTorch Dataset for training
        - val_dataset: PyTorch Dataset for validation/testing
        - lr: learning rate
        - batch_size: batch size
        - device: 'cuda' or 'cpu' (automatically selected if not provided)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _accuracy(self, outputs, targets):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)

    def calc_loss(self, inputs, targets):
        # Set to device
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Account for more structured targets            
        if len(targets.shape) == 3: 
            targets = targets.squeeze(1).argmax(dim = 1)
            
        # Get outputs
        outputs = self.model(inputs)
        
        # Calc loss
        loss = self.criterion(outputs, targets)

        # Calc accuracy        
        acc = (outputs.argmax(dim=1) == targets).sum() / inputs.size(0)
                
        return loss, acc

    def train(self, epochs: int = 10, verbosity = 10):
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            total_samples = 0
            num_batches = len(self.train_loader)
            
            print(f"\nEpoch {epoch}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                loss, acc = self.calc_loss(inputs, targets)

                self.optimizer.zero_grad()                                
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item() * inputs.size(0)
                train_acc_sum += acc.item() * inputs.size(0)
                total_samples += inputs.size(0)

                if batch_idx % verbosity == 0:
                    print(f"[Batch {batch_idx + 1}/{num_batches}] Train Loss: {loss.item():.4f}")

            train_loss_avg = train_loss_sum / total_samples
            train_accuracy = train_acc_sum / total_samples

            val_loss, val_accuracy = self.evaluate()
            
            print(f"\n[Epoch {epoch}] Train Loss: {train_loss_avg:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}")

    def evaluate(self):
        self.model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                loss, acc = self.calc_loss(inputs, targets)
                
                val_loss_sum += loss.item() * inputs.size(0)
                val_acc_sum += acc.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = val_loss_sum / total_samples
        accuracy = val_acc_sum / total_samples
        return avg_loss, accuracy
    