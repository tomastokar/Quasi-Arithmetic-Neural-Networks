import torch
import random
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from torch_geometric.datasets import ModelNet


class SetDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): Each element is (x, y)
                - x: A tensor of VARIABLE length along the first dimension, but FIXED in remaining dimensions.
                - y: A tensor (or scalar) with length 1.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (x, y)


def set_collate_fn(batch):
    """
    Custom collate function for handling variable-length x tensors.
    - Pads x along the first dimension.
    - Creates a mask for padding.
    - Stacks y tensors.
    """
    # Unzip batch into separate lists
    x_list, y_list = zip(*batch) 

    # Determine max sequence length for padding
    max_len = max(x.shape[0] for x in x_list)

    # Get feature dimensions (excluding first variable dimension)
    feature_dims = x_list[0].shape[1:]

    # Determine bath size
    batch_size = len(x_list)

    # Create padded tensor and mask    
    padded_x = torch.zeros((batch_size, max_len, *feature_dims))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, x in enumerate(x_list):
        length = x.shape[0]  # Original length before padding
        padded_x[i, :length] = x  # Copy data
        mask[i, :length] = 1  # Mark genuine values

    # Convert y into tensor
    y_tensor = torch.vstack([y for y in y_list])

    # Unsqueeze if needed
    if len(y_tensor.shape) == 1:
        y_tensor = y_tensor.unsqueeze(1)

    return padded_x, mask, y_tensor


def generate_point_cloud(n_points, n_features, center_high=10., center_low=-10., scale_max=10.0):
    """
    Generate a random point cloud with a varying center of gravity and diameter.
    
    Parameters:
    - n_points: The number of points in the point cloud.
    - n_features: The dimensionality of each point (number of features).
    - center: The center of gravity of the point cloud (torch tensor of shape (n_features,)). If None, will be random.
    - diameter: The diameter of the point cloud. If None, it will be randomly chosen between 1 and 10.
    
    Returns:
    - A tensor of shape (n_points, n_features) representing the generated point cloud.
    """

    # Random center of gravity (mean) in the feature space
    center = (center_high - center_low) * torch.rand(n_features) + center_low
    
    # Random diameter between 1 and 10
    scale = torch.rand(1).item() * scale_max + 1.
    
    # # Generate random points uniformly distributed in a sphere with the specified diameter
    # # First, generate random points uniformly distributed in the unit sphere
    points = torch.randn(n_points, n_features)
    # points = points / points.norm(p=2, dim=1, keepdim=True)  # Normalize to unit sphere
    
    # # Scale the points to achieve the desired diameter
    # points = points * diameter / 2  # Diameter is twice the radius, so scale by diameter/2
    
    # Shift the points to the specified center
    points = (points * scale) + center
    
    return points
    

def make_vector_aggregates(agg_func: callable, dim: int = 16, num: int = 10000, min_k: int = 1, max_k: int = 256):
    data = []
    for _ in range(num):
        k = random.randint(min_k, max_k)
        x = generate_point_cloud(k, dim)
        # x = torch.rand(k, dim) 
        y = agg_func(x)
        data.append((x, y))

    dataset = SetDataset(data)    
    return dataset
    
    
def stack_MNIST(dataset, num_samples, min_k = 1, max_k = 1024, aggregation_fn=torch.mean):
    """
    Creates a list of tensors where each tensor contains stacked MNIST images, 
    and labels are aggregated using the provided function.
    
    Args:
        dataset (torch.utils.data.Dataset): The MNIST dataset.
        num_samples (int): Number of stacked tensors to generate.
        K (int): Maximum number of images per stacked tensor.
        aggregation_fn (function): Function to aggregate labels (default is mean).
    
    Returns:
        List of tuples: [(stacked_tensor, aggregated_label), ...]
    """
    data_list = []
    
    for _ in range(num_samples):
        # Randomly sample the number of images (between 1 and K)
        num_images = random.randint(min_k, max_k)
        
        # Randomly select images from dataset
        indices = random.sample(range(len(dataset)), num_images)
        images, labels = zip(*[dataset[i] for i in indices])
        
        # Stack images into a single tensor (shape: [num_images, 1, 28, 28])
        stacked_tensor = torch.stack(images)  # Shape: (num_images, 1, 28, 28)
        
        # Aggregate labels using the specified function
        aggregated_label = aggregation_fn(torch.tensor(labels).float())
        
        # Store the result
        data_list.append((stacked_tensor, aggregated_label))
    
    return data_list


def create_stacked_MNIST(train_size: int = 10000, eval_size: int = 3000, test_size: int = 1000, 
                         min_k = 2, max_k = 1024, agg_func: callable = torch.mean):
    
    # Define transform
    transform = transforms.Compose([transforms.ToTensor()])
        
    # Load MNIST dataset
    mnist = {}
    
    # Train portion
    mnist['train'] = datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )

    # Test portion
    mnist['test'] = datasets.MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=transform
    )

    num_samples = {
        'train' : train_size, 
        'eval' : eval_size,
        'test' : test_size
    }

    # Generate stacked MNIST datasets    
    data = {}        
    for name, n in num_samples.items():
        d = stack_MNIST(
            mnist['train' if name != 'test' else 'test'], 
            num_samples=n, 
            min_k=min_k,
            max_k=max_k,
            aggregation_fn=agg_func
        )
        
        data[name] = SetDataset(d)
    
    return data


def fetch_omniglot():
    # Define a transform to convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single-channel grayscale        
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Download and load the Omniglot dataset
    train_dataset = datasets.Omniglot(
        root="./data",
        background=True,  # True for the background set (training), False for evaluation
        download=True,
        transform=transform
    )

    test_dataset = datasets.Omniglot(
        root="./data",
        background=False,  # False for the evaluation set
        download=True,
        transform=transform
    )
    
    # Extract character-alphabet mapping
    character_mapping = []
    for character in train_dataset._characters:
        alphabet, char_name = character.split("/")
        character_mapping.append({"Character": char_name, "Alphabet": alphabet})
        
    for character in test_dataset._characters:
        alphabet, char_name = character.split("/")
        character_mapping.append({"Character": char_name, "Alphabet": alphabet})    
    
    # Convert to DataFrame and save as CSV
    character_mapping = pd.DataFrame(character_mapping)
    
    # Assign numerical encoding to the Alphabet column
    character_mapping["Alphabet_ID"] = character_mapping["Alphabet"].astype('category').cat.codes    
    
    # Create dictionary mapping character ID (index) to Alphabet ID
    character_mapping = {idx: row["Alphabet_ID"] for idx, row in character_mapping.iterrows()}    
    
    # Combine train and test datasets into a single dataset
    full_dataset = ConcatDataset([train_dataset, test_dataset])
        
    return full_dataset, character_mapping


def create_stacked_Omniglot(train_size: int = 20000, 
                            eval_size: int = 3000, 
                            test_size: int = 1000, 
                            min_k = 5, max_k = 50):
    
    # Fetch data
    omniglot, character_mapping = fetch_omniglot()
    
    # Init data list
    data_list = []
    
    total_size = train_size + eval_size + test_size
    for _ in range(total_size):
        # Randomly sample the number of images (between 1 and K)
        num_images = random.randint(min_k, max_k)
        
        # Randomly select images from dataset
        indices = random.sample(range(len(omniglot)), num_images)
        images, labels = zip(*[omniglot[i] for i in indices])
        
        # Map character ids to alphabet ids
        labels = [character_mapping[l] for l in labels]
        
        # Stack images into a single tensor (shape: [num_images, 1, 28, 28])
        stacked_tensor = torch.stack(images)  # Shape: (num_images, 1, 28, 28)
        
        # Create labels 
        target = torch.zeros(1, 50)
        target[:,labels] = 1.0
                 
        # Store the result
        data_list.append((stacked_tensor, target))
        
    data = {
        'train' : data_list[:train_size],
        'eval' : data_list[train_size:(train_size + eval_size)],
        'test' : data_list[-test_size:],        
    }
    
    return data

    
class ModelNet40Dataset(Dataset):
    def __init__(self, root="data/ModelNet40", split="train", dataset_size=10000, min_k=512, max_k=1024, seed=42, normalize = True):
        """
        A PyTorch dataset for ModelNet40 with random subsampling.

        Args:
            root (str): Path to the dataset.
            split (str): "train" or "test".
            dataset_size (int): Number of samples in the dataset.
            min_k (int): Minimum number of points per sample.
            max_k (int): Maximum number of points per sample.
            seed (int): Random seed for reproducibility.
        """      
        self.root = root  
        self.min_k = min_k
        self.max_k = max_k
        self.normalize = True
        self.split = split
        self.dataset_size = dataset_size
        self.normalize = normalize
        
        # Ensure reproducibility
        random.seed(seed)
        
        # Build dataset
        self.select_data()
        
    def subsample(self, points):
        # Randomly choose number of points for this sample
        k = random.randint(self.min_k, self.max_k)

        # Subsample if necessary
        if points.shape[0] > k:
            indices = torch.randperm(points.shape[0])[:k]  # Random subset
            points = points[indices]
            
        return points
        
    def select_data(self):
        
        # Load data
        data = ModelNet(root=self.root, name="40", train=(self.split == "train"))
        
        # Sample dataset_size random indices
        indices = random.choices(range(len(data)), k=self.dataset_size)
        
        # Init dataset
        self.dataset = []
        
        # Select instances
        for idx in indices:
            # Select instance
            d = data[idx]
            points, label = d.pos, d.y  # (N, 3) point cloud and class label
                        
            # Subsample points
            points = self.subsample(points)
            
            # Normalize to unit sphere if enabled
            if self.normalize:
                points = self.normalize_point_cloud(points)
                
            # Add to dataset
            self.dataset.append((points, label))

    def normalize_point_cloud(self, points):
        """Normalizes a point cloud to fit within a unit sphere centered at (0, 0, 0)."""
        center = points.mean(dim=0)  # Compute centroid
        points = points - center  # Center the point cloud
        scale = torch.max(torch.norm(points, dim=1))  # Find max distance
        points = points / scale  # Scale to unit sphere
        return points
            
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        points, label = self.dataset[idx]
        return points, label


def modelnet_collate_fn(batch):
    """
    Custom collate function to pad variable-length point clouds
    and create a boolean mask.
    
    Args:
        batch (list of tuples): Each element is (points, label)
            - points: (num_points, 3) tensor
            - label: Scalar tensor
    
    Returns:
        padded_points: (batch_size, max_num_points, 3) tensor
        labels: (batch_size,) tensor
        mask: (batch_size, max_num_points) boolean tensor
    """
    points_list, labels = zip(*batch)  # Separate data and labels
    
    batch_size = len(points_list)
    max_num_points = max(p.shape[0] for p in points_list)  # Find max num of points in batch
    
    # Initialize padded tensors
    padded_points = torch.zeros((batch_size, max_num_points, 3), dtype=torch.float32)  # (B, max_N, 3)
    mask = torch.zeros((batch_size, max_num_points), dtype=torch.bool)  # (B, max_N) Boolean mask
    
    for i, points in enumerate(points_list):
        num_points = points.shape[0]
        padded_points[i, :num_points, :] = points  # Copy real points
        mask[i, :num_points] = True  # Mark real points as True
    
    labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor
    
    return padded_points, mask, labels
        

def create_modelnet40_splits(train_size: int = 20000, 
                             eval_size: int = 3000, 
                             test_size: int = 1000, 
                             min_k=512, max_k=1024, 
                             root="data/ModelNet40"):
    """
    Creates train, validation, and test splits for ModelNet40.

    Args:
        root (str): Root directory for ModelNet40 dataset.
        min_k (int): Minimum number of points per sample.
        max_k (int): Maximum number of points per sample.
        val_ratio (float): Fraction of training set to be used as validation.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with 'train', 'eval', and 'test' datasets.
    """
    # Load full training and test datasets
    
    data = {}
    for name, size in zip(['train', 'eval', 'test'], [train_size, eval_size, test_size]):    
        data[name] = ModelNet40Dataset(root=root, dataset_size=size, split=name, min_k=min_k, max_k=max_k)
        
    return data

    
def get_MNIST_test_images(num_images = 3000):
    data = datasets.MNIST(
        root="./data", 
        train=False, 
        download = True, 
        transform=transforms.Compose([transforms.ToTensor()])
    )
        
    loader = DataLoader(
        data, 
        shuffle = False, 
        batch_size = num_images
    )
    
    X, y = next(iter(loader))
    
    # Convert to batch with 1 set of num_images
    X = X.unsqueeze(0) 
    
    return X, y

    