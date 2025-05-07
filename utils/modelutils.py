import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dims: list = []):
        super(MLP, self).__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RevNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims = [256,]):
        super(RevNetBlock, self).__init__()
        
        # Generate a fixed random split at initialization
        split_indices = torch.randperm(input_dim)[:input_dim // 2]  # Random half
        self.input_dim = input_dim
        self.mask = torch.zeros(input_dim, dtype=torch.bool)
        self.mask[split_indices] = True  # Boolean mask for indexing

        # Define transformation functions F and G           
        self.F = MLP(input_dim // 2, input_dim // 2, hidden_dims)
        self.G = MLP(input_dim // 2, input_dim // 2, hidden_dims)

    def forward(self, x):
        """
        Forward pass implementing:
            y1 = x1 + F(x2)
            y2 = x2 + G(y1)
        """
        x1, x2 = x[..., self.mask], x[..., ~self.mask]
        
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)

        y = torch.empty_like(x)
        y[..., self.mask] = y1
        y[...,~self.mask] = y2
        
        return y

    def inverse(self, y):
        """
        Inverse pass implementing:
            x2 = y2 - G(y1)
            x1 = y1 - F(x2)
        """
        y1, y2 = y[..., self.mask], y[..., ~self.mask]

        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)

        x = torch.empty_like(y)
        x[..., self.mask] = x1
        x[...,~self.mask] = x2
        
        return x


class RevNet(nn.Module):
    def __init__(self, input_dim, num_blocks, hidden_dims=[256,]):
        super(RevNet, self).__init__()
        self.input_dim = input_dim
        self.blocks = nn.ModuleList([
            RevNetBlock(input_dim, hidden_dims) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """Forward pass through all RevNetBlocks."""
        for block in self.blocks:
            x = block(x)
        return x
    
    def inverse(self, y):
        """Inverse pass through all RevNetBlocks in reverse order."""
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y


class NKM(nn.Module):
    def __init__(self, input_dim: int, num_blocks: int = 2, psi_dims: list = [128,]):
        super(NKM, self).__init__()        
        self.psi = RevNet(input_dim=input_dim, num_blocks = num_blocks, hidden_dims=psi_dims)
        self.input_dim = self.psi.input_dim
        
    def forward(self, x: torch.tensor, mask: torch.tensor):
        h = self.psi(x)
        h[~mask] = 0
        mu = h.mean(dim = 1)
        return self.psi.inverse(mu)


class DeepSet(nn.Module):
    def __init__(self, encoder: nn.Module = None, predictor: nn.Module = None, 
                 latent_dim: int = 32, input_dim: int = None, output_dim: int = None, hidden_dims = [256,]):
        super(DeepSet, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide predictor or output_dim.")

    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.tensor, mask: torch.tensor):
        h = self.encoder(x)
        h[~mask] = 0
        mu = h.sum(dim = 1)
        return self.predictor(mu)


class MeanSet(nn.Module):
    def __init__(self, encoder: nn.Module = None, predictor: nn.Module = None, 
                 latent_dim: int = 32, input_dim: int = None, output_dim: int = None, hidden_dims = [256,]):
        
        super(MeanSet, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide predictor or output_dim.")

    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.tensor, mask: torch.tensor):
        h = self.encoder(x)
        h[~mask] = 0
        mu = h.mean(dim = 1)
        return self.predictor(mu)


class HPDS(nn.Module):
    def __init__(self, 
                 encoder: nn.Module = None, 
                 predictor: nn.Module = None, 
                 latent_dim: int = 32, 
                 input_dim: int = None, 
                 output_dim: int = None, 
                 hidden_dims = [256,],
                 p_init = 1.0):
        
        super(HPDS, self).__init__()
        
        self.p = nn.Parameter(torch.tensor(p_init))
        
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide predictor or output_dim.")

    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.tensor, mask: torch.tensor):        
        h = self.encoder(x)                
        h[~mask] = 0       
        h = torch.pow(F.softplus(h), self.p) 
        mu = h.mean(dim = 1)       
        mu = torch.pow(mu.clamp(min=1e-6), 1. / (self.p + 1e-6))  
        return self.predictor(mu)


class PointNet(nn.Module):
    def __init__(self, encoder: nn.Module = None, predictor: nn.Module = None, 
                 latent_dim: int = 32, input_dim: int = None, output_dim: int = None, hidden_dims = [256,]):
        
        super(PointNet, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide predictor or output_dim.")

    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.tensor, mask: torch.tensor):
        h = self.encoder(x)
        h[~mask] = 0
        mu = h.max(dim = 1).values
        return self.predictor(mu)


class SmallImageEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(SmallImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(64 * 3 * 3, latent_dim)
    
    def forward(self, x):
        batch_size, set_size, c, h, w = x.shape  # Expecting (batch_size, set_size, 1, 28, 28)
        x = x.view(batch_size * set_size, c, h, w)  # Flatten set dimension
        x = self.conv(x)
        x = x.view(batch_size * set_size, -1)
        x = self.fc(x)
        x = x.view(batch_size, set_size, -1)  # Reshape back to (batch_size, set_size, latent_dim)
        return x


class QUANN(nn.Module):
    def __init__(self, 
                 encoder: nn.Module = None, 
                 predictor: nn.Module = None, 
                 latent_dim: int = 32, 
                 input_dim: int = None, 
                 output_dim: int = None, 
                 num_blocks: int = 1,
                 num_nkms: int = 1, 
                 psi_dims: list = [128,],
                 hidden_dims:list = [128,]) -> None:
        
        super(QUANN, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide decoder or output_dim.")
        
        self.nkms = nn.ModuleList(
            [
                NKM(input_dim=latent_dim, num_blocks = num_blocks, psi_dims=psi_dims)
                for _ in range(num_nkms)
            ]
        )
                    
    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        z = self.encoder(x)
        mu = sum([nkm(z, mask) for nkm in self.nkms])
        out = self.predictor(mu)
        return out  


class MAB(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fc = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, Q, K):
        Q = self.ln1(Q + self.multihead_attn(Q, K, K, need_weights=False)[0])
        return self.ln2(Q + self.fc(Q))

class SAB(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mab = MAB(dim, num_heads)
    
    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads=4, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, num_heads)
    
    def forward(self, X):
        S = self.S.repeat(X.size(0), 1, 1)  # Repeat for batch size
        return self.mab(S, X)


class SetTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=128, num_heads=1, hidden_dims = [], projector: nn.Module = None):
        super().__init__()
        if projector is not None:
            self.projector = projector
        else:
            self.projector = nn.Linear(input_dim, latent_dim)
            
        self.encoder = SAB(latent_dim, num_heads)
        self.pooling = PMA(latent_dim, num_heads, num_seeds=1)
        self.fc_out = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
        
    
    def forward(self, X: torch.tensor, mask: torch.tensor):
        X = self.projector(X)                       
        X = self.encoder(X)                     
        X[~mask] = 0               
        X = self.pooling(X)                  
        return self.fc_out(X).squeeze(1)


class QUANNTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 latent_dim: int = 128, 
                 num_heads: int = 1, 
                 num_blocks: int = 1,
                 num_nkms: int = 1,
                 psi_dims: list = [256,], 
                 hidden_dims: list = [],
                 projector: nn.Module = None):
        
        super().__init__()
        if projector is not None:
            self.projector = projector
        else:
            self.projector = nn.Linear(input_dim, latent_dim)
            

        self.encoder = SAB(latent_dim, num_heads)        
        
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
        
        self.nkms = nn.ModuleList(
            [
                NKM(input_dim=latent_dim, num_blocks = num_blocks, psi_dims=psi_dims)
                for _ in range(num_nkms)
            ]
        )
        
    
    def forward(self, X: torch.tensor, mask: torch.tensor):
        X = self.projector(X)                       
        X = self.encoder(X)                            
        mu = sum([nkm(X, mask) for nkm in self.nkms])
        return self.predictor(mu)


class ImageClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int, output_dim: int):
        """
        Parameters:
        - encoder: nn.Module that takes MNIST images and returns latent embeddings
        - latent_dim: dimension of the encoder's output (latent representation)
        - num_classes: number of classes for classification (default is 10 for MNIST)
        """
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the encoder and predictor.

        Parameters:
        - x: input tensor representing MNIST images, shape (batch_size, 1, 28, 28)

        Returns:
        - logits: raw scores for each class, shape (batch_size, num_classes)
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1) # Encoder expects a set of images - convert to batch of n = 1 sets
        latent = self.encoder(x)               # shape: (batch_size, latent_dim)
        logits = self.predictor(latent)        # shape: (batch_size, num_classes)
        return logits.squeeze(1)
    
    
class RealNVPCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256]):
        super(RealNVPCoupling, self).__init__()

        # Define a fixed random binary mask (same structure as in RevNetBlock)
        split_indices = torch.randperm(input_dim)[:input_dim // 2]
        self.input_dim = input_dim
        self.mask = torch.zeros(input_dim, dtype=torch.bool)
        self.mask[split_indices] = True  # true = x1, false = x2

        # Scale (s) and shift (t) networks
        self.scale_net = MLP(input_dim // 2, input_dim // 2, hidden_dims)
        self.shift_net = MLP(input_dim // 2, input_dim // 2, hidden_dims)

    def forward(self, x):
        """
        Forward pass:
            x1 = x[mask], x2 = x[~mask]
            s, t = scale_net(x1), shift_net(x1)
            y1 = x1
            y2 = x2 * exp(s) + t
        """
        x1, x2 = x[..., self.mask], x[..., ~self.mask]

        s = self.scale_net(x1)
        t = self.shift_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t

        y = torch.empty_like(x)
        y[..., self.mask] = y1
        y[..., ~self.mask] = y2
        return y

    def inverse(self, y):
        """
        Inverse pass:
            y1 = y[mask], y2 = y[~mask]
            s, t = scale_net(y1), shift_net(y1)
            x1 = y1
            x2 = (y2 - t) * exp(-s)
        """
        y1, y2 = y[..., self.mask], y[..., ~self.mask]

        s = self.scale_net(y1)
        t = self.shift_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[..., self.mask] = x1
        x[..., ~self.mask] = x2
        return x


class RealNVP(nn.Module):
    def __init__(self, input_dim, num_blocks, hidden_dims=[256]):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.blocks = nn.ModuleList([
            RealNVPCoupling(input_dim, hidden_dims) for _ in range(num_blocks)
        ])

    def forward(self, x):
        """Forward pass through all RealNVP coupling layers."""
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        """Inverse pass through all RealNVP blocks in reverse order."""
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y


class NKM_nvp(nn.Module):
    def __init__(self, input_dim: int, num_blocks: int = 2, psi_dims: list = [128,]):
        super(NKM_nvp, self).__init__()        
        self.psi = RealNVP(input_dim=input_dim, num_blocks = num_blocks, hidden_dims=psi_dims)
        self.input_dim = self.psi.input_dim
        
    def forward(self, x: torch.tensor, mask: torch.tensor):
        h = self.psi(x)
        h[~mask] = 0
        mu = h.mean(dim = 1)
        return self.psi.inverse(mu)


class QUANN_nvp(nn.Module):
    def __init__(self, 
                 encoder: nn.Module = None, 
                 predictor: nn.Module = None, 
                 latent_dim: int = 32, 
                 input_dim: int = None, 
                 output_dim: int = None, 
                 num_blocks: int = 1,
                 num_nkms: int = 1, 
                 psi_dims: list = [128,],
                 hidden_dims:list = [128,]) -> None:
        
        super(QUANN_nvp, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide decoder or output_dim.")
        
        self.nkms = nn.ModuleList(
            [
                NKM_nvp(input_dim=latent_dim, num_blocks = num_blocks, psi_dims=psi_dims)
                for _ in range(num_nkms)
            ]
        )
                    
    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        z = self.encoder(x)
        mu = sum([nkm(z, mask) for nkm in self.nkms])
        out = self.predictor(mu)
        return out  