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


class NKMSum(nn.Module):
    def __init__(self, input_dim: int, num_blocks: int = 2, psi_dims: list = [128,]):
        super(NKMSum, self).__init__()        
        self.psi = RevNet(input_dim=input_dim, num_blocks = num_blocks, hidden_dims=psi_dims)
        self.input_dim = self.psi.input_dim
        
    def forward(self, x: torch.tensor, mask: torch.tensor):
        h = self.psi(x)
        h[~mask] = 0
        mu = h.sum(dim = 1)  # Sum instead of mean for abblation experiments
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


class QUANNSum(nn.Module):
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
        
        super(QUANNSum, self).__init__()
        
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
        
        self.nkm_sums = nn.ModuleList(
            [
                NKMSum(input_dim=latent_dim, num_blocks = num_blocks, psi_dims=psi_dims)
                for _ in range(num_nkms)
            ]
        )
                    
    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        z = self.encoder(x)
        mu = sum([nkm_sum(z, mask) for nkm_sum in self.nkm_sums])
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
    

class FSPool(nn.Module):
    def __init__(self, input_dim, pool_size):
        """
        Args:
            input_dim: Number of feature dimensions (D)
            pool_size: Maximum number of set elements (N), i.e., the assumed max cardinality
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.pool_size = pool_size

        # Learnable weights (one per rank position)
        self.weights = nn.Parameter(torch.randn(pool_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, N, D) — batch of sets
            mask: Optional binary mask of shape (B, N) (1 = valid, 0 = pad)
        Returns:
            Tensor of shape (B, D) — pooled representation
        """
        B, N, D = x.shape
        device = x.device

        if mask is not None:
            # Replace padded entries with -inf for sorting
            # mask = mask.unsqueeze(-1)  # (B, N, 1)
            # x = x.masked_fill(mask == 0, float('-inf'))
            x[~mask] = float('-inf')

        # Sort values along the N (set element) axis — per feature
        x_sorted, _ = x.sort(dim=1, descending=True)  # (B, N, D)
        
        # Remove infinities back to zeros
        x_sorted[~mask] = 0.
        
        # Truncate or pad sorted x to self.pool_size
        if N < self.pool_size:
            pad = torch.full((B, self.pool_size - N, D), 0., device=device)
            x_sorted = torch.cat([x_sorted, pad], dim=1)
        elif N > self.pool_size:
            x_sorted = x_sorted[:, :self.pool_size, :]
            
        # Apply softmax over weights for stability
        w = F.softmax(self.weights, dim=0)  # (pool_size,)

        # Weighted sum: sum_i w_i * x_sorted[:, i, :]
        x_pooled = torch.einsum('n,bnd->bd', w, x_sorted)  # (B, D)
        
        return x_pooled


class FSPoolSetfunc(nn.Module):
    def __init__(self, 
                 encoder: nn.Module = None, 
                 predictor: nn.Module = None, 
                 latent_dim: int = 32, 
                 input_dim: int = None, 
                 output_dim: int = None, 
                 pool_size: int = 20,
                 hidden_dims:list = [128,]) -> None:
        
        super(FSPoolSetfunc, self).__init__()
        
        # Encoder
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        # Predictor
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide decoder or output_dim.")
        
        # Pooling
        self.pool = FSPool(latent_dim, pool_size)
        

    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):                
        z = self.encoder(x)                
        mu = self.pool(z, mask)        
        out = self.predictor(mu)
        return out  
             

class SlotAttention(nn.Module):
    def __init__(self, slot_dim: int, num_slots: int = 6, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters

        # Learnable initial slots (num_slots, slot_dim)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, slot_dim))

        # Project inputs
        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)

        # GRU and MLP for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def forward(self, x, mask=None):
        """
        x: (B, N, D)    - input set with padding
        mask: (B, N)    - binary mask (1 = valid, 0 = pad)
        Returns:
            slots: (B, num_slots, slot_dim)
        """
        B, N, D = x.shape
        eps = 1e-8

        # Normalize input
        x = self.norm_input(x)

        # Initialize slots from learnable Gaussian
        mu = self.slots_mu.expand(B, -1, -1)  # (B, num_slots, slot_dim)
        sigma = F.softplus(self.slots_sigma).expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots

            # Project queries, keys, values
            q = self.q_proj(self.norm_slots(slots))  # (B, num_slots, slot_dim)
            k = self.k_proj(x)                  # (B, N, slot_dim)
            v = self.v_proj(x)                  # (B, N, slot_dim)

            # Attention: (B, num_slots, N)
            attn_logits = torch.einsum('bid,bjd->bij', q, k) / (self.slot_dim ** 0.5)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
            attn = F.softmax(attn_logits, dim=-1) + eps

            # Normalize over input tokens
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Aggregate updates: (B, num_slots, slot_dim)
            updates = torch.einsum('bjn,bnd->bjd', attn, v)

            # Slot update via GRU
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)

            # Add feedforward update
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots.mean(dim = 1)


class SlotAttModel(nn.Module):
    def __init__(self, 
                 encoder: nn.Module = None, 
                 predictor: nn.Module = None,
                 latent_dim: int = 32, 
                 input_dim: int = None, 
                 output_dim: int = None, 
                 num_slots: int = 6,
                 slot_iters: int = 3,
                 hidden_dims:list = [128,]) -> None:

        super(SlotAttModel, self).__init__()
        
        # Encoder
        if encoder is not None:
            self.encoder = encoder            
        elif input_dim is not None and latent_dim is not None:
            self.set_default_encoder(input_dim, latent_dim, hidden_dims)
        else:
            raise ValueError("Either provide encoder or input_dim.")
        
        # Predictor
        if predictor is not None:
            self.predictor = predictor   
        elif output_dim is not None and latent_dim is not None:
            self.set_default_predictor(latent_dim, output_dim, hidden_dims)
        else:
            raise ValueError("Either provide decoder or output_dim.")
        
        # Pooling
        self.slot_attention = SlotAttention(latent_dim, num_slots, slot_iters)

    def set_default_encoder(self, input_dim, latent_dim, hidden_dims):
        self.encoder = MLP(input_dim, latent_dim, hidden_dims=hidden_dims)
    
    def set_default_predictor(self, latent_dim, output_dim, hidden_dims):
        self.predictor = MLP(latent_dim, output_dim, hidden_dims=hidden_dims)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        z = self.encoder(x)                
        mu = self.slot_attention(z, mask)       
        out = self.predictor(mu)
        return out  
        
        


# class RepSet(nn.Module):
#     def __init__(self, input_dim, output_dim, projector: nn.Module = None, num_hidden_sets: int = 32, hidden_set_size: int = 16):
#         super().__init__()
#         self.num_hidden_sets = num_hidden_sets
#         self.hidden_set_size = hidden_set_size
#         self.input_dim = input_dim

#         # Hidden sets: (num_hidden_sets, hidden_set_size, input_dim)
#         self.hidden_sets = nn.Parameter(torch.randn(num_hidden_sets, hidden_set_size, input_dim))
#         self.output_layer = nn.Linear(num_hidden_sets, output_dim)

#         # Projector of the inputs into 2d
#         if projector is not None:
#             self.projector = projector
#         else:
#             self.projector = nn.Linear(input_dim, latent_dim)        

#     def forward(self, x, mask):
#         """
#         Args:
#             x:     (B, D, N) — batch of padded input sets
#             mask:  (B, N) — binary mask (1 = valid, 0 = pad)
#         Returns:
#             (B, num_classes) — log probabilities
#         """
#         B = x.shape[0]        
#         P = self.num_hidden_sets
#         device = x.device

#         # Transpose x to (B, N, D)
#         x = x.transpose(1, 2).contiguous()  # (B, N, D)

#         # Prepare output container
#         rep_set_vecs = torch.zeros(B, P, device=device)

#         # Loop over batch only — hidden sets and similarities are vectorized
#         for b in range(B):
#             valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
#             if valid_idx.numel() == 0:
#                 continue  # skip empty set
#             xb = x[b, valid_idx]  # (n_b, D)

#             # Compute similarities with all hidden sets at once
#             # xb: (n_b, D), hidden_sets: (P, M, D)
#             # Output sim: (P, n_b, M)
#             sim = F.relu(torch.einsum('nd,pmd->pnm', xb, self.hidden_sets))  # (P, n_b, M)

#             # For each hidden set, solve max bipartite matching
#             for p in range(P):
#                 sim_p = sim[p].detach().cpu().numpy()  # (n_b, M)
#                 cost = -sim_p
#                 row_ind, col_ind = linear_sum_assignment(cost)
#                 match_score = sim_p[row_ind, col_ind].sum()
#                 rep_set_vecs[b, p] = match_score

#         logits = self.output_layer(rep_set_vecs)
#         return F.log_softmax(logits, dim=1)


# class RepSet(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, latent_dim: int, projector: nn.Module = None, num_hidden_sets: int = 32, hidden_set_size: int = 16):
#         super().__init__()
#         self.num_hidden_sets = num_hidden_sets
#         self.hidden_set_size = hidden_set_size
#         self.latent_dim = latent_dim

#         # Hidden sets: list of learnable prototype matrices (M x D)
#         self.hidden_sets = nn.ParameterList([
#             nn.Parameter(torch.randn(hidden_set_size, self.latent_dim))
#             for _ in range(num_hidden_sets)
#         ])

#         # Output head
#         self.output_layer = nn.Linear(num_hidden_sets, output_dim)
        
#         # Projector of the inputs into 2d
#         if projector is not None:
#             self.projector = projector
#         else:
#             self.projector = nn.Linear(input_dim, latent_dim)        
        

#     def forward(self, X: torch.tensor, mask: torch.tensor):
#         """
#         Args:
#             x: Tensor of shape (B, D, N) -- input sets with padding
#             mask: Tensor of shape (B, N) -- binary mask indicating valid set elements
#         Returns:
#             log-probabilities: Tensor of shape (B, num_classes)
#         """
        
        
#         B = X.shape[0]
#         device = X.device

#         reps = []

#         # Projection
#         X_ = self.projector(X)                       
        
#         for b in range(B):
#             # Select valid elements from the b-th set using the mask
#             valid_indices = mask[b].nonzero(as_tuple=True)[0]
#             if valid_indices.numel() == 0:
#                 # Handle empty set gracefully
#                 reps.append(torch.zeros(self.num_hidden_sets, device=device))
#                 continue

#             xb = X_[b, valid_indices, :].contiguous()  # (D, n_b)
#             # print(xb.shape)
#             # xb = xb.transpose(0, 1).contiguous()  # (n_b, D)
            
#             sim_vector = []
#             for H in self.hidden_sets:
#                 # H: (M, D), xb: (n_b, D)
#                 sim = torch.matmul(xb, H.T)  # (n_b, M)
#                 sim = F.relu(sim).detach().cpu().numpy()

#                 # Solve max bipartite matching
#                 cost_matrix = -sim
#                 cost, x_lap = lapjv(cost_matrix, extend_cost = True)
                
#                 D = torch.zeros((self.hidden_set_size, xb.shape[1]))
#                 for k in range(self.hidden_set_size):
#                     if x_lap[k] != -1:
#                         D[k, x_lap[k]] = 1
                        
#                 score = sim[row_ind, col_ind].sum()
#                 sim_vector.append(score)

#             reps.append(torch.tensor(sim_vector, dtype=torch.float32, device=device))

#         reps = torch.stack(reps, dim=0)  # (B, num_hidden_sets)
#         logits = self.output_layer(reps)  # (B, num_classes)
#         return F.log_softmax(logits, dim=1)