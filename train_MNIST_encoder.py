import torch
from utils.auxutils import set_device, set_seed
from utils.modelutils import SmallImageEncoder, ImageClassifier
from utils.trainutils import ImageClassifierTrainer
from torchvision import datasets, transforms

LATENT_DIM = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10
def main():
    
    # Set seed
    set_seed()
    
    # Set device
    device = set_device(0)
    
    # Init model
    encoder = SmallImageEncoder(latent_dim=LATENT_DIM)
    model = ImageClassifier(encoder=encoder, latent_dim=LATENT_DIM, output_dim=10)
        
    # Define transformnation
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load MNIST    
    data_train = datasets.MNIST( # Train portion
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    data_eval = datasets.MNIST( # Eval portion
        root="./data", 
        train=False, 
        download=True, 
        transform=transform
    )    
    
    trainer = ImageClassifierTrainer(
        model, 
        data_train, 
        data_eval, 
        lr = LEARNING_RATE, 
        batch_size = BATCH_SIZE,
        device = device
    )
    
    trainer.train(epochs=NUM_EPOCHS)
    
    encoder_state_dict = model.encoder.state_dict()
    torch.save(encoder_state_dict, "./MNIST_encoder_weights.pt")    
    

if __name__ == '__main__':
    main()