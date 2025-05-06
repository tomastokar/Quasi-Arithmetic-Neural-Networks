import torch
from utils.auxutils import set_device, set_seed
from utils.modelutils import SmallImageEncoder, ImageClassifier
from utils.trainutils import ImageClassifierTrainer
from utils.datautils import create_stacked_Omniglot

LATENT_DIM = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 20
def main():
    
    # Set seed
    set_seed()
    
    # Set device
    device = set_device(0)
    
    # Init model
    encoder = SmallImageEncoder(latent_dim=LATENT_DIM)
    model = ImageClassifier(encoder=encoder, latent_dim=LATENT_DIM, output_dim=50)
            
    # Load MNIST    
    data = create_stacked_Omniglot(
        train_size=60000,
        eval_size=10000,
        test_size=0,
        min_k=1, 
        max_k=1
    )
        
    trainer = ImageClassifierTrainer(
        model, 
        data['train'], 
        data['eval'], 
        lr = LEARNING_RATE, 
        batch_size = BATCH_SIZE,
        device = device
    )
    
    trainer.train(epochs=NUM_EPOCHS)
    
    encoder_state_dict = model.encoder.state_dict()
    torch.save(encoder_state_dict, "./Omniglot_encoder_weights.pt")    
    

if __name__ == '__main__':
    main()