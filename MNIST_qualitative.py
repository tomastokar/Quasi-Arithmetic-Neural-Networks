import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from utils.datautils import create_stacked_MNIST, get_MNIST_test_images
from utils.modelutils import SmallImageEncoder, QUANN, DeepSet, MeanSet, HPDS, PointNet, SetTransformer, QUANNTransformer
from utils.trainutils import SetTrainer, SetTester
from utils.auxutils import set_device, count_params, make_output_folder, set_seed

BATCH_SIZE = 32
LEARNING_RATEs = {
    'DeepSet' : 5e-4,
    'PointNet' : 5e-4,
    'MeanSet' : 5e-4,
    'HPDS' : 1e-4,
    'QUANN' : 1e-3,
    'SetTransformer' : 1e-4,
    'QUANNTransformer' : 1e-3
}
WEIGHT_DECAY = 1e-5
LATENT_DIM = 128
NUM_EPOCHS = 50
NUM_REPLICATES = 4
OUTPUT_DIR = './results/MNIST_qualitative/'
NUM_IMAGES = 1000
AGG_FUNCS = {
    'mean' : torch.mean,
    'mode' : lambda x: x.mode().values,
    'variance' : torch.var,
    'max' : torch.max,
    'sum' : torch.sum
}

def get_embeddings(X, y, encoder, device = 'cpu'):    
    with torch.no_grad():
        X = X.to(device)
        z = encoder(X)        
        z = z.squeeze(0)
        z_np = z.cpu().numpy()
        y_np = y.cpu().numpy()
        
    return {'z' : z_np, 'y' : y_np}
    
def main():
    # Set seed
    set_seed()
    
    # Set device
    device = set_device(0)
    
    # Init output folder
    make_output_folder(OUTPUT_DIR)
    
    # Fetch MNIST images and labels
    mnist_X, mnist_y = get_MNIST_test_images(NUM_IMAGES)
    
    # Init results container        
    results = []
    embeddings = {}
        
    for agg_func_name, agg_func in AGG_FUNCS.items():

        data = create_stacked_MNIST(
            train_size=20000,
            eval_size=2000,
            test_size=3000,
            min_k=2, 
            max_k=16,
            agg_func=agg_func
        )
                    
        models = {}

        models['DeepSet'] = DeepSet(
            encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
            output_dim=1, 
            latent_dim=LATENT_DIM,
            hidden_dims=[256,256]
        )        

        models['MeanSet'] = MeanSet(
            encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
            output_dim=1, 
            latent_dim=LATENT_DIM,
            hidden_dims=[256,256]
        )        
        
        models['PointNet'] = PointNet(
            encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
            output_dim=1, 
            latent_dim=LATENT_DIM,
            hidden_dims=[256,256]
        )

        models['HPDS'] = HPDS(
            encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
            output_dim=1, 
            latent_dim=LATENT_DIM,
            hidden_dims=[256,256]
        )

        models['QUANN'] = QUANN(
            encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
            output_dim=1, 
            latent_dim=LATENT_DIM,
            num_blocks=2,
            hidden_dims=[256,],
            psi_dims=[128,]
        )
        
        models['SetTransformer'] = SetTransformer(
            input_dim=LATENT_DIM,
            output_dim=1,
            latent_dim=LATENT_DIM,
            num_heads=1,
            projector=SmallImageEncoder(latent_dim=LATENT_DIM)
        )
        
        models['QUANNTransformer'] = QUANNTransformer(
            input_dim=LATENT_DIM,
            output_dim=1,
            latent_dim=LATENT_DIM,
            num_heads=1,
            num_blocks=2,
            psi_dims=[128,],
            projector=SmallImageEncoder(latent_dim=LATENT_DIM)
        )
            
        for model_name, model in models.items():
            print(model_name, count_params(model))                
                            
        for model_name, model in models.items():
            
            # Select learning rate
            lr = LEARNING_RATEs[model_name]
            
            # Print statement
            print('\n')
            print('Model: {}'.format(model_name))
            print('Agg func: {}'.format(agg_func_name))
            print('Learning rate: {}'.format(lr))
                
            # Init trainer
            trainer = SetTrainer(
                model=model,
                dataset=data['train'],
                eval_dataset=data['eval'],
                batch_size=BATCH_SIZE,
                lr = lr,
                weight_decay=WEIGHT_DECAY,
                loss_fn=nn.MSELoss(),
                device = device                    
            )

            # Train model                    
            _, eval_loss = trainer.train(num_epochs = NUM_EPOCHS)    
            eval_loss = eval_loss[-1]
                
            # Init tester
            tester = SetTester(
                model=model, 
                dataset=data['test'],
                loss_fn=nn.MSELoss(),
                batch_size=BATCH_SIZE,
                device=device
            )
                
            # Test model
            test_loss, _ = tester.test()
            
            # Count model params
            num_params = count_params(model)
            
            # Add to results
            results.append([agg_func_name, model_name, lr, num_params, eval_loss, test_loss])
                
            # Extract encoder/projector from the model
            if model_name in ['SetTransformer', 'QUANNTransformer']:                        
                encoder = model.projector
            else:
                encoder = model.encoder
            
            # Encode the test set instances
            emb = get_embeddings(mnist_X, mnist_y, encoder, device = device)
            embeddings['{}/{}'.format(agg_func_name, model_name)] = emb

    # Create pandas data frame
    results = pd.DataFrame(
    results, 
    columns = ['Agg func', 'Model', 'Learning rate', 'Parameters', 'Eval loss', 'Test loss']
    )
        
    # Save to file
    fn = os.path.join(OUTPUT_DIR, 'results.csv')
    results.to_csv(fn)

    # Save embeddings
    fn = os.path.join(OUTPUT_DIR, 'embeddings.npz')            
    np.savez(fn, **embeddings)
    
if __name__ == '__main__':
    main()
    