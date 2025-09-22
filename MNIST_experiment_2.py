import os
import copy
import torch
import pandas as pd
import torch.nn as nn

from utils.datautils import create_stacked_MNIST
from utils.modelutils import SmallImageEncoder, QUANN, DeepSet, MeanSet, HPDS, PointNet, SetTransformer, QUANNTransformer
from utils.trainutils import SetTrainer, SetTester
from utils.auxutils import set_device, count_params, make_output_folder, set_seed

BATCH_SIZE = 32
LEARNING_RATEs = [1e-4, 5e-4, 1e-3]
WEIGHT_DECAY = 1e-5
LATENT_DIM = 128
NUM_EPOCHS = 50
NUM_REPLICATES = 4
OUTPUT_DIR = './results/MNIST_experiment_2/'

AGG_FUNCS = {
    # 'median' : torch.median,
    'mean' : torch.mean,
    'mode' : lambda x: x.mode().values,
    # 'geometric_mean' : lambda x: x.prod().pow(1 / x.numel()),
    # 'log_mean_exp' : lambda x: torch.log(x + 1e-6).mean().exp(),
    # 'harmonic' : lambda x, dim=0, keepdim=False: (x.size(dim) / torch.sum(1.0 / (x + 1), dim=dim, keepdim=keepdim)),
    # 'midrange' : lambda x: (x.max() - x.min()) / 2.0,
    'variance' : torch.var,
    'max' : torch.max,
    'sum' : torch.sum,
    # 'sum_module_10' : lambda x: x.sum() % 10    
}

def main():
    
    # Set seed
    set_seed()
    
    # Set device
    device = set_device(0)
    
    # Setup output folder
    make_output_folder(OUTPUT_DIR)
    
    # Init image encoder
    encoder = SmallImageEncoder(latent_dim=LATENT_DIM)
    
    # Load paramters
    encoder.load_state_dict(torch.load("MNIST_encoder_weights.pt"))
    
    # Freeze all the weights
    for param in encoder.parameters():
        param.requires_grad = False    
    
    for replicate in range(NUM_REPLICATES):
        
        results = []
        
        for agg_func_name, agg_func in AGG_FUNCS.items():
    
            data = create_stacked_MNIST(
                train_size=20000,
                eval_size=2000,
                test_size=3000,
                min_k=2, 
                max_k=16,
                agg_func=agg_func
            )
            
            for lr in LEARNING_RATEs:
            
                models = {}

                models['DeepSet'] = DeepSet(
                    encoder=copy.deepcopy(encoder), 
                    output_dim=1, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )        

                models['MeanSet'] = MeanSet(
                    encoder=copy.deepcopy(encoder), 
                    output_dim=1, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )        
                
                models['PointNet'] = PointNet(
                    encoder=copy.deepcopy(encoder), 
                    output_dim=1, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )

                models['HPDS'] = HPDS(
                    encoder=copy.deepcopy(encoder), 
                    output_dim=1, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )

                models['QUANN'] = QUANN(
                    encoder=copy.deepcopy(encoder), 
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
                    projector=copy.deepcopy(encoder)
                )
                
                models['QUANNTransformer'] = QUANNTransformer(
                    input_dim=LATENT_DIM,
                    output_dim=1,
                    latent_dim=LATENT_DIM,
                    num_heads=1,
                    num_blocks=2,
                    psi_dims=[128,],
                    projector=copy.deepcopy(encoder)
                )
                
                for model_name, model in models.items():
                    print(model_name, count_params(model))                
                                
                for model_name, model in models.items():
                    
                    # Print statement
                    print('\n')
                    print('Replicate: {}'.format(replicate))
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
                    
                    # # Save encoder/projector for qualitative analysis
                    # if replicate == 0:
                    #     fn = os.path.join(OUTPUT_DIR, '{}_{}_encoder_weights.pt'.format(model_name, agg_func_name) )
                    #     if model_name in BINARY_METHODS:                        
                    #         torch.save(model.projector.state_dict(), fn)
                    #     else:
                    #         torch.save(model.encoder.state_dict(), fn)

        # Create pandas data frame
        results = pd.DataFrame(
        results, 
        columns = ['Agg func', 'Model', 'Learning rate', 'Parameters', 'Eval loss', 'Test loss']
        )
            
        # Save to file
        fn = os.path.join(OUTPUT_DIR, 'replicate_{}.csv'.format(replicate))
        results.to_csv(fn)
            
    
if __name__ == '__main__':
    main()
    