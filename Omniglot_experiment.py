import os
import pandas as pd
import torch.nn as nn

from utils.datautils import create_stacked_Omniglot
from utils.modelutils import SmallImageEncoder, QUANN, DeepSet, MeanSet, PointNet, SetTransformer, QUANNTransformer, HPDS, ImageClassifier
from utils.trainutils import SetTrainer, SetTester, ImageClassifierTrainer
from utils.evalutils import MultiLabelAccuracy
from utils.auxutils import set_device, count_params, make_output_folder, set_seed

BATCH_SIZE = 32
LEARNING_RATEs = [1e-4, 5e-4, 1e-3]
WEIGHT_DECAY = 1e-5
LATENT_DIM = 128
NUM_EPOCHS = 50
NUM_REPLICATES = 4
MIN_K = 5
MAX_Ks = [10, 15, 20, 25]
OUTPUT_DIR = './results/Omniglot_experiment/'

def main():
    # Set seed
    set_seed(21)
    
    # Set device
    device = set_device(0)
    
    # Set output folder
    make_output_folder(OUTPUT_DIR)
    
    for replicate in range(NUM_REPLICATES):
        
        results = []

        # Singular data for image classification
        data_singles = create_stacked_Omniglot(
            train_size=60000,
            eval_size=10000,
            test_size=0,
            min_k=1, 
            max_k=1
        )
        
        for max_k in MAX_Ks:
            data = create_stacked_Omniglot(
                train_size=20000,
                eval_size=2000,
                test_size=3000,
                min_k=MIN_K, 
                max_k=max_k
            )
            
            for lr in LEARNING_RATEs:
            
                models = {}

                models['DeepSet'] = DeepSet(
                    encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
                    output_dim=50, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )        

                models['MeanSet'] = MeanSet(
                    encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
                    output_dim=50, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )        
                
                models['PointNet'] = PointNet(
                    encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
                    output_dim=50, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )

                models['HPDS'] = HPDS(
                    encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
                    output_dim=50, 
                    latent_dim=LATENT_DIM,
                    hidden_dims=[256,256]
                )
                                    
                models['QUANN'] = QUANN(
                    encoder=SmallImageEncoder(latent_dim=LATENT_DIM), 
                    output_dim=50, 
                    latent_dim=LATENT_DIM,
                    num_blocks=2,
                    psi_dims=[128,],
                    hidden_dims=[256,]
                )        
                
                models['SetTransformer'] = SetTransformer(
                    input_dim=LATENT_DIM,
                    output_dim=50,
                    latent_dim=LATENT_DIM,
                    num_heads=1,
                    projector=SmallImageEncoder(latent_dim=LATENT_DIM)
                )
                
                models['QUANNTransformer'] = QUANNTransformer(
                    input_dim=LATENT_DIM,
                    output_dim=50,
                    latent_dim=LATENT_DIM,
                    num_heads=1,
                    num_blocks=2,
                    psi_dims=[128,],
                    projector=SmallImageEncoder(latent_dim=LATENT_DIM)
                )
                
                for model_name, model in models.items():
                    print(model_name, count_params(model))            

                for model_name, model in models.items():
                    
                    # Print statement
                    print('\n')
                    print('Replicate: {}'.format(replicate))
                    print('Model: {}'.format(model_name))
                    print('Max K: {}'.format(max_k))
                    print('Learning rate: {}'.format(lr))
                    
                    # Init trainer
                    trainer = SetTrainer(
                        model=model,
                        dataset=data['train'],
                        eval_dataset=data['eval'],
                        batch_size=BATCH_SIZE,
                        lr = lr,
                        weight_decay=WEIGHT_DECAY,
                        loss_fn=nn.BCEWithLogitsLoss(),
                        perf_fn=MultiLabelAccuracy(balanced = True),
                        device = device
                        
                    )

                    # Train model
                    _, eval_loss = trainer.train(num_epochs = NUM_EPOCHS)    
                    eval_loss = eval_loss[-1]
                    
                    # Init tester
                    tester = SetTester(
                        model=model, 
                        dataset=data['test'],
                        loss_fn=nn.BCEWithLogitsLoss(),
                        perf_fn=MultiLabelAccuracy(balanced=True),
                        batch_size=BATCH_SIZE,
                        device=device
                    )
                    
                    # Test model
                    test_loss, test_perf = tester.test()
                    
                    # Count model params
                    num_params = count_params(model)
                                        
                    # Extract encoder/projector from the model
                    if model_name in ['SetTransformer', 'QUANNTransformer']:                        
                        encoder = model.projector
                    else:
                        encoder = model.encoder
                        
                    # Freeze all the weights
                    for param in encoder.parameters():
                        param.requires_grad = False    
                                                
                    # Initiate image classificer
                    model = ImageClassifier(
                        encoder=encoder, 
                        latent_dim=LATENT_DIM, 
                        output_dim=50
                    )
                    
                    # Initiate the trainer
                    trainer = ImageClassifierTrainer(
                        model, 
                        data_singles['train'], 
                        data_singles['eval'], 
                        lr = 1e-3, 
                        batch_size = 32,
                        device = device
                    )
                    
                    # Train image classifier
                    trainer.train(epochs=10)
                    
                    # Get results from eval set
                    img_loss, img_accuracy = trainer.evaluate()
                    
                    # Add to results
                    results.append([max_k, model_name, lr, num_params, eval_loss, test_loss, test_perf, img_loss, img_accuracy])

                            
        # Create pandas data frame
        results = pd.DataFrame(
            results, 
            columns = ['Max_k', 'Model', 'Learning rate', 'Parameters', 'Eval loss', 'Test loss', 'Test perf', 'Img Classif Loss', 'Img Classif Perf']
        )
            
        # Save to file
        fn = os.path.join(OUTPUT_DIR, 'replicate_{}.csv'.format(replicate))
        results.to_csv(fn)
            
    
if __name__ == '__main__':
    main()
    
