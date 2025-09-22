import os
import pandas as pd
import torch.nn as nn

from utils.datautils import create_modelnet40_splits, modelnet_collate_fn
from utils.modelutils import FSPoolSetfunc, SlotAttModel, MLP
from utils.trainutils import SetTrainer, SetTester
from utils.evalutils import multi_class_accuracy, MyCrossEntropyLoss
from utils.auxutils import set_device, count_params, make_output_folder

BATCH_SIZE = 32
LEARNING_RATEs = [1e-4, 5e-4, 1e-3]
WEIGHT_DECAY = 1e-5
LATENT_DIM = 128
NUM_EPOCHS = 50
NUM_REPLICATES = 4
MIN_Ks = [256, 512, 1024]
MAX_K = 2048
OUTPUT_DIR = './results/ModelNet_experiment_posthoc/'

def main():
    device = set_device(0)
    
    make_output_folder(OUTPUT_DIR)
    
    for replicate in range(NUM_REPLICATES):
        
        results = []
    
        for min_k in MIN_Ks:
            data = create_modelnet40_splits(
                train_size=20000,
                eval_size=3000,
                test_size=3000,
                min_k=min_k,
                max_k=MAX_K
            )
            
            for lr in LEARNING_RATEs:
                    
                models = {}
                                    
                models['FSPool'] = FSPoolSetfunc(
                    latent_dim=LATENT_DIM,
                    encoder = MLP(input_dim=3, output_dim=LATENT_DIM, hidden_dims=[256,256]),
                    predictor= MLP(input_dim=LATENT_DIM, output_dim=40, hidden_dims=[256,])
                )        
                                
                models['SlotAtt'] = SlotAttModel(
                    input_dim=LATENT_DIM,
                    output_dim=40,
                    latent_dim=LATENT_DIM,
                    encoder=MLP(input_dim=3, output_dim=LATENT_DIM, hidden_dims=[256,256]),
                    hidden_dims=[] 
                )
                
                for model_name, model in models.items():
                    print(model_name, count_params(model))

                for model_name, model in models.items():
                    
                    # Print statement
                    print('\n')
                    print('Replicate: {}'.format(replicate))
                    print('Model: {}'.format(model_name))
                    print('Min K: {}'.format(min_k))
                    print('Learning rate: {}'.format(lr))                    
                    
                    # Init trainer
                    trainer = SetTrainer(
                        model=model,
                        dataset=data['train'],
                        eval_dataset=data['eval'],
                        batch_size=BATCH_SIZE,
                        lr = lr,
                        weight_decay=WEIGHT_DECAY,
                        loss_fn=nn.CrossEntropyLoss(),
                        perf_fn=multi_class_accuracy,
                        collate_fn=modelnet_collate_fn,
                        device = device                
                    )

                    # Train model
                    _, eval_loss = trainer.train(num_epochs = NUM_EPOCHS)    
                    eval_loss = eval_loss[-1]
                    
                    # Init tester
                    tester = SetTester(
                        model=model, 
                        dataset=data['test'],
                        loss_fn=nn.CrossEntropyLoss(),
                        perf_fn=multi_class_accuracy,
                        collate_fn=modelnet_collate_fn,
                        batch_size=BATCH_SIZE,
                        device=device
                    )
                    
                    # Test model
                    test_loss, test_perf = tester.test()
                    
                    # Count model params
                    num_params = count_params(model)
                    
                    # Add to results
                    results.append([min_k, model_name, lr, num_params, eval_loss, test_loss, test_perf])
                            
        # Create pandas data frame
        results = pd.DataFrame(
            results, 
            columns = ['Min_K', 'Model', 'Learning rate', 'Parameters', 'Eval loss', 'Test loss', 'Test perf']
        )
            
        # Save to file
        fn = os.path.join(OUTPUT_DIR, 'replicate_{}.csv'.format(replicate))
        results.to_csv(fn)
            
    
if __name__ == '__main__':
    main()
    
