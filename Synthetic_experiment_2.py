import os
import pandas as pd
import torch.nn as nn

from utils.trainutils import SetTrainer, SetTester
from utils.datautils import make_vector_aggregates
from utils.aggutils import get_aggregation_function
from utils.auxutils import set_device, count_params, make_output_folder
from utils.modelutils import QUANN, MLP

INPUT_DIM = 16
MIN_K = 2
MAX_K = 1024
TRAIN_SET_SIZE = 10000
EVAL_SET_SIZE = 1000
TEST_SET_SIZE = 1000
AGG_FUNCS = [
  # 'L1_medoid', 
  'L2_medoid', 
  # 'P1_mean', 
  'P2_mean', 
  'midpoint',
  'geometric_median', 
  'marginal_median',
  'log_sum_exp', 
  'norm_max',
  # 'norm_min', 
  'row_max', 
  # 'row_min', 
  # 'row_sum',
  'variance',   
  'skewness' 
]
LEARNING_RATEs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
BATCH_SIZE = 32
NUM_REPLICATES = 5
NUM_EPOCHS = 10
LATENT_DIM = 16
OUTPUT_DIR = './results/Synthetic_experiment_2/'

def make_datasets(agg_func_name):
    agg_func = get_aggregation_function(agg_func_name)
      
    size = {
      'train' : TRAIN_SET_SIZE, 
      'eval' : EVAL_SET_SIZE, 
      'test' : EVAL_SET_SIZE
    }
    
    data = {}
    for n, s in size.items():
      data[n] = make_vector_aggregates(
        agg_func = agg_func,
        dim = INPUT_DIM,
        num = s,
        min_k = 2 if agg_func_name in ['skewness', 'variance', 'kurosis'] else 1,
        max_k = MAX_K
      )  

    return data


def main():
  
  device = set_device(0)
  
  make_output_folder(OUTPUT_DIR)
    
  for replicate in range(NUM_REPLICATES):
  
    results = []
  
    for agg_func_name in AGG_FUNCS:
      
      print('\nExperiment with {}\n'.format(agg_func_name))
          
      # Generate dataset
      data = make_datasets(agg_func_name)
      
      for lr in LEARNING_RATEs:

          models = {}

          models['QUANN_1'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=1,
              psi_dims=[32,],
              hidden_dims=[128,],
          )

          models['QUANN_2'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=2,
              psi_dims=[32,],
              hidden_dims=[128,],
          )
          
          models['QUANN_3'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=4,
              psi_dims=[32,],
              hidden_dims=[128,],
          )     
          
          models['QUANN_1a'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=1,
              psi_dims=[32,32],
              hidden_dims=[128,],
          )

          models['QUANN_2a'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=2,
              psi_dims=[32,32],
              hidden_dims=[128,],
          )
          
          models['QUANN_3a'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=4,
              psi_dims=[32,32],
              hidden_dims=[128,],
          )             
          
          models['QUANN_1c'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=1,
              psi_dims=[64],
              hidden_dims=[128,],
          )

          models['QUANN_2c'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=2,
              psi_dims=[64],
              hidden_dims=[128,],
          )
          
          models['QUANN_3c'] = QUANN(
              input_dim=INPUT_DIM,
              output_dim=INPUT_DIM, 
              latent_dim=LATENT_DIM,
              num_blocks=4,
              psi_dims=[64],
              hidden_dims=[128,],
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
                model, 
                data['train'], 
                eval_dataset=data['eval'],
                loss_fn=nn.MSELoss(),
                lr = lr,
                batch_size = BATCH_SIZE,
                device=device
              )
            
              # Train
              _, eval_losses = trainer.train(num_epochs=NUM_EPOCHS)
                    
              # Test
              tester = SetTester(
                model,
                data['test'],
                loss_fn=nn.MSELoss(),
                batch_size = BATCH_SIZE,
                device=device
              )
            
              # Test
              test_loss, _ = tester.test()    
              print('Test Loss = {:1.4f}\n'.format(test_loss))
          
              # Count model params
              num_params = count_params(model)          
            
              # Add to results
              results.append([agg_func_name, model_name, num_params, lr, test_loss] + eval_losses)
        
    # Create pandas data frame
    results = pd.DataFrame(
      results, 
      columns = ['Aggregator', 'Model', 'Parameters', 'Learning rate', 'Test loss'] + ['Eval loss {}'.format(i + 1) for i in range(NUM_EPOCHS)]
    )
    
    # Save to file
    fn = os.path.join(OUTPUT_DIR, 'replicate_{}.csv'.format(replicate))
    results.to_csv(fn)


if __name__ == '__main__':
    main()