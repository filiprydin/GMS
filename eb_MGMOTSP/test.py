##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src


from eb_MGMOTSP.MGMOTSPTester import TSPTester
from MGMOTSPProblemDef import get_random_problems

##########################################################################################
import time
import hvwfg

##########################################################################################
# parameters
env_params = {
    'distribution': "FLEX", # FIX or FLEX
    'problem_size': 20,
    'pomo_size': 20,
    'emax': 2
}

training_method = "Chb" # Linear or Chb

encoder_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8, 
    'ff_hidden_dim': 512,
    "dropout": 0.1, 
}

decoder_params = {
    'embedding_dim': 128,
    'qkv_dim': 16, 
    'head_num': 8,
    'logit_clipping': 10,
    'eval_type': 'argmax',
}
decoder_params["training_method"] = training_method

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load_path': './result/FLEX2_20.pt',  # path of pre-trained model
    'data_load_path': './data/FLEX2_20',  # path of test data. If None, random problems will be generated.
    'reference': [15, 15],
    'test_episodes': 200, 
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 50
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp',
        'filename': 'run_log'
    }
}

### Config end

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(key + ": {}".format(env_params[key])) for key in env_params.keys()]
    logger.info('Training Method: {}'.format(training_method))
    [logger.info(key + ": {}".format(encoder_params[key])) for key in encoder_params.keys()]
    [logger.info(key + ": {}".format(decoder_params[key])) for key in decoder_params.keys()]
    [logger.info(key + ": {}".format(tester_params[key])) for key in tester_params.keys()]

def load_problems(path):
    edge_dists = torch.load(os.path.join(path, "dists"), weights_only=True)
    edge_to_node = torch.load(os.path.join(path, "edge_to_node"), weights_only=True)
    
    return edge_dists, edge_to_node

##########################################################################################
def main(n_sols = 101):
    
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()

    tester = TSPTester(
                    training_method=training_method,
                    env_params=env_params,
                    encoder_params=encoder_params,
                    decoder_params=decoder_params,
                    tester_params=tester_params)
        
    if tester_params['data_load_path'] is not None:
        shared_problem, edge_to_node = load_problems(tester_params['data_load_path'])
        shared_problem = shared_problem.to(tester.device)
        edge_to_node = edge_to_node.to(tester.device)
    else:
        shared_problem, edge_to_node = get_random_problems(env_params['distribution'], tester_params['test_episodes'], env_params['problem_size'], env_params['emax'])

    prefs = torch.zeros((n_sols, 2))
    for i in range(n_sols):
        prefs[i, 0] = 1 - 0.01 * i
        prefs[i, 1] = 0.01 * i

    timer_start = time.time()

    sols = tester.run(shared_problem, prefs, edge_to_node)
    
    timer_end = time.time()
    
    total_time = timer_end - timer_start
    
    ref = np.asarray(tester_params['reference'])
    hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
    hv_ratio =  hv / (ref[0] * ref[1])
    
    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hv_ratio))

##########################################################################################
if __name__ == "__main__":
    main()
