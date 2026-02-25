##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src

from dh_MGMOCVRP.MGMOCVRPTrainer import CVRPTrainer

##########################################################################################
# parameters
env_params = {
    'distribution': "FLEX", # FIX or FLEX
    'problem_size': 20,
    'pomo_size': 20,
    'emax': 2
}

training_method = "Chb" # Either Linear or Chb
curriculum_learning = False

encoder_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 4,
    'qkv_dim': 16,
    'head_num': 8, 
    'ff_hidden_dim': 512,
    "dropout": 0.1, 
}

head2_params = {
    'attention_layers': 2, # Standard node-based attention layers
    'ff_hidden_dim': 512,
    'embedding_dim': 128,
    'qkv_dim': 16, 
    'head_num': 8,
    'logit_clipping': 20,
    'eval_type': 'argmax',
}

head1_params = {
    'only_by_distance': False, # Only use scalarized distance -> not trainable
    'attention_layers': 2, # GREAT layers before pruning decoder
    'eval_type': 'argmax',
    'logit_clipping': 20 
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4, 
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [180,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'initial_dist': 'XASY', # Distribution for first e_0 epochs, only XASY implemented
    'e_0': 5, # Initial epochs to train head 2 on simple graph 
    'epochs': 200,
    'train_episodes': 100*1000,
    'train_batch_size': 64,
    'samples_per_instance_h1': 4, # Number of edge selections per instance in h1
    'pomo_size_h1': 20, # Possibly reduce pomo size for h1 training to save memory
    'logging': {
        'model_save_interval': 5,
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/20250304_110849_train_test_tsp',  # directory path of pre-trained model and log files saved.
        'epoch': 70,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train__cvrp',
        'filename': 'run_log'
    }
}

def curriculum_function(epoch):

    problem_size = 20
    pomo_size = problem_size
    problem_size = problem_size + 1
    fwd_batch_size = 64 # NOTE: Gradient accumulates over batch_size, but we can run with fwd_batch_size to save memory for large problems
    batch_size = 64
    emax = 2

    return problem_size, pomo_size, batch_size, fwd_batch_size, emax

### Config end

head2_params["training_method"] = training_method
head1_params["training_method"] = training_method

##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)

    _print_config()

    # We treat depot as any other node
    env_params['problem_size'] += 1

    trainer = CVRPTrainer(
                    training_method=training_method,
                    curriculum_learning=curriculum_learning,
                    curriculum_function=curriculum_function,
                    env_params=env_params,
                    encoder_params=encoder_params,
                    head1_params=head1_params,
                    head2_params=head2_params,
                    optimizer_params=optimizer_params,
                    trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 1
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 10

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(key + ": {}".format(env_params[key])) for key in env_params.keys()]
    [logger.info("Curriculum Learning: {}".format(curriculum_learning))]
    [logger.info("Training Method: {}".format(training_method))]
    [logger.info(key + ": {}".format(encoder_params[key])) for key in encoder_params.keys()]
    [logger.info(key + ": {}".format(head1_params[key])) for key in head1_params.keys()]
    [logger.info(key + ": {}".format(head2_params[key])) for key in head2_params.keys()]
    [logger.info(key + ": {}".format(optimizer_params[key])) for key in optimizer_params.keys()]
    [logger.info(key + ": {}".format(trainer_params[key])) for key in trainer_params.keys()]

##########################################################################################

if __name__ == "__main__":
    main()
