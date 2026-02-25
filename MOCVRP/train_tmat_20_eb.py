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

from MOCVRPTrainer import CVRPTrainer

##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'distribution': "TMAT" # EUC, TMAT, XASY
}

architecture = "GMS-EB" # GMS-DH or GMS-EB
# GMS-DH: Change GREAT_params and dh_params for encoder
# GMS-EB: Change GREAT_params for encoder

training_method = "Chb" # Linear or Chb
curriculum_learning = False

GREAT_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8, 
    'ff_hidden_dim': 512,
    "great_asymmetric": True, # True for TMAT/XASY, False for EUC
    "dropout": 0.1, 
}

dh_params = {
    'L1': 5, # GNN layer number, overrides GREAT_params['encoder_layer_num']
    'L2': 2 # Transformer layer number
}

decoder_params = {
    'embedding_dim': 128,
    'qkv_dim': 16, 
    'head_num': 8,
    'logit_clipping': 10,
    'eval_type': 'argmax',
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
    'epochs': 200,
    'train_episodes': 100*1000,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 5,
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './Final_result/edge_50',  # directory path of pre-trained model and log files saved.
        'epoch': 20,  # epoch version of pre-trained model to laod.
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
    batch_size = 64
    fwd_batch_size = batch_size
    problem_size += 1
    pomo_size = problem_size - 1

    return problem_size, pomo_size, batch_size, fwd_batch_size

### Config end

if architecture == "GMS-DH":
    encoder = "hybrid"
    decoder = "MP"

    encoder_params = dh_params
    encoder_params['edge_attention_params'] = GREAT_params
elif architecture == "GMS-EB":
    encoder = "GREAT-E"
    decoder = "MP-E"
    encoder_params = GREAT_params

decoder_params["training_method"] = training_method
##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)

    _print_config()

    # We treat depot as any other node
    env_params['problem_size'] += 1

    trainer = CVRPTrainer(encoder=encoder,
                    decoder=decoder,
                    training_method=training_method,
                    curriculum_learning=curriculum_learning,
                    curriculum_function=curriculum_function,
                    env_params=env_params,
                    encoder_params=encoder_params,
                    decoder_params=decoder_params,
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
    logger.info('Model: {}'.format(architecture))
    [logger.info(key + ": {}".format(env_params[key])) for key in env_params.keys()]
    [logger.info("Curriculum Learning: {}".format(curriculum_learning))]
    [logger.info("Training Method: {}".format(training_method))]
    [logger.info(key + ": {}".format(encoder_params[key])) for key in encoder_params.keys()]
    [logger.info(key + ": {}".format(decoder_params[key])) for key in decoder_params.keys()]
    [logger.info(key + ": {}".format(optimizer_params[key])) for key in optimizer_params.keys()]
    [logger.info(key + ": {}".format(trainer_params[key])) for key in trainer_params.keys()]

##########################################################################################

if __name__ == "__main__":
    main()
