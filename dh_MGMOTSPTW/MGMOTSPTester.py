import torch

from logging import getLogger

from MGMOTSPEnv import TSPEnv as Env
from MGMOTSPModel import TSPModel as Model
from MGMOTSPProblemDef import augment_data

from einops import rearrange

from utils.utils import *

class TSPTester:
    def __init__(self,
                 training_method,
                 env_params,
                 encoder_params,
                 head1_params,
                 head2_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.tester_params = tester_params
        self.training_method = training_method

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(encoder_params, head1_params, head2_params, self.env)
        
        checkpoint_fullname = tester_params['model_load_path']
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, edge_attr, edge_to_node, service_time, tw_start, tw_end, prefs, print_results=True):
        self.time_estimator.reset()
    
        self.prefs = prefs
        sols_avg = torch.zeros_like(prefs)
            
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            sols = self._test_one_batch(edge_attr, edge_to_node, service_time, tw_start, tw_end, batch_size, episode)
            sols_avg = sols_avg + sols * batch_size / test_num_episode

            episode += batch_size

            ############################
            # Logs
            ############################
            all_done = (episode == test_num_episode)
            if all_done and print_results:
                for i in range(prefs.shape[0]):
                    self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f} ".format(sols_avg[i, 0], sols_avg[i, 1]))

        return sols_avg.cpu().numpy()
                
    def _test_one_batch(self, edge_attr, edge_to_node, service_time, tw_start, tw_end, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
            
        self.env.batch_size = batch_size 
        self.env.edge_attr = edge_attr[episode: episode + batch_size, :, :]
        self.env.edge_to_node = edge_to_node[episode: episode + batch_size, :, :]
        self.env.service_time = service_time[episode: episode + batch_size, :]
        self.env.tw_start = tw_start[episode: episode + batch_size, :]
        self.env.tw_end = tw_end[episode: episode + batch_size, :]
        self.env.emax = self.env_params['emax']
        
        if aug_factor > 1:
            self.env.batch_size = self.env.batch_size * aug_factor
            self.env.edge_attr, self.env.edge_to_node, self.env.service_time, self.env.tw_start, self.env.tw_end, aug_factors = augment_data(self.env.edge_attr, self.env.edge_to_node, 
                                                                                  self.env.service_time, self.env.tw_start, self.env.tw_end, aug_factor)
            
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
        
        sols = torch.zeros((self.prefs.shape[0], 2), dtype=torch.float)

        for j, pref in enumerate(self.prefs):
            if j == 0:
                encode = True # Only encode once
            else:
                encode = False

            self.model.eval()
            with torch.no_grad():
                self.model.decoder_1.assign(pref)
                self.model.decoder_2.assign(pref)

                if encode: 
                    self.model.encode()

                self.model.pre_forward_head_1(encode=encode)

                selected_edges, _ = self.model.prune()

                reset_state, _, _ = self.env.reset(selected_edges)
                
                self.model.pre_forward_head_2(reset_state)
                
            state, reward, done = self.env.pre_step()
            
            while not done:
                with torch.no_grad():
                    selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
            
            # reward was negative, here we set it to positive to calculate TCH
            reward = - reward

            # Renormalize distances
            if aug_factor > 1:
                for i in range(aug_factor):
                    factor  = aug_factors[i]
                    reward[i * batch_size: (i + 1) * batch_size, :, 1] = reward[i * batch_size: (i + 1) * batch_size, :, 1] / factor        

            if self.training_method == "Chb":
                tch_reward = pref * reward    
                tch_reward , _ = tch_reward.max(dim = 2)
            elif self.training_method == "Linear":
                tch_reward = (pref * reward).sum(dim=2)

            # set back reward to negative
            reward = -reward
            tch_reward = -tch_reward
        
            tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            
            tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)') 
            _ , max_idx_aug = tch_reward_aug.max(dim=1)
            max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0],1)
            max_reward_obj1 = rearrange(reward[:,:,0].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
            max_reward_obj2 = rearrange(reward[:,:,1].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
        
            sols[j, 0] = -max_reward_obj1.float().mean()
            sols[j, 1] = -max_reward_obj2.float().mean()
        
        return sols

     
