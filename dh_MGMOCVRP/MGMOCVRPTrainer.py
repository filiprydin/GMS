import torch
from logging import getLogger

from dh_MGMOCVRP.MGMOCVRPEnv import CVRPEnv as Env
from dh_MGMOCVRP.MGMOCVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

class CVRPTrainer:
    def __init__(self,
                 training_method,
                 curriculum_learning,
                 curriculum_function,
                 env_params,
                 encoder_params,
                 head1_params,
                 head2_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.training_method = training_method
        self.head1_params = head1_params

        self.curriculum_learning = curriculum_learning
        self.curriculum_function = curriculum_function

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.env = Env(**self.env_params)
        self.model = Model(encoder_params, head1_params, head2_params, self.env)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.e0 = trainer_params['e_0']
        self.samples_h1 = trainer_params['samples_per_instance_h1']
        self.pomo_size_h1 = trainer_params['pomo_size_h1']

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            if epoch <= self.e0:
                self.logger.info('Training Initial Model')
            else:
                self.logger.info('Training Head 1 and Head 2')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score_obj1, train_score_obj2, train_loss_h1, train_loss_h2 = self._train_one_epoch_together(epoch)
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_loss', epoch, train_loss_h1)
            self.result_log.append('train_loss', epoch, train_loss_h2)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
       
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_motsp-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
    
    def _train_one_epoch_together(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
    
        loss_AM_h1 = AverageMeter()
        loss_AM_h2 = AverageMeter()

        # Overwrite some things if using curriculum learning
        if self.curriculum_learning:
            problem_size, pomo_size, train_batch_size, max_batch_size, emax = self.curriculum_function(epoch)

            self.accumulation_steps = train_batch_size // max_batch_size
            self.env.problem_size = problem_size
            self.env.pomo_size = pomo_size
            self.env.emax_max = emax
        else:
            max_batch_size = self.trainer_params['train_batch_size']
            self.accumulation_steps = 1

        self.model.zero_grad()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        self.accumulation_step = 0
        self.final_batch = False
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(max_batch_size, remaining)

            if batch_size == remaining:
                self.final_batch = True
            
            # Train in different way depending on which epoch
            if epoch <= self.e0:
                avg_score_obj1, avg_score_obj2, avg_loss_h2 = self._train_one_batch_init(batch_size)
                avg_loss_h1 = 0
            else:
                avg_score_obj1, avg_score_obj2, avg_loss_h1, avg_loss_h2 = self._train_one_batch(batch_size)
                self.accumulation_step += 1

            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM_h1.update(avg_loss_h1, batch_size)
            loss_AM_h2.update(avg_loss_h2, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss h1: {:.4f},  Loss h2: {:.4f}'
                                    .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                            score_AM_obj1.avg, score_AM_obj2.avg, loss_AM_h1.avg, loss_AM_h2.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss h1: {:.4f},  Loss h2: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, loss_AM_h1.avg, loss_AM_h2.avg))
        
        # Change back environment
        if self.curriculum_learning:
            self.env.problem_size = self.env_params['problem_size']
            self.env.pomo_size = self.env_params['pomo_size']
            self.env.emax_max = self.env_params['emax']

        self.prev_problem_size = self.env.problem_size

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM_h1.avg, loss_AM_h2.avg
    
    def _train_one_batch_init(self, batch_size):
        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size, simple_graph = True, simple_graph_dist = self.trainer_params["initial_dist"])

        lambda1 = torch.rand([1])
        pref = torch.tensor([lambda1, 1 - lambda1])

        self.model.decoder_2.assign(pref)

        self.model.encode()
        # Simple graph -> do not prune, use all edges
        with torch.no_grad():
            selected_edges = torch.arange(self.env.problem_size * (self.env.problem_size - 1))
            selected_edges = selected_edges.unsqueeze(0).expand(batch_size, -1)
            reset_state, _, _ = self.env.reset(selected_edges)

        self.model.pre_forward_head_2(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward

        if self.training_method == "Chb":
            tch_reward = pref * reward    
            tch_reward , _ = tch_reward.max(dim = 2)
        elif self.training_method == "Linear":
            tch_reward = (pref * reward).sum(dim=2)

        # set back reward to negative
        reward = -reward
        tch_reward = -tch_reward

        log_prob = prob_list.log().sum(dim=2)
        # shape = (batch, group)

        tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)

        tch_loss = -tch_advantage * log_prob # Minus Sign
        # shape = (batch, group)
        loss_mean = tch_loss.mean()

        # Score
        ###############################################
        _ , max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = reward[:,:,0].gather(1, max_idx)
        max_reward_obj2 = reward[:,:,1].gather(1, max_idx)

        score_mean_obj1 = - max_reward_obj1.float().mean()
        score_mean_obj2 = - max_reward_obj2.float().mean()

        #Step & Return
        ################################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()

    def _train_one_batch(self, batch_size):
        # Trains both h1 and h2

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)

        u = torch.rand([1])
        if u < 0.25:
            lambda1 = torch.tensor([0.0])
            pref = torch.tensor([lambda1, 1 - lambda1])
        elif u < 0.50:
            lambda1 = torch.tensor([1.0])
            pref = torch.tensor([lambda1, 1 - lambda1])
        else:
            lambda1 = torch.rand([1])
            pref = torch.tensor([lambda1, 1 - lambda1])

        self.model.decoder_1.assign(pref)
        self.model.decoder_2.assign(pref)

        self.model.encode()
        self.model.pre_forward_head_1()

        selected_edges, probs = self.model.prune(samples_per_instance=self.samples_h1)

        # Change environment
        if not self.curriculum_learning:
            self.env.pomo_size = self.pomo_size_h1
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
        self.env.edge_attr = self.env.edge_attr.repeat(self.samples_h1, 1, 1)
        self.env.edge_to_node = self.env.edge_to_node.repeat(self.samples_h1, 1, 1)
        self.env.demands = self.env.demands.repeat(self.samples_h1, 1)

        with torch.no_grad():
            reset_state, _, _ = self.env.reset(selected_edges)

        self.model.pre_forward_head_2(reset_state)

        prob_list = torch.zeros(size=(batch_size * self.samples_h1, self.env.pomo_size, 0))

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward

        if self.training_method == "Chb":
            tch_reward = pref * reward    
            tch_reward , _ = tch_reward.max(dim = 2)
        elif self.training_method == "Linear":
            tch_reward = (pref * reward).sum(dim=2)

        # set back reward to negative
        reward = -reward
        tch_reward = -tch_reward # (B, P)

        ### Train head 1 ###
        max_tch_reward, max_idx = tch_reward.max(dim=1)

        log_prob_h1 = probs.log().sum(dim=1)
        # shape = (batch)

        # Calculate baseline -> average over samples
        mean_rewards = max_tch_reward.view(self.samples_h1, batch_size).mean(dim=0)
        mean_rewards_exp = mean_rewards.repeat(self.samples_h1)        
        tch_advantage_h1 = max_tch_reward - mean_rewards_exp

        tch_loss_h1 = -tch_advantage_h1 * log_prob_h1 # Minus Sign
        # shape = (batch)
        loss_mean_h1 = tch_loss_h1.mean() / self.accumulation_steps

        if not self.head1_params['only_by_distance']:
            loss_mean_h1.backward()

        ### Train head 2 ###

        log_prob_h2 = prob_list.log().sum(dim=2)
        # shape = (batch, group)

        tch_advantage_h2 = tch_reward - tch_reward.mean(dim=1, keepdim=True)

        tch_loss_h2 = -tch_advantage_h2 * log_prob_h2 # Minus Sign
        # shape = (batch, group)
        loss_mean_h2 = tch_loss_h2.mean() / self.accumulation_steps
        loss_mean_h2.backward()

        #Step & Return
        ################################################

        if (self.accumulation_step + 1) % self.accumulation_steps == 0 or self.final_batch:
            self.optimizer.step()
            self.model.zero_grad()

        # Score
        ###############################################
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = reward[:,:,0].gather(1, max_idx)
        max_reward_obj2 = reward[:,:,1].gather(1, max_idx)

        score_mean_obj1 = - max_reward_obj1.float().mean()
        score_mean_obj2 = - max_reward_obj2.float().mean()

        # Set back pomo size
        if not self.curriculum_learning:
            self.env.pomo_size = self.env_params['pomo_size']

        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean_h1.item() * self.accumulation_steps, loss_mean_h2.item() * self.accumulation_steps