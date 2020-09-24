from drq import DRQAgent
from video import VideoRecorder
from replay_buffer import ReplayBuffer
from logger import Logger
import utils
import os
import time
from datetime import datetime

import dmc2gym
import torch
torch.backends.cudnn.benchmark = True  # GPU alg optimization


def make_env(env,
             seed,
             image_size,
             action_repeat,
             frame_stack):
    """Helper function to create dm_control environment"""
    if env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = env.split('_')[0]
        task_name = '_'.join(env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=image_size,
                       width=image_size,
                       frame_skip=action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=frame_stack)

    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Experiment(object):
    def __init__(self,
                 log_save_tb=True,
                 log_frequency_step=10000,
                 agent_name='drq',
                 device='cuda',
                 env='cartpole_swingup',
                 seed=1,
                 image_size=84,
                 action_repeat=8,
                 frame_stack=3,
                 replay_buffer_capacity=100000,
                 image_pad=4,
                 save_video=True
                 ):
        work_dir = os.getcwd()
        datetime_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        log_dir = f'logs/{agent_name}_{env}_{seed}_{datetime_str}'

        self.logger = Logger(log_dir,
                             save_tb=log_save_tb,
                             log_frequency=log_frequency_step,
                             agent=agent_name,
                             action_repeat=action_repeat
                             )

        utils.set_seed_everywhere(seed)
        self.env = make_env(env, seed, image_size, action_repeat, frame_stack)

        self.agent = DRQAgent(obs_shape=self.env.observation_space.shape,
                              action_shape=self.env.action_space.shape,
                              action_range=(
                                  float(self.env.action_space.low.min()),
                                  float(self.env.action_space.high.max())
                              ),
                              device=device
                              )

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          replay_buffer_capacity,
                                          image_pad,
                                          device
                                          )

        self.video_recorder = VideoRecorder(
            work_dir if save_video else None)
        self.step = 0

    def evaluate(self,
                 num_eval_episodes=10,
                 ):
        average_episode_reward = 0
        for episode in range(num_eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self,
            num_train_steps=1000000,
            num_train_iters=1,
            num_seed_steps=1000,
            eval_frequency=5000
            ):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > num_seed_steps))

                # evaluate agent periodically
                if self.step % eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < num_seed_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= num_seed_steps:
                for _ in range(num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


if __name__ == '__main__':
    Experiment().run()
