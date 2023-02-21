import os, yaml
import random
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiment.standard_self_attention.elsp_callback import CustomCallback
from experiment.env_create import env_creator
from experiment.standard_self_attention.module.set_transformer import CustomActorCriticPolicy, \
    ExtractorForSSA


with open('config.yaml', "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['ssa']
seed = config['train']['random_seed']
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
cpu_num = config['train']['cpu_num']
total_steps = eval(config['train']['total_steps'])
n_steps = config['train']['n_steps']
batch_size = cpu_num * n_steps
n_epochs = config['train']['n_epochs']
lr = config['train']['lr']
gamma = config['train']['gamma']
env_series = 100
device = 'cuda:0' if th.cuda.is_available() else 'cpu'


def train(env, path=None):
    model = PPO(CustomActorCriticPolicy, policy_kwargs={"extractor": ExtractorForSSA},
                device=device, env=env, verbose=0, learning_rate=lr,
                n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma)
    setattr(model, 'total_steps', total_steps)
    setattr(model, 'env_root_path', env.get_attr("env_path")[0])
    setattr(model, 'cfg_root_path', env.get_attr("cfg_path")[0])
    setattr(model, 'env_path', os.path.basename(env.get_attr("env_path")[0]))
    setattr(model, 'cfg_path', os.path.basename(env.get_attr("cfg_path")[0]))
    setattr(model, 'env_series', env_series)
    callback = CustomCallback(env, model, no_eval=True, n_eval_steps=10)
    import atexit
    def at_exit():
        callback.on_training_end()
        pass

    atexit.register(at_exit)

    model.learn(total_timesteps=total_steps, callback=callback)


def train_ssa(env_no: int):
    from hook.hook import Hook
    Hook.hook_all()
    env_config = config[f'env{env_no}']
    _env_creator = lambda : env_creator(env_config['env_series'], env_config['env_id'], 1, env_config['env_type'])
    _env_creator()
    env = SubprocVecEnv([_env_creator for i in range(cpu_num)])
    train(env)


if __name__ == '__main__':
    train_ssa()
    