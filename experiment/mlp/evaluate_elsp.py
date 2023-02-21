import os, yaml
import random
import time
import numpy as np
import torch as th
from tqdm import tqdm
from stable_baselines3.ppo import PPO
from elsp_env_manager.base import Variables
from elsp_env_manager.elsp_env import ELSPEnv, VARIABLE_MAP_FILE_NAME, VARIABLE_DEFINE_FILE_NAME
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.TimeCalculator import time_calculator as tc
from utils.xlsx_helper import EvalDataSaver
from utils.TimeCalculator import time_calculator as tc
from experiment.env_create import env_creator


def set_random_seed(_seed=7):
    random.seed(_seed)
    np.random.seed(_seed)
    if _seed is not None:
        th.manual_seed(_seed)
        th.cuda.manual_seed(_seed)
        th.cuda.manual_seed_all(_seed)


with open('config.yaml', "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['mlp']
seed = config['evaluate']['random_seed']
set_random_seed(seed)
cpu_num = config['evaluate']['cpu_num']
eval_times = config['evaluate']['eval_times']


def evaluate(env: SubprocVecEnv, eval_times, deterministic, model):
    obs_list = env.reset()
    done_num = 0
    bar = tqdm(total=eval_times)
    all_time = 0
    cfg_path = env.get_attr('cfg_path', [0])[0]
    profit_list = []
    storage_list = []
    csl_list = []
    idle_percent_list = []
    backlog_penalties_list = []
    inventory_dict_list = []
    csl_dict_list = []
    sales_volumes_dict_list = []
    decay_dict_list = []
    while True:
        # tc.st('predict')
        action = model.predict(obs_list, deterministic=deterministic)
        # tc.ed('predict')
        actions = []
        for index in range(len(action[0])):
            actions.append(int(action[0][index]))
        obs_list, reward, dones, inf = env.step(actions)
        for _inf in inf:
            variable = Variables.load(map_json_file=os.path.join(cfg_path, VARIABLE_MAP_FILE_NAME),
                                      define_json_file=os.path.join(cfg_path, VARIABLE_DEFINE_FILE_NAME),
                                      inf=_inf)
            delta_time = variable.get_variable('DAYS')
            all_time += delta_time
        for index in range(len(dones)):
            done = dones[index]
            if done:
                done_num += 1
                variable = Variables.load(map_json_file=os.path.join(cfg_path, VARIABLE_MAP_FILE_NAME),
                                          define_json_file=os.path.join(cfg_path, VARIABLE_DEFINE_FILE_NAME),
                                          inf=inf[index])
                profit_list.append(variable.get_variable("Profit"))
                storage_list.append(variable.get_variable("Storage"))
                csl_list.append(variable.get_variable("CSL"))
                idle_percent_list.append(variable.get_variable("IDLE_PERCENT"))
                backlog_penalties_list.append(variable.get_variable("Backlog penalties"))
                inventory_dict_list.append(variable.get_variable("INVENTORY_DICT"))
                csl_dict_list.append(variable.get_variable("CSL_DICT"))
                sales_volumes_dict_list.append(variable.get_variable("SALES_VOLUME_DICT"))
                decay_dict_list.append(variable.get_variable("DECAY_DICT"))
                bar.update(1)
                # bar.set_description("mean:%f" % (sum(profit_list) / len(profit_list)))
                if done_num >= eval_times:
                    break
        bar.set_description("mean:%f day %.2f" % (
        0 if len(profit_list) == 0 else (sum(profit_list) / len(profit_list)), all_time / eval_times / 2520))
        if done_num >= eval_times:
            break
    bar.close()
    return (profit_list, storage_list, csl_list, idle_percent_list, backlog_penalties_list,
            inventory_dict_list, csl_dict_list, sales_volumes_dict_list, decay_dict_list)


def evaluate_mlp(env_no, demand_scale, model_path, result_xlsx_path,):
    result_xlsx_path = result_xlsx_path if result_xlsx_path.startswith('/') else os.getcwd()+'/'+result_xlsx_path
    model_path = model_path if model_path.startswith('/') else os.getcwd()+'/'+model_path
    row = 1
    deterministic = False
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    env_config = config[f'env{env_no}']
    env_series = env_config['env_series']
    env_id = env_config['env_id']
    _env_creator = lambda : env_creator(env_series, env_id, 1, env_config['env_type'])
    _env_creator()
    env = SubprocVecEnv([_env_creator for i in range(cpu_num)])
    model = PPO.load(model_path, device=device)
    
    tc.st('eval')
    profit_list, storage_list, csl_list, idle_percent_list, backlog_penalties_list, inventory_dict_list, csl_dict_list, sales_volumes_dict_list, decay_dict_list = evaluate(env=env, eval_times=eval_times, deterministic=deterministic, model=model)
    tc.ed('eval')
    eval_data_saver = EvalDataSaver(file_name=result_xlsx_path)
    eval_data_saver.algorithm = 'PPO_MLP_{}_{}'.format(cpu_num, 'T' if deterministic else 'F')
    eval_data_saver.env_id = '%03d%02d' % (env_series, env_id)
    eval_data_saver.seed = seed
    buff = [
        [demand_scale, 'mean', 'std', 'max', 'min'],
        ['Profit', sum(profit_list) / len(profit_list), np.std(profit_list, ddof=1), max(profit_list), min(profit_list)],
        ['Storage', sum(storage_list) / len(storage_list), np.std(storage_list, ddof=1), max(storage_list), min(storage_list)],
        ['CSL', sum(csl_list) / len(csl_list), np.std(csl_list, ddof=1), max(csl_list), min(csl_list)],
        ['IDLE', sum(idle_percent_list) / len(idle_percent_list), np.std(idle_percent_list, ddof=1), max(idle_percent_list), min(idle_percent_list)],
        ['Backlog penalties', sum(backlog_penalties_list) / len(backlog_penalties_list),
         np.std(backlog_penalties_list, ddof=1), max(backlog_penalties_list), min(backlog_penalties_list)],
    ]
    probable_actions = env_creator().get_probable_actions()
    probable_actions.pop(0)
    __ = {}
    for csl_dict in csl_dict_list:
        for probable_action in probable_actions:
            if probable_action not in __:
                __[probable_action] = []
            __[probable_action].append(csl_dict[str(probable_action)])
    for k, v in __.items():
        buff.append(['CSL[%d]' % k, sum(v) / len(v), np.std(v, ddof=1), max(v), min(v)])
    eval_data_saver.set_eval_data(model_path, demand_scale, buff)
    h = len(buff)
    for i in range(10):
        eval_data_saver.sort(1 + (h + 1) * i, 5, h)
    eval_data_saver.xlsx_helper.save()

if __name__ == "__main__":
    pass