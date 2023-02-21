import os

from elsp_env_manager.elsp_env import ELSPEnv


env_cache = {
    'cached': None
}


def get_last_env():
    return env_cache['cached']


def env_creator(env_series=None, env_id=None, demand_scale=None, env_type=''):
    if env_series is not None and env_id is not None:
        env_path = os.getcwd()+"/experiment/envs%s/elsp_env%s_%03d%02d" % (env_type, env_type, env_series, env_id)
    else:
        env_path = get_last_env().env_path
    if demand_scale is None:
        demand_scale = get_last_env().demand_scale
    env = ELSPEnv(env_path, os.getcwd()+"/experiment/envs%s/" % env_type,
                   demand_scale=demand_scale)
    env_cache['cached'] = env
    return env
