import random


import gym
import numpy as np
import os
from elsp_env_manager.base import Product, MID, Variables
from elsp_env_manager.process_units.process_line import ProcessLine
from elsp_env_manager.process_units.process_units import ProductionUnit, SchedulingUnit, StorageUnit, SaleUnit, \
    CustomerUnit
from elsp_env_manager.base.constants import *
from utils.TimeCalculator import time_calculator
try:
    import torch as th
    has_torch = True
except ImportError:
    has_torch = False

# seed = 7
# random.seed(seed)
# np.random.seed(seed)


class ELSPEnv(gym.Env):

    def __init__(self, env_path, cfg_path, demand_scale=1):
        self.env_path = env_path
        self.cfg_path = cfg_path
        self.time_unit = 'day'
        self.time = 0
        self.demand_scale = demand_scale
        Product.load(os.path.join(env_path, PRODUCTS_FILE_NAME))
        ProductionUnit.load(os.path.join(env_path, PRODUCTION_UNITS_FILE_NAME), self)
        StorageUnit.load(os.path.join(env_path, STORAGE_UNITS_FILE_NAME), self)
        CustomerUnit.load(os.path.join(env_path, CUSTOMER_UNITS_FILE_NAME), env=self, demand_scale=demand_scale)
        SaleUnit.load(os.path.join(env_path, SALE_UNITS_FILE_NAME), self)
        SchedulingUnit.load(os.path.join(env_path, SCHEDULING_UNITS_FILE_NAME), self)
        ProcessLine.load(os.path.join(env_path, PROCESS_LINES_FILE_NAME), self)
        assert len(SchedulingUnit.scheduling_units) == 1
        self.scheduling_unit: SchedulingUnit = SchedulingUnit.scheduling_units[MID(SCHEDULING_UNITS_ID_PREFIX + "_0")]
        self.variables = Variables.load(map_json_file=os.path.join(cfg_path, VARIABLE_MAP_FILE_NAME),
                                        define_json_file=os.path.join(cfg_path, VARIABLE_DEFINE_FILE_NAME),
                                        inf=self.scheduling_unit.inf)
        num = self.scheduling_unit.scheduling.obs
        self.observation_space = gym.spaces.Box(np.zeros_like(self.scheduling_unit.scheduling.obs),
                                                np.ones_like(self.scheduling_unit.scheduling.obs))
        self.action_space = gym.spaces.Discrete(len(self.get_available_actions()))
        setattr(self.action_space, 'flexible', True)
        self.recorded_inf = {}
        self.count_recorded_inf = {}
        self.count_started = {}

    def record(self, k, v):
        if k not in self.recorded_inf:
            self.recorded_inf[k] = []
        self.recorded_inf[k].append(v)

    def count_record(self, k):
        if k not in self.count_started or not self.count_started[k] or len(self.count_recorded_inf[k]) == 0:
            self.count_start(k)
            self.count_started[k] = True
        self.count_recorded_inf[k][-1] += 1

    def count_start(self, k):
        if k not in self.count_recorded_inf:
            self.count_recorded_inf[k] = []
        self.count_recorded_inf[k].append(0)

    def get_recorded_inf(self):
        buff = self.recorded_inf
        self.recorded_inf = {}
        ep_num = 0
        for k, _ in buff.items():
            ep_num = max(ep_num, len(_))
        for k, v in self.count_recorded_inf.items():
            buff[k] = v[0:ep_num]
            self.count_recorded_inf[k] = v[ep_num:]
        return buff

    def step(self, action):
        available_actions = self.get_available_actions()
        # print(action, "available_actions_len", len(available_actions))
        action = available_actions[action]
        inf = self.scheduling_unit.inf
        self.count_record(str(action))
        while True:
            self.scheduling_unit.process(action)
            if self.scheduling_unit.needs_scheduling() or self.scheduling_unit.done:
                if self.scheduling_unit.done:
                    self.scheduling_unit.inf = {}
                    for k, v in self.count_started.items():
                        self.count_started[k] = False
                    self.record("Profit", self.variables.get_variable("Profit"))
                    inf["Profit"] = self.variables.get_variable("Profit")
                break
        return self.scheduling_unit.obs, self.scheduling_unit.reward, self.scheduling_unit.done, inf

    def reset(self):
        self.time = 0
        env_path = self.env_path
        cfg_path = self.cfg_path
        Product.load(os.path.join(env_path, PRODUCTS_FILE_NAME))
        ProductionUnit.load(os.path.join(env_path, PRODUCTION_UNITS_FILE_NAME), self)
        StorageUnit.load(os.path.join(env_path, STORAGE_UNITS_FILE_NAME), self)
        CustomerUnit.load(os.path.join(env_path, CUSTOMER_UNITS_FILE_NAME), self.demand_scale, self)
        SaleUnit.load(os.path.join(env_path, SALE_UNITS_FILE_NAME), self)
        SchedulingUnit.load(os.path.join(env_path, SCHEDULING_UNITS_FILE_NAME), self)
        ProcessLine.load(os.path.join(env_path, PROCESS_LINES_FILE_NAME), self)
        assert len(SchedulingUnit.scheduling_units) == 1
        self.scheduling_unit: SchedulingUnit = SchedulingUnit.scheduling_units[MID(SCHEDULING_UNITS_ID_PREFIX + "_0")]
        self.variables = Variables.load(map_json_file=os.path.join(cfg_path, VARIABLE_MAP_FILE_NAME),
                                        define_json_file=os.path.join(cfg_path, VARIABLE_DEFINE_FILE_NAME),
                                        inf=self.scheduling_unit.inf)
        self.scheduling_unit.reset()
        return self.scheduling_unit.obs

    def get_available_actions(self, scheduling_unit=None):
        if scheduling_unit is None:
            scheduling_unit = self.scheduling_unit
        demand_dict = scheduling_unit.customer_unit.demand.demand_dict
        demand_dist = scheduling_unit.customer_unit.demand.demand_dist
        available_actions = [0]
        for p, demand in demand_dict.items():
            if demand['not_fit'] > 0 or not demand_dist[p].is_always_zero(self.time):
                available_actions.append(int(p))
        return available_actions

    def get_probable_actions(self, scheduling_unit=None):
        if scheduling_unit is None:
            scheduling_unit = self.scheduling_unit
        demand_dict = scheduling_unit.customer_unit.demand.demand_dict
        available_actions = [0]
        for p, demand in demand_dict.items():
            available_actions.append(int(p))
        return available_actions

    def random_action(self):
        available_actions = self.get_available_actions()
        return int(random.random() * len(available_actions)) % len(available_actions)

    def render(self, mode='human'):
        pass

    def get_demand_u(self):
        if len(CustomerUnit.customer_units) == 1:
            for k, v in CustomerUnit.customer_units.items():
                v: CustomerUnit
                return v.demand.get_demand_u()
        else:
            raise NotImplementedError

    def decode_features(self, features):
        # features = np.array(features)
        feature_dim = features.shape[-1]
        env_feature_dim = self.scheduling_unit.scheduling.env_feature_dim
        prod_feature_dim = self.scheduling_unit.scheduling.prod_feature_dim
        prod_num = (feature_dim - env_feature_dim) // prod_feature_dim
        if has_torch and not isinstance(features, np.ndarray):
            _obs = th.split(features, [feature_dim - env_feature_dim, env_feature_dim], dim=1)
        else:
            _obs = np.split(features, [feature_dim - env_feature_dim], axis=1)
        s1 = _obs[0]
        s1 = s1.reshape((prod_num, env_feature_dim) if len(s1.shape) == 1 else (s1.shape[0], prod_num, prod_feature_dim))
        s2 = _obs[1]
        return s1, s2


if __name__ == "__main__":
    time_calculator.st("main")
    env = ELSPEnv(env_path="/home/mi-nan/workspace/myPyCharm/elsp_drl/experiment/envs/elsp_env_01114",
                  cfg_path="/home/mi-nan/workspace/myPyCharm/elsp_drl/experiment/envs/", demand_scale=1)


    def log_inf(_inf):
        for k, v in _inf.items():
            if not isinstance(v, dict):
                print(k, v)
            else:
                for sk, sv in v.items():
                    print(k, sk, sv)


    for i in range(1):
        time_calculator.st("ep")
        obs = env.reset()
        #print(obs)
        action = 1
        while True:
            obs, reward, done, inf = env.step(env.random_action())
            print(len(env.get_available_actions()), len(obs))
            # 36.04320240231678
            obs = np.array([obs, obs.copy()])
            s1, s2 = env.decode_features(obs)
            #print(s1)
            action += 1
            # log_inf(inf)
            if done:
                break
        time_calculator.ed("ep")
        """print("Profit", env.variables.get_variable("Profit"))
        print("Revenue", env.variables.get_variable("Revenue"))
        print("Total costs", env.variables.get_variable("Total costs"))
        print("\tSeed", env.variables.get_variable("Seed"))
        print("\tUSP", env.variables.get_variable("USP"))
        print("\t\tReplacement ATF filters", env.variables.get_variable("Replacement ATF filters"))
        print("\t\tCell culture setup", env.variables.get_variable("Cell culture setup"))
        print("\tDSP", env.variables.get_variable("DSP"))
        print("\tChangeover", env.variables.get_variable("Changeover"))
        print("\tStorage", env.variables.get_variable("Storage"))
        print("\tBacklog penalties", env.variables.get_variable("Backlog penalties"))
        print("\tWastage", env.variables.get_variable("Wastage"))
        print("CSL", env.variables.get_variable("CSL"))"""
    """
        print(
            inf["sale_SaleUnit_0"]["sales_volume"] - inf["setup_ProductionUnit_0"]["setup cost"]
            - inf["sale_SaleUnit_0"]["backlog_penalty"] - inf["store_StorageUnit_1"]["inventory cost"] \
            - inf["launch_ProductionUnit_0"]["launch cost"] \
            - inf["conversion_ProductionUnit_0"]["total cost"] \
            - inf["conversion_ProductionUnit_1"]["total cost"]
        )"""
    print(env.get_recorded_inf())
    time_calculator.ed("main")
