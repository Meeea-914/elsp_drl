import math
import numpy as np
import copy
from abc import ABC, abstractmethod

from elsp_env_manager.base import Distribution, MID, Substance, Material, NormalDistribution
from elsp_env_manager.base.model_manager import load_model
from elsp_env_manager.base.constants import *


class Process(ABC):
    to_refresh_keys = ['time']

    @abstractmethod
    def __init__(self, process_unit):
        self.process_unit = process_unit
        self.obs_inf_k = self.__class__.__name__.lower() + "_" + str(self.process_unit.m_id)

    def get_attr(self, name, prod: Substance):  # which_may_be_up_to_product
        if getattr(self, name) == "up_to_product":
            return prod.model_json[name]
        else:
            return getattr(self, name)

    def get_distribution(self, name, prod: Substance):
        if isinstance(prod, MID):
            prod = Substance.substance[prod]
        if getattr(self, name) == "up_to_product":
            return Distribution.load(prod.model_json, name)
        else:
            return getattr(self, name)

    def set_obs_inf(self, k, v, replace=True):
        if not hasattr(self.process_unit.env, 'scheduling_unit'):
            return
        if k not in self.process_unit.env.scheduling_unit.inf:
            replace = True
        if replace:
            self.process_unit.env.scheduling_unit.inf[k] = v
        else:
            for v_k, v_v in v.items():
                self.process_unit.env.scheduling_unit.inf[k][v_k] = v_v

    def refresh_obs_inf_keys(self):
        for k in self.to_refresh_keys:
            for obs_inf_k, v in self.process_unit.env.scheduling_unit.inf.items():
                if k in self.process_unit.env.scheduling_unit.inf[obs_inf_k]:
                    self.process_unit.env.scheduling_unit.inf[obs_inf_k].pop(k)

    def get_obs_inf(self, k, sk, default):
        if k not in self.process_unit.env.scheduling_unit.inf:
            self.process_unit.env.scheduling_unit.inf[k] = default
        if sk not in self.process_unit.env.scheduling_unit.inf[k]:
            self.process_unit.env.scheduling_unit.inf[k][sk] = default[sk]
        return self.process_unit.env.scheduling_unit.inf[k]

    def refresh_increment_inf(self, k, v):
        org = self.get_obs_inf(self.obs_inf_k, k, {
            k: 0
        })
        org[k] += v
        self.set_obs_inf(self.obs_inf_k, org)
        pass

    def refresh_inf(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class ProductionProcess(Process):

    def reset(self):
        super().reset()
        self.finished = False
        self.finish_time = 0
        self.cost = 0
        self.last_prod = None
        pass

    def __init__(self, process_unit):
        super().__init__(process_unit)
        self.finished = False
        self.finish_time = -1
        self.start_time = -1
        self.cost = 0
        self.last_prod = None

    def process(self, prod, lead_process=None):
        if lead_process is not None and not lead_process.finished:
            self.finished = False
            self.start_time = -1
            self.set_obs_inf(self.obs_inf_k, {'time':True}, replace=False)
        elif prod is not None and not self.finished:
            t = self.process_unit.get_time()
            if self.start_time == -1:
                self.start_time = t
            lead_process_finish_time = 0 if lead_process is None else lead_process.finish_time
            if self._justify(prod, t - lead_process_finish_time):
                self.finished = True
                self.finish_time = t
                self.last_prod = prod
            self.set_obs_inf(self.obs_inf_k, {'time':{'start':self.start_time, 'current':t,
                                                      'end':self.finish_time,
                                                      'env_time': self.process_unit.env.time,
                                                      'prod': prod.m_id.get_level_value(1)}}, replace=False)
        else:
            self.set_obs_inf(self.obs_inf_k, {'time':True}, replace=False)
            pass
        return self.finished

    @abstractmethod
    def _justify(self, prod, t):
        pass


class Setup(ProductionProcess):

    def __init__(self, process_unit):
        super().__init__(process_unit)
        # self.obs_inf_k = "setup_" + str(self.process_unit.m_id)

    def refresh_inf(self):
        self.refresh_increment_inf("setup cost", self.cost)
        self.cost = 0

    @abstractmethod
    def _justify(self, prod, t):
        pass

    @staticmethod
    def load(json, process_unit):
        if 'setup' not in json:
            return None
        else:
            json = json['setup']
        if json["type"] == "changeover":
            return ChangeoverSetup(setup_expiry=Distribution.load(json, "setup_expiry"),
                                   changeover_time=Distribution.load(json, "changeover_time"),
                                   changeover_cost=Distribution.load(json, "changeover_cost"),
                                   turnaround_time=Distribution.load(json, "turnaround_time"),
                                   turnaround_cost=Distribution.load(json, "turnaround_cost"),
                                   process_unit=process_unit)
        else:
            raise NotImplementedError


class FixedSetup(Setup):

    def __init__(self, time: Distribution, cost: Distribution, process_unit):
        super().__init__(process_unit)
        self.time = time
        self.cost = cost

    def _justify(self, prod, t):
        time = self.time.sample(t)
        finish = t >= time
        if finish:
            self.cost = self.cost.sample(t)
        self.refresh_inf()
        return finish


class ChangeoverSetup(Setup):

    def reset(self):
        super().reset()
        self.changeover_cost_v = 0

    def __init__(self, setup_expiry: Distribution, changeover_time: Distribution, turnaround_time: Distribution,
                 changeover_cost: Distribution,
                 turnaround_cost: Distribution, process_unit):
        super().__init__(process_unit)
        self.setup_expiry = setup_expiry
        self.changeover_time = changeover_time
        self.changeover_cost = changeover_cost
        self.turnaround_time = turnaround_time
        self.turnaround_cost = turnaround_cost

        self.changeover_cost_v = 0

    def needs_changeover(self, prod, t):
        needs_changeover = (not prod.same_series(self.last_prod)) or (
                self.process_unit.process_start_time - self.process_unit.last_process_end_time) >= self.get_distribution(
            "setup_expiry", prod).sample(t)
        if not needs_changeover:
            print('no need changeover')
        return needs_changeover

    def _justify(self, prod: Substance, t):
        time = self.get_distribution("changeover_time", prod).sample(
            t) if self.needs_changeover(prod, t) else self.get_distribution("turnaround_time",
                                                                            prod).sample(t)
        finish = t >= time
        if finish:
            self.refresh_increment_inf("setup time", t)
            if self.needs_changeover(prod, t):
                changeover_cost = self.get_distribution("changeover_cost", prod).sample(t)
                self.cost = changeover_cost
                self.changeover_cost_v = changeover_cost
                self.refresh_increment_inf("changeover cost", changeover_cost)
                self.refresh_increment_inf("changeover time", t)
            else:
                self.cost = self.get_distribution("turnaround_cost", prod).sample(t)
                self.refresh_increment_inf("turnaround cost", changeover_cost)
                self.refresh_increment_inf("turnaround time", t)
        self.refresh_inf()
        return finish


class Failure(ProductionProcess):

    def __init__(self, failure_rate: Distribution, failure_cost_per_time: Distribution, process_unit):
        super().__init__(process_unit)
        self.failure_rate = failure_rate
        self.failure_cost_per_time = failure_cost_per_time
        # self.obs_inf_k = 'failure_' + str(self.process_unit.m_id)

    def _justify(self, prod, t):
        rate = self.get_distribution("failure_rate", prod).sample(t)
        failed = np.random.uniform() <= rate
        if failed:
            self.cost = self.get_distribution("failure_cost_per_time", prod).sample(t)
            self.refresh_increment_inf("failure cost", self.cost)
        self.refresh_inf()
        return failed

    @staticmethod
    def load(json, process_unit):
        if 'failure' not in json:
            return None
        else:
            json = json['failure']
        if json["type"] == "failure":
            return Failure(failure_rate=Distribution.load(json, "failure_rate"),
                           failure_cost_per_time=Distribution.load(json, "failure_cost_per_time"),
                           process_unit=process_unit)
        else:
            raise NotImplementedError


class Launch(ProductionProcess):

    def __init__(self, launch_time: Distribution, launch_cost: Distribution, process_unit):
        super().__init__(process_unit)
        self.launch_time = launch_time
        self.launch_cost = launch_cost
        # self.obs_inf_k = "launch_" + str(self.process_unit.m_id)
    
    def reset(self, faild, *args):
        if faild:
            if len(args) > 0:
                lead_process_finish_time = args[0]
                self.refresh_increment_inf("launch time", self.process_unit.get_time() - lead_process_finish_time)
            else:
                self.refresh_increment_inf("launch time", self.process_unit.get_time())
        super().reset()

    def _justify(self, prod: Substance, t):
        time = self.launch_time.sample(t)
        finish = t >= time
        if finish:
            self.cost = self.get_distribution("launch_cost", prod).sample(t)
            self.refresh_increment_inf("launch time", t)
            self.refresh_inf()
        else:
            self.refresh_inf()
        return finish

    def refresh_inf(self):
        self.refresh_increment_inf("launch cost", self.cost)
        self.cost = 0

    @staticmethod
    def load(json, process_unit):
        if 'launch' not in json:
            return None
        else:
            json = json['launch']
        if json["type"] == "launch":
            return Launch(launch_time=Distribution.load(json, "launch_time"),
                          launch_cost=Distribution.load(json, "launch_cost"),
                          process_unit=process_unit)
        else:
            raise NotImplementedError


class Conversion(ProductionProcess):

    def __init__(self,
                 conversion_prepare_cost: Distribution,
                 conversion_rate_per_day: Distribution, conversion_cost_per_day: Distribution,
                 conversion_rate: Distribution, conversion_cost: Distribution,
                 conversion_period: Distribution,
                 produce_delay: Distribution,
                 process_unit):
        super().__init__(process_unit)
        self.conversion_prepare_cost = conversion_prepare_cost
        self.conversion_rate_per_day = conversion_rate_per_day
        self.conversion_cost_per_day = conversion_cost_per_day
        self.conversion_rate = conversion_rate
        self.conversion_cost = conversion_cost
        self.conversion_period = conversion_period
        self.produce_delay = produce_delay
        self.prod = None
        # self.obs_inf_k = 'conversion_' + str(self.process_unit.m_id)

        self.conversion_prepare_cost_v = 0
        self.conversion_finish_cost_v = 0
        self.conversion_daily_cost = 0
        self.obs_inf = {}

    def reset(self, faild, *args):
        if faild:
            if len(args) > 0:
                lead_process_finish_time = args[0]
                self.refresh_increment_inf("conversion time", self.process_unit.get_time() - lead_process_finish_time)
            else:
                self.refresh_increment_inf("conversion time", self.process_unit.get_time())
        super().reset()
        self.conversion_prepare_cost_v = 0
        self.conversion_finish_cost_v = 0
        self.conversion_daily_cost = 0
        self.obs_inf = {}
        self.prod = None

    def _justify(self, prod: Material, t):
        if t == 0:
            prepare_cost = self.get_distribution("conversion_prepare_cost", prod).sample(t)
            self.cost += prepare_cost
            self.conversion_prepare_cost_v += prepare_cost
            self.refresh_increment_inf("prepare cost", prepare_cost)
        time = self.get_distribution("conversion_period", prod).sample(t)
        produce_delay = self.get_distribution("produce_delay", prod).sample(t)
        output_rate = 0
        if t >= produce_delay:
            if self.prod is None:
                output_rate += self.get_distribution("conversion_rate_per_day", prod).sample(t)
            else:
                raise NotImplementedError
        daily_cost = self.get_distribution("conversion_cost_per_day", prod).sample(t)
        self.cost += daily_cost
        self.conversion_daily_cost += daily_cost
        self.refresh_increment_inf("daily cost", daily_cost)
        finish = t >= time
        if finish:
            finish_cost = self.get_distribution("conversion_cost", prod).sample(t)
            self.cost += finish_cost
            self.conversion_finish_cost_v += finish_cost
            self.refresh_increment_inf("finish cost", finish_cost)
            self.refresh_increment_inf("conversion time", t)
            if self.prod is None:
                output_rate += self.get_distribution("conversion_rate", prod).sample(t)
            else:
                raise NotImplementedError
        self.prod = prod.conversion(self.process_unit.consume_material and finish, output_rate)
        self.refresh_increment_inf("total cost", self.cost)
        self.cost = 0
        return finish

    def refresh_inf(self):
        pass

    def convert(self, prod):
        return

    @staticmethod
    def load(json, process_unit):
        if 'conversion' not in json:
            return None
        else:
            json = json['conversion']
        if json["type"] == "conversion":
            # print(json)
            return Conversion(conversion_prepare_cost=Distribution.load(json, "conversion_prepare_cost"),
                              conversion_rate_per_day=Distribution.load(json, "conversion_rate_per_day"),
                              conversion_cost_per_day=Distribution.load(json, "conversion_cost_per_day"),
                              conversion_rate=Distribution.load(json, "conversion_rate"),
                              conversion_cost=Distribution.load(json, "conversion_cost"),
                              conversion_period=Distribution.load(json, "conversion_period"),
                              produce_delay=Distribution.load(json, "produce_delay"),
                              process_unit=process_unit)
        else:
            raise NotImplementedError


class Store(Process):

    def reset(self):
        super().reset()
        self.inventory = {}
        self.total = 0
        self.prod_total = {}
        self.wastage = 0
        self.wast_cost = 0
        self.inventory_cost = 0

    def refresh_inf(self):
        pass

    def __init__(self, quality_guarantee_period: Distribution,
                 inventory_hold_cost_per_unit_per_unit_date: Distribution,
                 wastage_rate_per_unit: Distribution, process_unit):
        super().__init__(process_unit)
        self.quality_guarantee_period = quality_guarantee_period
        self.inventory_hold_cost_per_unit_per_unit_date = inventory_hold_cost_per_unit_per_unit_date
        self.wastage_rate_per_unit = wastage_rate_per_unit
        # self.obs_inf_k = "store_" + str(self.process_unit.m_id)

        self.inventory = {}
        self.total = 0
        self.prod_total = {}
        self.wastage = 0
        self.wast_cost = 0
        self.inventory_cost = 0

    def store(self, product):
        prod = product.m_id.get_level_value(1)
        if prod not in self.inventory:
            self.inventory[prod] = []
        if prod not in self.prod_total:
            self.prod_total[prod] = 0
        self.inventory[prod].append(product)
        self.add_quantity(prod, product.quantity)

    def refresh_total_quantity(self):
        self.total = 0
        for prod in self.inventory:
            self.prod_total[prod] = 0
            for s in self.inventory[prod]:
                self.total += s.quantity
                self.prod_total[prod] += s.quantity

    def add_quantity(self, prod, quantity):
        assert quantity >= 0
        self.prod_total[prod] += quantity
        self.total += quantity
        if self.total <= 0 or self.prod_total[prod] <= 0:
            self.refresh_total_quantity()
        assert self.total >= 0 and self.prod_total[prod] >= 0, (prod, quantity, self.prod_total[prod])

    def minus_quantity(self, prod, quantity):
        assert quantity >= 0
        self.prod_total[prod] -= quantity
        self.total -= quantity
        if self.total <= 0 or self.prod_total[prod] <= 0:
            self.refresh_total_quantity()
        assert self.total >= 0 and self.prod_total[prod] >= 0, (prod, quantity, self.total, self.prod_total[prod])

    def fetch(self, prod, quantity):
        if prod not in self.prod_total or quantity <= 0 or len(self.inventory[prod]) == 0 or self.prod_total[prod] == 0:
            return 0
        else:
            ret = 0
            while len(self.inventory[prod]) > 0:
                buff = self.inventory[prod][0]
                if buff.quantity <= quantity:
                    delta = buff.quantity if buff.quantity < self.prod_total[prod] else self.prod_total[prod]
                    self.minus_quantity(prod, delta)
                    ret += delta
                    quantity -= delta
                    self.inventory[prod].remove(buff)
                else:
                    self.minus_quantity(prod, quantity)
                    buff.quantity -= quantity
                    ret += quantity
                    quantity = 0
                n = self.fetch(prod, quantity)
                if n == 0:
                    break
            return ret

    def remove_overdue_product(self, no_need_to_check: dict = None):
        if no_need_to_check is None:
            no_need_to_check = {}
            for prod, _ in self.inventory.items():
                no_need_to_check[prod] = False
                return self.remove_overdue_product(no_need_to_check)
        else:
            no_need = True
            for prod, _no_need in no_need_to_check.items():
                sub = Substance.substance[MID(SUBSTANCE_ID_PREFIX + "_{}_2".format(prod))]
                if not _no_need:
                    no_need_to_check[prod] = len(self.inventory[prod]) == 0 or ((self.process_unit.env.time -
                                                                                 self.inventory[prod][
                                                                                     0].born_time) <= self.get_distribution(
                        "quality_guarantee_period", sub).sample(self.process_unit.get_time()))
                _no_need = no_need_to_check[prod]
                if not _no_need:
                    # print("wast")
                    buff = self.inventory[prod][0]
                    wast_quantity = buff.quantity
                    self.wastage += wast_quantity
                    self.minus_quantity(prod, wast_quantity)
                    self.wast_cost += wast_quantity * self.get_distribution("wastage_rate_per_unit", sub).sample(
                        self.process_unit.get_time())
                    self.inventory[prod].remove(buff)
                no_need = no_need and _no_need
            if no_need:
                self.set_obs_inf(self.obs_inf_k, {
                    "wastage": self.wastage,
                    "wast cost": self.wast_cost
                })
                return True
            else:
                return self.remove_overdue_product(no_need_to_check)
        pass

    def refresh(self):
        self.remove_overdue_product()
        for prod, inventory in self.prod_total.items():
            sub = Substance.substance[MID(SUBSTANCE_ID_PREFIX + "_{}_2".format(prod))]
            self.inventory_cost += self.total * self.get_distribution("inventory_hold_cost_per_unit_per_unit_date",
                                                                      sub).sample(self.process_unit.get_time())
        obs = self.get_obs_inf(self.obs_inf_k, "inventory cost", {"inventory cost": self.inventory_cost})
        obs["inventory cost"] = self.inventory_cost
        # for prob, inventory in self.inventory.items():
        pass

    @staticmethod
    def load(json, process_unit):
        if json["type"] == "product":
            return Store(quality_guarantee_period=Distribution.load(json, "quality_guarantee_period"),
                         inventory_hold_cost_per_unit_per_unit_date=Distribution.load(json,
                                                                                      "inventory_hold_cost_per_unit_per_unit_date"),
                         wastage_rate_per_unit=Distribution.load(json, "wastage_rate_per_unit"),
                         process_unit=process_unit)
        else:
            raise NotImplementedError


class Sale(Process):

    def reset(self):
        super().reset()
        self.sales_volume = 0
        self.backlog_decay = 0
        self.backlog_penalty = 0

    def refresh_inf(self):
        pass

    def __init__(self, price_per_unit: Distribution,
                 backlog_penalty_cost_per_unit_per_day: Distribution,
                 backlog_decay_per_half_year: float, process_unit):
        super().__init__(process_unit)
        self.price_per_unit = price_per_unit
        self.backlog_penalty_cost_per_unit_per_day = backlog_penalty_cost_per_unit_per_day
        self.backlog_decay_per_half_year = 1 - math.pow(backlog_decay_per_half_year, 1 / 180)
        # self.obs_inf_k = "sale_" + str(self.process_unit.m_id)

        self.sales_volume = 0
        self.backlog_decay = 0
        self.backlog_penalty = 0

    def sale(self, product_storage_unit, customer_unit):
        total_demand = 0
        total_demand_fit = 0
        for prod, d in customer_unit.demand.demand_dict.items():
            ret = product_storage_unit.fetch(prod, d["not_fit"])
            d["fit"] += ret
            total_demand += d["all"]
            total_demand_fit += d["fit"]
            d["not_fit"] -= ret
            sub = Substance.substance[MID(SUBSTANCE_ID_PREFIX + "_{}_2".format(prod))]
            sales_volume = ret * self.get_distribution("price_per_unit", sub).sample(self.process_unit.get_time())
            self.sales_volume += sales_volume
            d["sales_volume"] += sales_volume
            backlog_penalty = d["not_fit"] * self.get_distribution("backlog_penalty_cost_per_unit_per_day",
                                                                         sub).sample(self.process_unit.get_time())
            self.backlog_penalty += backlog_penalty
            d["backlog_penalty"] += backlog_penalty
            decay = d["not_fit"] * self.backlog_decay_per_half_year
            self.backlog_decay += decay
            d["decay"] += decay
            d["not_fit"] -= decay
        # self.set_obs_inf(self.obs_inf_k, {"sales volume": self.sales_volume, "backlog decay": self.backlog_decay,
        #                                   "backlog penalty": self.backlog_penalty,
        #                                   "csl": total_demand_fit / total_demand * 100})

        product_storage_unit.store.refresh()
        store: Store = product_storage_unit.store
        demand: Demand = customer_unit.demand
        _all = {}
        fit = {}
        not_fit = {}
        decay = {}
        csl = {}
        inventory = {}
        sales_volume = {}
        backlog_penalty = {}
        unit_price = {}
        for p, d in demand.demand_dict.items():
            _all[p] = d['all']
            fit[p] = d['fit']
            not_fit[p] = d['not_fit']
            decay[p] = d['decay']
            csl[p] = d['fit'] / d['all'] if d['all'] != 0 else 0
            inventory[p] = store.prod_total[p] if p in store.prod_total else 0
            sales_volume[p] = d['sales_volume']
            backlog_penalty[p] = d['backlog_penalty']
            unit_price[p] = d["sales_volume"] / d['fit'] if d['fit'] > 0 else 0
        new_csl = total_demand_fit / total_demand * 100
        last_csl = self.process_unit.env.variables.get_variable("CSL")
        self.set_obs_inf(self.obs_inf_k,
                         {"sales volume": self.sales_volume, "backlog decay": self.backlog_decay,
                          "backlog penalty": self.backlog_penalty,
                          "csl": new_csl,
                          "delta csl": new_csl - last_csl,
                          "all order values": _all, "fit order values": fit,
                          "not fit order values": not_fit,"decay values": decay,
                          "inventory values": inventory,
                          "unit price s": unit_price,
                          "sales volume s": sales_volume,
                          "csl s": csl})

    @staticmethod
    def load(json, process_unit):
        if json["type"] == "sale":
            # print(json)
            return Sale(price_per_unit=Distribution.load(json, "price_per_unit"),
                        backlog_penalty_cost_per_unit_per_day=Distribution.load(json,
                                                                                "backlog_penalty_cost_per_unit_per_day"),
                        backlog_decay_per_half_year=(1 if "backlog_decay_per_half_year" not in json else float(
                            json["backlog_decay_per_half_year"])),
                        process_unit=process_unit)
        else:
            raise NotImplementedError


class Demand(Process):

    __initial_demand_dict__ = {"all": 0, "fit": 0, "not_fit": 0,
                               "decay": 0, "sales_volume": 0, "backlog_penalty": 0}

    def reset(self):
        super().reset()
        self.demand_dict = {}  # 用于保存累计值
        for prod, dist in self.demand_json.items():
            self.demand_dict[prod] = copy.deepcopy(self.__initial_demand_dict__)

    def refresh_inf(self):
        pass

    def get_demand_u(self):
        u_table = []
        for prod, dist in self.demand_dist.items():
            if isinstance(dist, NormalDistribution):
                u_table.append(dist.mean.calculate(0))
            else:
                raise NotImplementedError
        return u_table

    def __init__(self, demand_json: dict, demand_scale, process_unit):
        super().__init__(process_unit)
        self.demand_json = demand_json
        self.demand_dist = {}  # 用于保存分布
        self.demand_scale = demand_scale

        self.demand_dict = {}  # 用于保存累计值
        for prod, dist in demand_json.items():
            self.demand_dist[prod] = Distribution.load(demand_json, prod)
            self.demand_dict[prod] = copy.deepcopy(self.__initial_demand_dict__)
        pass

    def generate(self):
        for prod, dist in self.demand_dist.items():
            dist: Distribution
            _demand = dist.sample(self.process_unit.get_time()) * self.demand_scale
            self.demand_dict[prod]["all"] += _demand
            self.demand_dict[prod]["not_fit"] += _demand

    def fit(self, prob, quantity):
        self.demand_dict[prob]["fit"] += quantity


class Scheduling(Process):

    def reset(self):
        super().reset()
        self.obs = []
        self.idle_cnt = 0
        available_actions = self.process_unit.env.get_available_actions()
        for i in range(self.prod_feature_dim * len(available_actions) + self.env_feature_dim):
            self.obs.append(0)
        return self.obs

    def __init__(self, scheduling_period: Distribution,obs_define: dict, reward_define, process_unit):
        super().__init__(process_unit)
        self.scheduling_period = scheduling_period
        # self.obs_inf_k = 'scheduling_' + str(self.process_unit.m_id)
        self.product_obs_define = obs_define["product"]
        self.prod_feature_dim = len(self.product_obs_define)
        self.env_obs_define = obs_define["env"]
        self.env_feature_dim = len(self.env_obs_define)
        self.reward_define = reward_define
        available_actions = []
        for sub_id in Substance.substance:
            if sub_id.get_level_value(1) not in available_actions:
                available_actions.append(sub_id.get_level_value(1))

        self.last_scheduling_time = 0
        self.last_profit = 0
        self.delta_time = 0
        self.delta_profit = 0
        self.last_action = None
        self.action = None
        self.obs = []
        self.idle_cnt = 0
        for i in range(self.prod_feature_dim * len(available_actions) + self.env_feature_dim):
            self.obs.append(0)

    def on_new_scheduling(self, action):
        self.last_scheduling_time = self.process_unit.get_time()
        self.last_profit = self.process_unit.env.variables.get_variable("Profit")
        self.delta_time = 0
        self.delta_profit = 0
        self.obs = []
        self.last_action = 0 if self.action is None else self.action
        self.action = action

    def on_scheduling_finished(self):
        self.delta_time = self.process_unit.get_time() - self.last_scheduling_time
        self.delta_profit = self.process_unit.env.variables.get_variable("Profit") - self.last_profit

        last_action = self.last_action
        action = self.action
        delta_time = self.delta_time
        self.set_obs_inf(self.obs_inf_k, {"delta profit": self.delta_profit, "last_action":last_action,"action":action,
                                          "delta time":delta_time}, replace=False)
        # self.set_obs_inf(self.obs_inf_k, {"all order values": _all, "fit order values": fit,
        #                                   "not fit order values": not_fit,"decay values": decay,
        #                                   "inventory values": inventory,"delta profit": self.delta_profit,
        #                                   "csl s": csl, "last_action":last_action,"action":action,
        #                                   "delta time":delta_time, "current time":self.process_unit.get_time()})
        available_actions = self.process_unit.env.get_available_actions()
        for p in available_actions:
            self.set_obs_inf(self.obs_inf_k, {"p": p}, replace=False)
            for product_obs_var in self.product_obs_define:
                self.obs.append(self.process_unit.env.variables.get_variable(product_obs_var))
        for env_obs_ver in self.env_obs_define:
            self.obs.append(self.process_unit.env.variables.get_variable(env_obs_ver))

    def scheduling(self, action):
        try:
            action = int(action)
        except TypeError:
            pass
        if isinstance(action, int):
            if self.process_unit.needs_scheduling():
                self.on_new_scheduling(action)
            else:
                action = None
            self.refresh_obs_inf_keys()
            self.process_unit.get_latter_unit().process(action)
            self.process_unit.env.time += 1
            if True:
                self.set_obs_inf("available_actions", {self.process_unit.env.time:self.process_unit.env.get_available_actions()},
                                 replace=False)
                self.set_obs_inf("selected_action", {self.process_unit.env.time:action},
                                 replace=False)
            if action == 0:
                self.idle_cnt += 1
            self.set_obs_inf(self.obs_inf_k, {"current time":self.process_unit.get_time()}, replace=False)
            self.set_obs_inf(self.obs_inf_k, {'idle percent': self.idle_cnt / self.process_unit.env.time}, replace=False)
            done = self.process_unit.get_time() >= self.get_distribution("scheduling_period", None).sample()
            if self.process_unit.needs_scheduling() or done:
                self.on_scheduling_finished()
                self.set_obs_inf(self.obs_inf_k, {"p": action}, replace=False)
                reward = self.process_unit.env.variables.get_variable(self.reward_define)
                return self.obs, reward, done
            return self.obs, 0, done
        else:
            raise NotImplementedError

    @staticmethod
    def load(json, process_unit):
        if "type" not in json:
            if 'scheduling' not in json:
                return None
            else:
                json = json['scheduling']
        if json["type"] == "scheduling":
            p = Scheduling(scheduling_period=Distribution.load(json, "scheduling_period"),
                           obs_define=json["obs"],
                           reward_define=json["reward"],
                           process_unit=process_unit)
            return p
        else:
            raise NotImplementedError
