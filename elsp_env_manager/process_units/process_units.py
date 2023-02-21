from abc import ABC, abstractmethod
from elsp_env_manager.base import MID, Substance, Material, Product
from elsp_env_manager.base.constants import *
from elsp_env_manager.base.model_manager import load_model
from elsp_env_manager.process import Setup, Launch, Conversion, Demand, Sale, Store
from elsp_env_manager.process.process import Scheduling, Failure


class ProcessUnit(ABC):

    def __init__(self, m_id: MID, model_json, env):
        self._former_unit = None
        self._latter_unit = None
        self.env = env
        self.process_start_time = env.time
        self.last_process_start_time = env.time
        self.last_process_end_time = 0
        self.m_id = m_id
        self.model_json = model_json
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def get_time(self):
        return self.env.time - self.process_start_time

    def set_latter_unit(self, unit):
        self._latter_unit = unit
        unit._former_unit = self

    def set_former_unit(self, unit):
        self._former_unit = unit
        unit._latter_unit = self

    def get_latter_unit(self):
        return self._latter_unit

    def get_former_unit(self):
        return self._former_unit

    @abstractmethod
    def process(self, *args):
        pass

    @abstractmethod
    def reset(self):
        pass


class ProductionUnit(ProcessUnit):
    product_units = {}

    def reset(self):
        lead_process_finish_time = -1
        if self.setup is not None:
            lead_process_finish_time = self.setup.finish_time
            self.setup.reset()
        if self.launch is not None:
            not_finish = self.launch.finish_time == -1 or self.launch.finish_time == 0
            if not_finish == -1 and lead_process_finish_time != -1:
                self.launch.reset(True, lead_process_finish_time)
                lead_process_finish_time = self.get_time()
            else:
                lead_process_finish_time = self.launch.finish_time
        if self.conversion is not None:
            not_finish = self.conversion.finish_time == -1 or self.conversion.finish_time == 0
            if not_finish and lead_process_finish_time != -1:
                self.conversion.reset(True, lead_process_finish_time)
                lead_process_finish_time = self.get_time()
            else:
                lead_process_finish_time = self.conversion.finish_time
        if self.failure is not None:
            self.failure.reset()
        self.idle = True

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.setup = Setup.load(model_json, self)
        self.launch = Launch.load(model_json, self)
        self.conversion = Conversion.load(model_json, self)
        self.failure = Failure.load(model_json, self)
        self.process_list = ['setup','launch','conversion','failure']
        self.available_material_list = []
        self.consume_material = model_json["consume_material"]
        self.idle = True

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'idle':
            self.conversion.set_obs_inf(self.conversion.obs_inf_k, {'idle': value}, False)

    def process(self, material: Substance):
        # print(material.m_id.get_level_value(1))
        if material.model_json is None:
            self.idle = True
            if self.get_latter_unit() is not None:
                self.get_latter_unit().process(material)
            else:
                raise RuntimeError
            return True
        else:
            assert self.m_id.get_level_value(1) in material.model_json["incomplete_operation"]
            if self.idle:
                self.last_process_start_time = self.process_start_time
                self.process_start_time = self.env.time
                self.idle = False
            # print(material.model_json)
            if self.setup is not None:
                self.setup.process(material)
            if self.launch is not None:
                self.launch.process(material, self.setup)
            if self.failure is not None:
                failed = self.failure.process(material, self.launch)
                if failed:
                    # if self.get_time() < 20:
                    #     print('failed', self.get_time())
                    self.reset()
                    self.get_latter_unit().reset()
                    self.last_process_end_time = self.env.time
                    return failed

            finished = self.conversion.process(material, self.launch) if self.conversion is not None else True
            if isinstance(self.conversion.prod, Product):
                self.conversion.prod.born_time = self.env.time
            self.get_latter_unit().process(self.conversion.prod)

            self.conversion.prod = None
            if finished:
                if self.setup is not None:
                    self.setup.reset()
                if self.launch is not None:
                    self.launch.reset(False)
                if self.conversion is not None:
                    self.conversion.reset(False)
                self.idle = True
                self.last_process_end_time = self.env.time
            return finished

    @staticmethod
    def load(json_or_json_path, env):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for u_id, u in json.items():
            m_id = MID(_id="ProductionUnit_" + u_id)
            if u["type"] == 'continuous':
                # print(u_id, u)
                ProductionUnit.product_units[m_id] = ContinuousProductionUnit(m_id=m_id, model_json=u, env=env)
            elif u["type"] == 'batch':
                ProductionUnit.product_units[m_id] = BatchProductionUnit(m_id=m_id, model_json=u, env=env)
            else:
                raise NotImplementedError

    @staticmethod
    def get_unit_by_id(_id):
        u = ProductionUnit.product_units[MID(str(_id))] if not isinstance(_id, MID) else \
            ProductionUnit.product_units[_id]
        return u


class ContinuousProductionUnit(ProductionUnit):

    def reset(self):
        super().reset()
        self.converting_material = None

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.converting_material = None

    def process(self, material):
        if self.converting_material is not None and material is not None:
            raise RuntimeError
        if self.converting_material is None:
            self.converting_material = material
        res = super().process(self.converting_material)
        if res:
            self.converting_material = None
        return res


class BatchProductionUnit(ProductionUnit):

    def reset(self):
        for c_p_u in self.c_p_u_list:
            c_p_u.reset()
        for m in list(self.c_p_u_dict.keys()):
            self.c_p_u_dict.pop(m)

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.c_p_u_dict = {}
        self.c_p_u_list = []

    def get_c_p_u(self):
        for c_p_u in self.c_p_u_list:
            if c_p_u.idle:
                return c_p_u
        c_p_u = ContinuousProductionUnit(self.m_id, self.model_json, self.env)
        setattr(c_p_u, 'batch', True)
        self.c_p_u_list.append(c_p_u)
        return c_p_u

    def process(self, material: Material):
        m_to_remove = []
        for m, c_p_u in self.c_p_u_dict.items():
            c_p_u: ContinuousProductionUnit
            finished = c_p_u.process(m if c_p_u.idle else None)
            if finished:
                # print(material, m, m.quantity)
                m_to_remove.append(m)
        for m in m_to_remove:
            self.c_p_u_dict.pop(m)
        if material is not None and material.model_json is not None and material.quantity == 1:
            c_p_u = self.get_c_p_u()
            c_p_u.set_latter_unit(self.get_latter_unit())
            finished = c_p_u.process(material)
            if not finished:
                self.c_p_u_dict[material] = c_p_u
        return len(self.c_p_u_dict) == 0


class StorageUnit(ProcessUnit):
    storage_units = {}

    def process(self, material: Substance):
        if material.model_json is None:
            print("nothing to store")
            if self.get_latter_unit() is not None:
                self.get_latter_unit().process(material)
            else:
                print("warning: unit without follow unit")
        else:
            if material.quantity == 0:
                self.get_latter_unit().process(material)
            else:
                raise NotImplementedError
        pass

    @staticmethod
    def load(json_or_json_path, env):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for u_id, u in json.items():
            m_id = MID(_id="StorageUnit_" + u_id)
            # print(u)
            if u["type"] == "material":
                StorageUnit.storage_units[m_id] = MaterialStorageUnit(m_id, u, env)
            elif u["type"] == "product":
                StorageUnit.storage_units[m_id] = ProductStorageUnit(m_id, u, env)
            else:
                raise NotImplementedError

    @staticmethod
    def get_unit_by_id(_id):
        return StorageUnit.storage_units[MID(str(_id))] if not isinstance(_id, MID) else \
            StorageUnit.storage_units[_id]

    @abstractmethod
    def fetch(self, _id: MID, quantity: float) -> Substance:
        pass


class MaterialStorageUnit(StorageUnit):

    def reset(self):
        pass

    def fetch(self, _id: MID, quantity: float) -> Substance:
        if self.infinite:
            return Substance.instance(_id, quantity=quantity)
        else:
            raise NotImplementedError

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.infinite = model_json["infinite"]

    def process(self, material):
        if material is None:
            self.get_latter_unit().process(None)
        else:
            self.get_latter_unit().process(self.fetch(MID("Substance_{}_{}".format(material, 0)), 1))


class ProductStorageUnit(StorageUnit):

    def reset(self):
        self.store.reset()

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.store = Store.load(model_json, self)

    def process(self, product: Product):
        if product.quantity == 0:
            # self.get_latter_unit().process(product)
            pass
        else:
            self.store.store(product=product)
            pass

    def fetch(self, prod, quantity):
        return self.store.fetch(prod, quantity)


class SaleUnit(ProcessUnit):
    sale_units = {}

    def reset(self):
        self.sale.reset()

    def get_latter_unit(self):
        u = super().get_latter_unit()
        return u

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.sale = Sale.load(model_json, self)
        pass

    def process(self, product: Substance):
        if hasattr(product, "model_json") and product.model_json is None:
            print("nothing to sale")
            if self.get_latter_unit() is not None:
                self.get_latter_unit().process(product)
            else:
                raise NotImplementedError
        elif isinstance(product, tuple):
            customer_unit: CustomerUnit = product[1][1]
            product_storage_unit = product[1][0].last_unit()
            self.sale.sale(customer_unit=customer_unit, product_storage_unit=product_storage_unit)
            pass
        else:
            raise NotImplementedError
        pass

    @staticmethod
    def load(json_or_json_path, env):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for u_id, u in json.items():
            m_id = MID(_id="SaleUnit_" + u_id)
            SaleUnit.sale_units[m_id] = SaleUnit(m_id, u, env)

    @staticmethod
    def get_unit_by_id(_id):
        return SaleUnit.sale_units[MID(str(_id))] if not isinstance(_id, MID) else \
            SaleUnit.sale_units[_id]


class CustomerUnit(ProcessUnit):
    customer_units = {}

    def reset(self):
        self.demand.reset()

    def __init__(self, m_id: MID, model_json, demand_scale, env):
        super().__init__(m_id, model_json, env)
        self.demand: Demand = Demand(model_json["demand"], demand_scale, self)

    def process(self, *args):
        self.demand.generate()
        self.get_latter_unit().process(*args)
        pass

    @staticmethod
    def load(json_or_json_path, demand_scale, env):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for u_id, u in json.items():
            m_id = MID(_id=CUSTOMER_UNITS_ID_PREFIX + "_" + u_id)
            CustomerUnit.customer_units[m_id] = CustomerUnit(m_id, u, demand_scale, env)


class SchedulingUnit(ProcessUnit):
    scheduling_units = {}

    def reset(self):
        self.obs = self.scheduling.reset()
        latter_unit = self.get_latter_unit()
        while True:
            latter_unit.reset()
            latter_unit = latter_unit.get_latter_unit()
            if isinstance(latter_unit, SchedulingUnit):
                break
            pass

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        if model_json["production_unit"]["type"] == "production":
            self.production_unit = ProductionUnit.product_units[MID(PRODUCTION_UNITS_ID_PREFIX+"_{}".format(model_json["production_unit"]["id"]))]
            self.storage_unit = StorageUnit.storage_units[MID(STORAGE_UNITS_ID_PREFIX+"_1")]
            self.customer_unit = CustomerUnit.customer_units[MID(CUSTOMER_UNITS_ID_PREFIX+"_0")]
        else:
            raise NotImplementedError
        self.scheduling = Scheduling.load(model_json, self)
        
        self.obs = []
        self.reward = 0
        self.done = False
        self.inf = {}
        self.step_inf = {}
        self.action = None
        self.last_action = None

    def process(self, action):
        self.obs, self.reward, self.done = self.scheduling.scheduling(action)

    @staticmethod
    def is_idle_action(action):
        sub = Substance.get_instance_by_l1(action)
        return sub is not None and sub.model_json is None

    def needs_scheduling(self):
        return self.production_unit.idle

    @staticmethod
    def load(json_or_json_path, env):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for u_id, u in json.items():
            m_id = MID(_id=SCHEDULING_UNITS_ID_PREFIX + "_" + u_id)
            SchedulingUnit.scheduling_units[m_id] = SchedulingUnit(m_id, u, env)
