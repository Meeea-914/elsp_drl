from abc import ABC, abstractmethod

from elsp_env_manager.base import MID
from elsp_env_manager.base.constants import *
from elsp_env_manager.base.model_manager import load_model
from elsp_env_manager.process_units.process_units import ProcessUnit, ProductionUnit, StorageUnit, SchedulingUnit, \
    SaleUnit, CustomerUnit, ProductStorageUnit


class ProcessLine(ProcessUnit, ABC):
    process_line = {}

    def reset(self):
        for u in self.process_unit_list:
            u.reset()

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.process_unit_list = []

    def last_unit(self):
        return self.process_unit_list[-1] if not isinstance(self.process_unit_list[-1], ProcessLine) else self.process_unit_list[-1].last_unit()

    @abstractmethod
    def process(self, *args):
        pass

    @abstractmethod
    def set_latter_unit(self, unit):
        pass

    @abstractmethod
    def connect_units(self):
        pass

    @staticmethod
    def load(json_or_json_path, env):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for l_id, l in json.items():
            # print(l_id, l)
            m_id = MID("ProcessLine_{}".format(l_id))
            if l["type"] == 'serial':
                ProcessLine.process_line[m_id] = SerialProcessLine(m_id, l, env)
            elif l["type"] == 'parallel':
                ProcessLine.process_line[m_id] = ParallelProcessLine(m_id, l, env)
            else:
                raise NotImplementedError
        for line_id, line in ProcessLine.process_line.items():
            line: ProcessLine
            line.connect_units()
        pass

    @staticmethod
    def unit(unit_type, unit_id):
        if unit_type == "product":
            unit_id_str = PRODUCTION_UNITS_ID_PREFIX + "_{}".format(unit_id)
            unit_id = MID(unit_id_str)
            unit = ProductionUnit.product_units[unit_id]
        elif unit_type == "storage":
            unit_id_str = STORAGE_UNITS_ID_PREFIX + "_{}".format(unit_id)
            unit_id = MID(unit_id_str)
            unit = StorageUnit.storage_units[unit_id]
        elif unit_type == "process_line":
            unit_id_str = PROCESS_LINES_ID_PREFIX + "_{}".format(unit_id)
            unit_id = MID(unit_id_str)
            unit = ProcessLine.process_line[unit_id]
        elif unit_type == "schedule":
            unit_id_str = SCHEDULING_UNITS_ID_PREFIX + "_{}".format(unit_id)
            unit_id = MID(unit_id_str)
            unit = SchedulingUnit.scheduling_units[unit_id]
        elif unit_type == "sale":
            unit_id_str = SALE_UNITS_ID_PREFIX + "_{}".format(unit_id)
            unit_id = MID(unit_id_str)
            unit = SaleUnit.sale_units[unit_id]
        elif unit_type == "customer":
            unit_id_str = CUSTOMER_UNITS_ID_PREFIX + "_{}".format(unit_id)
            unit_id = MID(unit_id_str)
            unit = CustomerUnit.customer_units[unit_id]
        else:
            raise NotImplementedError
        return unit


class SerialProcessLine(ProcessLine):

    def connect_units(self):
        former_unit = None
        for unit_json in self.model_json["list"]:
            unit_type: str = unit_json["type"]
            unit_id: str = unit_json["m_id"]
            unit = ProcessLine.unit(unit_type, unit_id)
            if isinstance(former_unit, ProcessUnit):
                former_unit.set_latter_unit(unit)
            former_unit = unit
            self.process_unit_list.append(unit)

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)

    def set_latter_unit(self, unit):
        self.process_unit_list[-1].set_latter_unit(unit)

    def process(self, *args):
        self.process_unit_list[0].process(*args)
        pass


class ParallelCollectionUnit(ProcessUnit):

    def reset(self):
        pass

    def __init__(self, ppl, env):
        ppl: ParallelProcessLine
        super().__init__(ppl.m_id.same_level_id_of_type(PARALLEL_COLLECTION_UNITS_ID_PREFIX), {}, env)
        self.collection = {}
        self.current_unit = None
        self.ppl = ppl

    def process(self, *args):
        assert self.current_unit is not None
        assert len(args) == 1
        """if isinstance(self.current_unit, ProductStorageUnit):
            return"""
        self.collection[self.current_unit] = args[0]
        if self.current_unit == self.ppl.process_unit_list[-1]:
            self.get_latter_unit().process((self.collection, self.ppl.process_unit_list))

    def clear(self):
        self.collection.clear()


class ParallelProcessLine(ProcessLine):

    def set_latter_unit(self, unit):
        super().set_latter_unit(unit)
        self.collection_unit.set_latter_unit(unit)

    def get_latter_unit(self):
        return self.collection_unit.get_latter_unit()

    def connect_units(self):
        for unit_json in self.model_json["list"]:
            unit_type: str = unit_json["type"]
            unit_id: str = unit_json["m_id"]
            unit = ProcessLine.unit(unit_type, unit_id)
            unit.set_latter_unit(self.collection_unit)
            self.process_unit_list.append(unit)
        pass

    def __init__(self, m_id: MID, model_json, env):
        super().__init__(m_id, model_json, env)
        self.collection_unit = ParallelCollectionUnit(self, self.env)
        pass

    def process(self, *args):
        self.collection_unit.clear()
        for unit in self.process_unit_list:
            self.collection_unit.current_unit = unit
            unit.process(*args)
            pass
