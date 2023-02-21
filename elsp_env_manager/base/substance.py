import copy

from elsp_env_manager.base import MID, Distribution
from elsp_env_manager.base.constants import SUBSTANCE_ID_PREFIX
from elsp_env_manager.base.model_manager import load_model


class Substance(object):
    substance: dict = {}

    def __init__(self, _id: MID, model_json: dict = None):
        self.m_id = _id
        self.incomplete_operation = [] if model_json is None or "incomplete_operation" not in model_json else model_json["incomplete_operation"]
        self.quantity = 0
        self.model_json = model_json

    def same_series(self, l1_or_id):
        return self.m_id.get_level_value(1) == l1_or_id if not isinstance(l1_or_id, MID) else self.m_id.get_level_value(
            1) == l1_or_id.get_level_value(1)

    def next_production_substance(self):
        return Substance.instance(self.m_id.next_production_id(), 0)

    @staticmethod
    def load(json_or_json_path):
        if isinstance(json_or_json_path, dict):
            json = json_or_json_path
        else:
            json = load_model(json_or_json_path)
        for p_id, p in json.items():
            if p is None:
                m_id = MID(SUBSTANCE_ID_PREFIX + "_{}".format(p_id))
                Substance.substance[m_id] = Substance(m_id)
                continue
            for sp_id, sp in p.items():
                m_id = MID(SUBSTANCE_ID_PREFIX + "_{}".format(p_id) + "_{}".format(sp_id))
                if sp["type"] == "material":
                    Substance.substance[m_id] = Material(_id=m_id, model_json=sp)
                elif sp["type"] == 'product':
                    Substance.substance[m_id] = Product(_id=m_id, model_json=sp)
                else:
                    raise NotImplementedError
                # Product(m_id, [], quantity=0)

            # products[m_id] = Product(m_id, [], quantity=0)

    @staticmethod
    def instance(_id: MID, quantity: float):
        if _id not in Substance.substance and _id.id_level >= 2:
            _id = _id.upper()
        else:
            assert NotImplementedError
        if _id not in Substance.substance:
            return None
        sub: Substance = copy.deepcopy(Substance.substance[_id])
        sub.quantity = quantity
        return sub

    @staticmethod
    def get_instance_by_l1(l1: str):
        return Substance.instance(MID("Substance_{}".format(l1)), 0)

    @staticmethod
    def get_instance(id_or_l1):
        if id_or_l1 not in Substance.substance:
            sub = Substance.get_instance_by_l1(id_or_l1)
        else:
            sub = Substance.substance[id_or_l1]
        return sub


class Material(Substance):

    def __init__(self, _id: MID, model_json: dict = None):
        super().__init__(_id, model_json)

    def conversion(self, consume_material, rate):
        res: Substance = self.next_production_substance()
        res.quantity = rate * self.quantity
        if consume_material:
            self.quantity = 0
        return res


class Product(Substance):

    def __init__(self, _id, model_json: dict = None):
        super().__init__(_id, model_json)
        self.born_time = 0
