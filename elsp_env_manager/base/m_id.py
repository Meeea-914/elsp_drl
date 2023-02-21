from elsp_env_manager.base.constants import *


class MID(object):

    def __init__(self, _id: str):
        self._id = _id
        buff = _id.split("_")
        self.type = buff[0]
        self.id_level = 0
        for i in range(1, len(buff)):
            self.__setattr__("l{}".format(i), buff[i])
            self.id_level += 1
        assert self.id_level >= 1, _id

    def next_production_id(self):
        assert self.type == SUBSTANCE_ID_PREFIX
        assert self.id_level == 2
        return MID(self.get_id(self.id_level - 1) + '_{}'.format(int(self.get_level_value(2)) + 1))

    def get_level_value(self, lev):
        assert lev <= self.id_level
        return self.__getattribute__("l{}".format(lev))

    def get_id(self, lev=None):
        if lev is None or lev > self.id_level:
            return self._id
        else:
            _id = self.type
            for i in range(1, lev + 1):
                _id += "_{}".format(self.get_level_value(i))
            return _id

    def upper(self):
        return MID(self.get_id(self.id_level - 1))

    def same_level_id_of_type(self, _type):
        return MID(self._id.replace(self.type, _type))

    def __eq__(self, other):
        if not isinstance(other, MID):
            return False
        return self._id == other._id

    def __hash__(self):
        return self._id.__hash__()

    def __str__(self):
        return self._id
