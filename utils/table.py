import copy
import json


class Column(object):

    def __init__(self, origin_data, head_list: list, return_as_list=False,
                 data_decoder=None, data_encoder=None):
        self.origin_data = origin_data  # if isinstance(origin_data, list) else list(origin_data)
        self.return_as_list = return_as_list
        self.data_decoder = data_decoder
        self.data_encoder = data_encoder
        self.data = {}
        for i in range(len(head_list)):
            head = head_list[i]
            data = origin_data[i]
            if not isinstance(data, Column):
                self.data[head] = data if data_decoder is None or not callable(data_decoder) else data_decoder(head, data)
            else:
                self.data[head] = data
        self.head_key = self.data.keys()
        self.upper_column = None

    def to_update_str(self, head_list=None):
        update_list = []
        for head in self.head_list if head_list is None else head_list:
            if head != 'id':
                update_list.append('{} = {}'.format(head, self[head] if self.data_encoder is None or not callable(
                    self.data_encoder) else self.data_encoder(head, self[head])))
        return ','.join(update_list)

    def __getattr__(self, item):
        if item == 'head_list':
            return list(self.head_key)

    def __setitem__(self, key, value):
        self.data[key] = value if not callable(self.data_decoder) else self.data_decoder(key, value)

    def __getitem__(self, key):
        if key in self.head_list:
            item = self.data[key]
            return self.process_getitem(item)
        else:
            print(key, self.head_list)
            raise NotImplementedError

    def __contains__(self, key):
        return self.data.__contains__(key)

    def process_getitem(self, item):
        if isinstance(item, Column):
            if self.return_as_list:
                return list(item.data.values())
            return item.data
        else:
            return item

    def data_to_list(self):
        return list(self.data.values())

    def data_to_tuple(self):
        return tuple(self.data.values())

    def append(self, value, key):
        self.data[key] = value

    def __str__(self):
        return str(self.__class__) + str(self.data)


class Table(Column):

    def __init__(self, origin_data, head_list, return_as_list=False, data_decoder=None, data_encoder=None, row_head_key='id'):
        self.table_origin_data = origin_data
        self.table_head_list = head_list
        self.row_origin_data = []
        self.data_decoder = data_decoder
        self.data_encoder = data_encoder
        self.row_head_list = []
        self.transposed_data = {}
        self.return_as_list = return_as_list
        self.tb = None

        def empty(**kwargs):
            pass

        self.tb_change_listener: callable = empty
        self.needs_update_row = []
        for i in range(len(self.table_origin_data)):
            column = Column(origin_data=self.table_origin_data[i], head_list=self.table_head_list,
                            data_decoder=data_decoder, data_encoder=data_encoder)
            self.row_origin_data.append(column)
            self.row_head_list.append(column[row_head_key] if row_head_key in column else i)
        super().__init__(origin_data=self.row_origin_data, head_list=self.row_head_list, return_as_list=return_as_list,
                         data_decoder=data_decoder, data_encoder=data_encoder)

        # self.transposed_row_origin_data = []
        # for col in range(len(self.table_head_list)):
        #     head = head_list[col]
        #     col_column_origin_data = {}
        #     for row in range(len(self.table_origin_data)):
        #         if col == 0:
        #             self.col_head_list.append(row)
        #         col_column_origin_data[head] = self.table_origin_data[row][col]
        #     self.transposed_row_origin_data.append(Column(origin_data=col_column_origin_data, head_list=self.col_head_list))

    def __getitem__(self, key):
        if key in self.head_list:
            return super().process_getitem(self.data[key])
        elif key in self.table_head_list:
            if key not in self.transposed_data:
                origin_data = []
                for row, data in self.data.items():
                    origin_data.append(data[key])
                try:
                    self.transposed_data[key] = Column(origin_data=origin_data, head_list=self.row_head_list)
                except IndexError:
                    print(origin_data, self.row_head_list)
                    print(len(origin_data), len(self.row_head_list))
                    raise IndexError
            return super().process_getitem(self.transposed_data[key])
        else:
            print('not implemented', key, type(key), self.head_list, self.table_head_list)
            raise NotImplementedError

    def data_to_list(self):
        list_data = []
        for row, data in self.data.items():
            data: Column
            list_data.append(data.data)
        return list_data

    def data_dict(self, deep_copy=True):
        dict_data = {}
        for row, data in self.data.items():
            data: Column
            dict_data[row] = data.data
        return dict_data if not deep_copy else copy.deepcopy(dict_data)

    def retrieve_from_db(self, **kwargs):
        res = self.tb.retrieve(fields=self.table_head_list,
                               condition=self.tb.kw_to_condition(**kwargs))[-1:]
        tb = Table(origin_data=res, head_list=self.table_head_list,
                     return_as_list=self.return_as_list, data_decoder=self.data_decoder,
                     data_encoder=self.data_encoder)
        return tb

    def delete_by_id(self, _id):
        res = self.tb.delete_by_id(_id)
        if res and int(_id) in self.data:
            pop_inf = self.data.pop(int(_id))
            self.row_head_list.pop(self.row_head_list.index(int(_id)))
            self.transposed_data.clear()
            self.transposed_data = {}
            self.tb_change_listener(change_type='delete', inf=pop_inf)
            return res
        else:
            return res

    def add_row_needs_update(self, _id):
        _id = int(_id)
        if _id not in self.needs_update_row:
            self.needs_update_row.append(_id)
            self.tb_change_listener(change_type='update', inf=self.data[_id].data)
        if len(self.needs_update_row) > 0:
            print('table: ', self.tb.tb_name, 'needs update, id: ', self.needs_update_row)

    def append(self, data, key=None):
        if isinstance(data, list):
            for d in data:
                self.append(d)
        elif isinstance(data, tuple):
            column = Column(origin_data=data, head_list=self.table_head_list, data_decoder=self.data_decoder)
            self.append(column)
        elif isinstance(data, Column) and data.head_list == self.table_head_list:
            index = len(self.data) if 'id' not in data else data['id']
            if index not in self.row_head_list:
                self.row_head_list.append(index)
            self.data[index] = data
            self.tb_change_listener(change_type='append', inf=data.data)
            self.transposed_data.clear()
            self.transposed_data = {}
        elif isinstance(data, Table) and data.table_head_list == self.table_head_list:
            for index, column in data.data.items():
                self.append(column)
        else:
            print(data, type(data))
            raise NotImplementedError(data)


if __name__ == '__main__':
    def decoder(head, value):
        if isinstance(value, str) and (value.startswith('[')):
            return json.loads(value)
        else:
            return value


    table = Table(
        [(1, '[]', 'test02', '1', '[]'), (2, '[]', 'test02', '2', '[]'), (3, '[]', 'test02', '3', '[]')],
        ['id', 'banned_sub_permission', 'creator', 'name', 'sub_permission'], return_as_list=True, data_decoder=decoder
    )
    print(table.data_dict())
