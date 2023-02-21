import json
import os
import re

from elsp_env_manager.base.model_manager import load_model


class Variables(object):

    def __init__(self, map_json: dict, define_json: dict, inf=None):
        self.map_json = map_json
        self.define_json = define_json
        self.inf = inf

    @staticmethod
    def find_variable(_str: str):
        c = _str.count('$')
        assert c % 2 == 0
        st = 0
        ed = -1
        v_list = []
        for i in range(c//2):
            st = _str.find("$", ed + 1)
            ed = _str.find("$", st + 1)
            v_list.append(_str[st + 1:ed])
        return v_list
    
    def set_variable(self, name, value):
        assert self.inf is not None
        if name in self.map_json:
            map_json = self.map_json[name]
            k = map_json["process_name"] + "_" + map_json["m_id"]
            if k in self.inf and map_json["label"] in self.inf[k]:
                self.inf[k][map_json["label"]] = value

    def get_variable(self, name):
        assert self.inf is not None
        if name in self.map_json:
            map_json = self.map_json[name]
            k = map_json["process_name"] + "_" + map_json["m_id"]
            if k in self.inf and map_json["label"] in self.inf[k]:
                return self.inf[k][map_json["label"]]
            else:
                return 0
                #raise RuntimeWarning
        elif name in self.define_json:
            expression = self.define_json[name]
            expression_2 = expression
            if isinstance(expression, str):
                v_list = Variables.find_variable(expression)
                _locals = locals()
                cnt = 0
                for v in v_list:
                    # print(v, self.get_variable(v))
                    v_name = "var_" + str(cnt)
                    v_name_2 = "var_" + str(cnt) + '_2'
                    cnt += 1
                    if v == 'P':
                        #print(v_list)
                        pass
                    _locals[v_name] = self.get_variable(v)
                    _locals[v_name_2] = self.get_variable(v) if v != 'P' else str(self.get_variable(v))
                    expression = expression.replace("${}$".format(v), v_name)
                    expression_2 = expression_2.replace("${}$".format(v), v_name_2)
                try:
                    res = eval(expression)
                except KeyError:
                    try:
                        res = eval(expression_2)
                    except KeyError as e:
                        return 0
                    except ZeroDivisionError:
                        print('ZeroDivisionError', name, expression_2, _locals)
                        raise ZeroDivisionError
                except ZeroDivisionError:
                    # print('ZeroDivisionError', name, self.inf)
                    return 0
                except AttributeError:
                    print(name, expression, self.inf)
                    raise AttributeError
                except TypeError:
                    return 0
                except Exception:
                    print(name, self.inf)
                    raise Exception
                return res
            else:
                raise NotImplementedError
        else:
            print(name)
            raise NotImplementedError
        pass

    @staticmethod
    def load(map_json_file, define_json_file, inf):
        if isinstance(map_json_file, dict):
            map_json = map_json_file
        else:
            map_json = load_model(map_json_file)
        if isinstance(define_json_file, dict):
            define_json = define_json_file
        else:
            define_json = load_model(define_json_file)

        return Variables(map_json=map_json, define_json=define_json, inf=inf)

