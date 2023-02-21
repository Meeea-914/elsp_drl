import random
import numpy as np

seed = None
random.seed(seed)
np.random.seed(seed)


class BSP2(object):
    def __init__(self, u_table):
        self.u_table = u_table
        self.prod_num = len(self.u_table)
        self.m = 0
        self.I_table = []
        for i in range(self.prod_num):
            self.I_table.append(0)
        self.Y1_table = []
        self.Y2_table = []
        self.Y3_table = []
        self.Y4_table = []
        for i in range(self.prod_num):
            self.Y1_table.append(0)
            self.Y2_table.append(0)
            self.Y3_table.append(0)
            self.Y4_table.append(0)

    def choose_action(self, y, last_m, inv):
        self.set_y(y)
        self.set_state(last_m, inv)
        return self.choose_product_to_produce()

    def choose_product_to_produce(self):
        m = self.m - 1
        if self.m > 0 and self.I_table[m] < self.Y4_table[m]:
            return self.m
        elif self.get_z1() is not []:
            z = self.get_z1()
            _min, min_index = -1, -1
            for i in z:
                run_out_time = self.I_table[i] / self.u_table[i]
                if _min == -1 or _min > run_out_time:
                    _min = run_out_time
                    min_index = i
                elif _min == run_out_time:
                    if random.random() >= 0.5:
                        min_index = i
            return min_index + 1
        elif self.m > 0 and self.I_table[m] < self.Y2_table[m]:
            return self.m
        elif self.get_z3() is not []:
            z = self.get_z3()
            _min, min_index = -1, -1
            for i in z:
                run_out_time = self.I_table[i] / self.u_table[i]
                if _min == -1 or _min > run_out_time:
                    _min = run_out_time
                    min_index = i
                elif _min == run_out_time:
                    if random.random() >= 0.5:
                        min_index = i
            return min_index + 1
        else:
            return 0

    def get_z1(self):
        z = []
        for i in range(self.prod_num):
            if self.I_table[i] <= self.Y1_table[i]:
                z.append(i)
        return z

    def get_z3(self):
        z = []
        for i in range(self.prod_num):
            if self.I_table[i] <= self.Y3_table[i]:
                z.append(i)
        return z

    def set_state(self, m, I_table):
        self.m = m
        self.I_table = I_table

    def set_policy_parameters(self, y1, y2, y3, y4):
        self.Y1_table = y1
        self.Y2_table = y2
        self.Y3_table = y3
        self.Y4_table = y4

    def set_policy_parameters_by_tables(self,tables):
        self.Y1_table = tables[0]
        self.Y2_table = tables[1]
        self.Y3_table = tables[2]
        self.Y4_table = tables[3]

    def get_y_tables(self):
        y = [self.Y1_table, self.Y2_table, self.Y3_table,
             self.Y4_table]
        return y

    def get_y(self):
        y = []
        for i in range(self.prod_num):
            y.append(self.Y1_table[i])
            y.append(self.Y2_table[i])
            y.append(self.Y3_table[i])
            y.append(self.Y4_table[i])
        return y

    def set_y(self, y):
        assert len(y) == self.prod_num * 4
        for i in range(self.prod_num):
            self.Y1_table[i] = y[i * 4 + 0]
            self.Y2_table[i] = y[i * 4 + 1]
            self.Y3_table[i] = y[i * 4 + 2]
            self.Y4_table[i] = y[i * 4 + 3]


if __name__ == "__main__":
    from elsp_env_manager.elsp_env import ELSPEnv
    from elsp_env_manager.eval_env import EvalEnv
    env = ELSPEnv("../../../elsp_env_00014", "../../../")
    agent = BSP2(env.get_demand_u())

    def policy_maker(last_action, obs):
        y = [1.5522879261492974, 61.5522879261493, 1.5522879261492974, 61.5522879261493, 0.0, 120.0, 0.0, 60.0, 0.0,
             5.759476184616176, 0.0, 5.759476184616176, 0.0, 60.0, 0.0, 0.0, 25.158372645803503, 25.158372645803503,
             25.158372645803503, 25.158372645803503, 0.0, 0.0, 0.0, 0.0]
        inv = []
        for i in range(agent.prod_num):
            inv.append(obs[3 * (i + 1) + 2])
        return agent.choose_action(y, last_m=last_action, inv=inv)

    def eval_data_getter(last_inf):
        profit = last_inf["Profit"]
        return profit

    eval_env = EvalEnv(policy_maker=policy_maker,
                       envs=[ELSPEnv("../../../elsp_env_00014", "../../../") for _ in range(20)], max_process=20,
                       eval_data_getter=eval_data_getter)
    res = eval_env.eval(100)
    mean = sum(res) / len(res)
    print(mean)
    eval_env.destroy()


