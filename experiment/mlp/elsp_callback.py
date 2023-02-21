import json
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elsp_env_manager.eval_env import EvalEnv
from elsp_env_manager.elsp_env import ELSPEnv

from experiment.env_create import env_creator
from experiment.mlp.evaluate_elsp import evaluate


def time_str():
    now = int(time.time())
    time_array = time.localtime(now)
    other_style_time = time.strftime("%Y--%m--%d %H-%M-%S", time_array)
    return other_style_time


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def policy_maker(self, last_action, obs):
        action = self.model.predict(np.array(obs), deterministic=self.eval_deterministic)
        a = int(action[0][0])
        return a

    def eval_data_getter(self, last_inf):
        profit = last_inf["Profit"]
        # print(profit)
        return profit

    def __init__(self, _env: SubprocVecEnv, _model: PPO, verbose=0, eval_th=32.5e4, n_eval_steps=15, no_eval=False, ):
        super(CustomCallback, self).__init__(verbose)
        self.env = _env
        self.model = _model
        self.org_data = []
        self.mean_profits = []
        self.max_profits = []
        self.all_ep_num = 0
        self.mean_action_num = []
        self.inf_buff = {}
        self.log_num = self.model.n_envs
        self.eval_th = eval_th
        self.n_eval_steps = n_eval_steps
        self.eval_cnt = 0
        self.eval_deterministic = False
        self.path = './result/env_path_%s/algorithm_%s/total_%e_n_steps_%e_batch_size_%e_n_epochs_%e_env_num_%d_lr_%e_gama_%.3f/time_%s' % \
                    (self.model.env_path, self.model.__class__.__name__+'+MLP', self.model.total_steps, self.model.n_steps, self.model.batch_size,
                     self.model.n_epochs, self.model.n_envs,
                     self.model.learning_rate, self.model.gamma, time_str())
        if not no_eval:
            self.eval_env = SubprocVecEnv([env_creator for i in range(60)])
            #self.eval_env = EvalEnv(envs=[ELSPEnv(self.model.env_root_path, self.model.cfg_root_path) for _ in range(55)], policy_maker=self.policy_maker, eval_data_getter=self.eval_data_getter, max_process=55)
            pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        b = self._on_rollout_end_dummy()
        for k, v in b.items():
            if k not in self.inf_buff:
                self.inf_buff[k] = []
            self.inf_buff[k].extend(v)
        print('num_timesteps', self.model.num_timesteps, 'time:', time.time() - self.model.start_time)
        if "Profit" not in self.inf_buff:
            self.inf_buff["Profit"] = []
        cnt = len(self.inf_buff["Profit"])
        if cnt < self.log_num:
            print(self.log_num - cnt)
            return
        else:
            ln = cnt - cnt % self.log_num
            labels = ["Profit"]
            for i in range(len(self.inf_buff)):
                labels.append(str(i))
            msg = {}
            for label in labels:
                if label in self.inf_buff:
                    msg[label] = self.inf_buff[label][:ln]
                    self.inf_buff[label] = self.inf_buff[label][ln:]
                else:
                    pass # msg[label] = [0]
        log_str = ""
        for i in range(len(msg)):
            if str(i) in msg:
                for j in range(len(self.mean_action_num), i + 1):
                    self.mean_action_num.append([])
                action_mean = np.mean(msg[str(i)])
                self.mean_action_num[i].append(action_mean)
                log_str += ("\t%s:%.2f" % (str(i), action_mean))
        print('\t\tall_ep_num:', self.all_ep_num,
              '\tep_num:', len(msg["Profit"]),
              '\tprofit mean:%.2f' % np.mean(msg["Profit"]),
              '\tprofit max:%.2f' % np.max(msg["Profit"]),
              '', log_str)
        self.org_data.extend(msg["Profit"])
        self.mean_profits.append(np.mean(msg["Profit"]))
        self.max_profits.append(np.max(msg["Profit"]))
        # print(pd.Series(self.mean_profits).rank().tolist())
        if self.mean_profits[-1] > self.eval_th:
            self.eval_cnt += 1
            print(self.eval_cnt)
            if self.eval_cnt >= self.n_eval_steps:
                self.eval_cnt = 0
                #p = self.path + '/temp.zip'
                #self.model.save(p)\
                self.eval_deterministic = False
                # profits = self.eval_env.eval(100)
                profit_list, storage_list, csl_list, backlog_penalties_list = evaluate(env=self.eval_env, eval_times=100,
                                                                                       deterministic=self.eval_deterministic, model=self.model)
                eval_mean_false = sum(profit_list)/len(profit_list)
                p = self.path + '/ef_%.2f_mean_p_%.2f_max_p%.2f.zip' % (eval_mean_false, self.mean_profits[-1], self.max_profits[-1])
                print("eval p:", p)
                self.model.save(p)
        if pd.Series(self.mean_profits).rank().tolist()[-1] >= (len(self.mean_profits) - 3):
            p = self.path + '/mean_p_%.2f_max_p%.2f.zip' % (self.mean_profits[-1], self.max_profits[-1])
            self.model.save(p)
            print(p)
        pass
        if len(self.mean_profits) % 10 == 1:
            self._on_training_end()

    def _on_rollout_end_single(self):
        """
        This event is triggered before updating the policy.
        """
        return

    def _on_rollout_end_dummy(self):
        """
        This event is triggered before updating the policy.
        """
        _infs = self.env.env_method('get_recorded_inf')
        # print(ep_num_infs)
        b = {}
        for _inf in _infs:
            for k, v in _inf.items():
                if k not in b:
                    b[k] = []
                b[k].extend(v)
        return b

    def _on_training_end(self) -> None:
        if len(self.max_profits) <= 0:
            return
        x = [i for i in range(len(self.max_profits))]
        plt.plot(x, self.max_profits, color="r", label='$max$')
        plt.plot(x, self.mean_profits, color="b", label='$mean$')
        plt.legend()
        plt.savefig(self.path + ('/learning curve %d.png' % self.model.num_timesteps))
        plt.clf()
        plt.plot([i for i in range(len(self.org_data))], self.org_data, color="r", label='$mean$')
        plt.savefig(self.path + ('/org learning curve %d.png' % self.model.num_timesteps))
        f = open(self.path+'/org_data.txt', 'w+')
        f.write(json.dumps(self.org_data))
        f.close()
        plt.clf()
        c = ['r', 'b', 'g', 'k', 'tan', 'y', 'c', 'm', 'coral', 'teal', 'pink', 'peru']
        x = [i for i in range(len(self.mean_action_num[0]))]
        plt.plot(x, self.mean_action_num[0], color=c[0], label='$idle$')
        plt.legend()
        plt.savefig(self.path + '/learning curve-idle.png')
        plt.clf()
        for i in range(1, len(self.mean_action_num)):
            plt.plot(x, self.mean_action_num[i], color=c[i], label=('$product:%d$' % i) if i > 0 else '$idle$')
        plt.legend()
        plt.savefig(self.path + '/learning curve-product len %d.png' % len(self.mean_action_num))
        plt.clf()
        p = self.path + '/finish'
        self.model.save(p)
        # plt.show()
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
