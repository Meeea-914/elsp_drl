from multiprocessing.connection import Listener, Client, Connection
from multiprocessing import Process
from threading import Thread
from gym import Env
import gym
import time
import tqdm

ADDRESS = ('localhost', 6003)
AUTHKEY = b'123456'


class ClientEnv(object):

    def __init__(self, env: Env):
        self.done = True
        self.env = env
        self.obs = env.reset()
        self.conn = Client(ADDRESS, authkey=AUTHKEY)
        process = Process(target=self._main)
        process.start()

    def _main(self):
        while True:
            recv = self.conn.recv()
            if recv == "reset":
                obs = self.env.reset()
                self.conn.send(obs)
            elif recv == "destroy":
                # print("destroy")
                break
            else:
                res = self.env.step(recv)
                self.conn.send(res)


class EvalEnv(object):

    def __init__(self, policy_maker, envs, eval_data_getter, max_process=1):
        assert len(envs) >= max_process
        self.listener = Listener(ADDRESS, authkey=AUTHKEY)
        self.conn_list = []
        self.policy_maker = policy_maker
        self.eval_data_getter = eval_data_getter
        self.destroyed = False
        Thread(target=self._accept).start()
        self.client_env_list = []
        bar = tqdm.tqdm(total=max_process)
        bar.set_description("creating env")
        for i in range(max_process):
            client_env = ClientEnv(env=envs[i])
            bar.update(1)
        bar.close()
            # self.client_env_list.append(client_env)

    def _accept(self):
        while not self.destroyed:
            conn = self.listener.accept()
            self.conn_list.append(conn)
            # print("accepted:", len(self.conn_list))

    def eval(self, eval_times):
        bar = tqdm.tqdm(total=eval_times)
        bar.set_description("evaluating")
        eval_res_list = []
        obs_list = []
        action_list = []
        last_finished_num = 0
        for conn in self.conn_list:
            conn: Connection
            conn.send("reset")
            obs = conn.recv()
            obs_list.append(obs)
            action_list.append(0)
        while True:
            # print(1)
            if len(self.conn_list) > 0:
                p_n = len(self.conn_list)
                for i in range(p_n):
                    conn = self.conn_list[i]
                    obs = obs_list[i]
                    action = action_list[i]
                    action = self.policy_maker(action, obs)
                    action_list[i] = action
                    conn.send(action)
                for i in range(p_n):
                    conn = self.conn_list[i]
                    obs, reward, done, inf = conn.recv()
                    obs_list[i] = obs
                    if done:
                        eval_res_list.append(self.eval_data_getter(inf))
                        conn.send("reset")
                        obs = conn.recv()
                        obs_list[i] = obs
                        action_list[i] = action
                if len(eval_res_list) >= eval_times:
                    # print("Process:%d/%d" % (len(eval_res_list), eval_times))
                    bar.update(100 - last_finished_num)
                    bar.close()
                    return eval_res_list[0:100]
                else:
                    if last_finished_num < len(eval_res_list):
                        bar.update(len(eval_res_list) - last_finished_num)
                        last_finished_num = len(eval_res_list)
                        # print("Process:%d/%d" % (len(eval_res_list), eval_times))
            else:
                print("no conn")
                time.sleep(0.01)
            pass

    def destroy(self):
        p_n = len(self.conn_list)
        for i in range(p_n):
            conn = self.conn_list[i]
            conn.send("destroy")
        self.destroyed = True


if __name__ == "__main__":
    def env_maker():
        return gym.make("CartPole-v0")


    def policy_maker(last_action, obs):
        return 0


    def eval_data_getter(last_inf):
        return 0


    eval_env = EvalEnv(policy_maker, envs=[env_maker() for i in range(2)], max_process=2,
                       eval_data_getter=eval_data_getter)
    print(eval_env.eval(2))

