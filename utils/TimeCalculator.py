import time
import datetime

top_task = None


class TaskTime(object):

    def __init__(self, name, as_top_task=False):
        global top_task
        self.name = name
        self.time_span_list = []
        self.start_time = -1
        self.end_time = -1
        self.total = 0
        self.already = 0
        self.log = True
        if as_top_task or top_task is None:
            top_task = self

    def start(self):
        assert (self.start_time == -1), self.name
        self.start_time = time.time()

    def end(self):
        assert (self.end_time == -1), self.name
        self.end_time = time.time()
        self.task_exec_once()

    def task_exec_once(self):
        global top_task
        delta = self.end_time - self.start_time
        self.time_span_list.append([self.start_time, self.end_time, delta])
        self.total += delta
        self.end_time = self.start_time = -1
        tcm = self.total / len(self.time_span_list)
        top_task.refresh()
        if self.log:
            print('\t\t\t\t\t task:%s executed %d times,tct:%e s,tcm:%.e s, tpt:%.4f%%,tpm:%.4f%%,top:%s' %
                  (self.name, len(self.time_span_list), self.total, tcm, self.total * 100. / top_task.already,
                   tcm * 100. / top_task.already, top_task.name))

    def refresh(self):
        self.already = time.time() - self.start_time
        if self.already == 0:
            print("warning: already equals zero")
            self.already = 1


class TimeCalculator(object):

    def __init__(self):
        self.task_time_dict = {}
        self.log = True
        pass

    def st(self, name, as_top_task=False):
        if name not in self.task_time_dict:
            self.task_time_dict[name] = TaskTime(name, as_top_task)
            self.task_time_dict[name].log = self.log
        tt: TaskTime = self.task_time_dict[name]
        tt.start()

    def ed(self, name):
        assert name in self.task_time_dict
        tt: TaskTime = self.task_time_dict[name]
        tt.end()

    def set_log(self, log=True):
        self.log = log
        for n, t in self.task_time_dict.items():
            t: TaskTime
            t.log = log


time_calculator = TimeCalculator()
if __name__ == '__main__':
    time_calculator.st('loop')
    for j in range(1111):
        time_calculator.st('loop1')
        for i in range(100000):
            # print(i)
            pass
        time_calculator.ed('loop1')
        time_calculator.st('loop2')
        for i in range(100000):
            # print(i)
            pass
        time_calculator.ed('loop2')
    time_calculator.ed('loop')
