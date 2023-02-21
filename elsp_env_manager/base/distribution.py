from abc import ABC, abstractmethod
import math
import numpy as np


class UnaryPolynomial(object):

    def __init__(self, polynomial_coefficient):
        assert not isinstance(polynomial_coefficient, dict), polynomial_coefficient
        # 兼容直接给定常数项
        if not isinstance(polynomial_coefficient, list):
            polynomial_coefficient = [polynomial_coefficient]
        self.polynomial_coefficient = polynomial_coefficient

    def is_always_zero(self):
        zero = True
        for coe in self.polynomial_coefficient:
            zero &= (coe==0)
            if not zero:
                break
        return zero

    def calculate(self, x: float = 1):
        res = 0
        for i in range(len(self.polynomial_coefficient)):
            coe = self.polynomial_coefficient[i]
            if coe == 0:
                continue
            res += coe * (math.pow(x, i))
        return res


class Distribution(ABC):

    @abstractmethod
    def sample(self, x: float):
        pass

    @abstractmethod
    def is_always_zero(self, x):
        pass

    @staticmethod
    def load(json, name):
        if name not in json:
            return Distribution.load({"0": 0}, "0")
        elif json[name] == "up_to_product":
            return json[name]
        json = json[name]
        return Distribution.load_2(json)

    @staticmethod
    def load_2(json):
        if isinstance(json, list) and isinstance(json[0], str):
            if json[0] == 'n':
                return NormalDistribution(json[1], json[2], False if len(json)<4 else json[3])
            else:
                assert NotImplementedError
        elif isinstance(json, dict):
            return SubsectionDistribution(distribution_dict=json)
        else:
            return Deterministic(json)


class SubsectionDistribution(Distribution):

    def __init__(self, distribution_dict:dict):
        self.distribution_dict = distribution_dict
        self.distribution_dict_buff = {}
        for k, v in self.distribution_dict.items():
            self.distribution_dict_buff[k] = Distribution.load_2(v)
        if 'default' in self.distribution_dict_buff:
            self.default = self.distribution_dict_buff['default']
        else:
            raise NotImplementedError

    def __getitem__(self, x) -> Distribution:
        dist = None
        for k in list(self.distribution_dict.keys()):
            if k == 'default':
                continue
            else:
                try:
                    if eval(k):
                        dist = self.distribution_dict_buff[k]
                except Exception:
                    pass
        if dist is None:
            dist = self.default
        return dist

    def sample(self, x: float):
        return self[x].sample(x)

    def is_always_zero(self, x):
        return self[x].is_always_zero(x)


class UniformDistribution(Distribution):

    def __init__(self, a_polynomial_coefficient: list, b_polynomial_coefficient: list):
        self.a: UnaryPolynomial = UnaryPolynomial(a_polynomial_coefficient)
        self.b: UnaryPolynomial = UnaryPolynomial(b_polynomial_coefficient)

    def sample(self, x: float = 1):
        return np.random.uniform(self.a.calculate(x), self.b.calculate(x))

    def is_always_zero(self, x):
        return self.a.calculate(x) == 0 and self.b.calculate() == 0


class NormalDistribution(Distribution):

    def __init__(self, mean_polynomial_coefficient: list, std_polynomial_coefficient: list, std_multiply_mean: bool=False):
        self.mean: UnaryPolynomial = UnaryPolynomial(mean_polynomial_coefficient)
        self.std: UnaryPolynomial = UnaryPolynomial(std_polynomial_coefficient)
        self.std_multiply_mean = std_multiply_mean

    def sample(self, x: float = 1):
        mean = self.mean.calculate(x)
        return np.random.normal(mean, self.std.calculate(x) * (mean if self.std_multiply_mean else 1))

    def is_always_zero(self, x):
        return self.mean.calculate(x) == 0 and self.std.calculate() == 0


class Deterministic(Distribution):

    def __init__(self, base_coefficient: list):
        self.base: UnaryPolynomial = UnaryPolynomial(base_coefficient)

    def sample(self, x: float = 1):
        return self.base.calculate(x)

    def is_always_zero(self, x):
        return self.sample(x) == 0


if __name__ == "__main__":
    from utils.TimeCalculator import time_calculator
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    """
    一、 UnaryPolynomial 测试
    1. calculate test
    print(UnaryPolynomial([1, 1, 1]).calculate(2))  # 1 + x + x^2|x=2 =7
    print(UnaryPolynomial([1, 1, 1]).calculate(3))  # 1 + x + x^2|x=3 =13
    print(UnaryPolynomial([1, 2, 1]).calculate(3))  # 1 + 2x + x^2|x=3 =16
    print(UnaryPolynomial([1, 2, 3]).calculate(3))  # 1 + 2x + 3x^2|x=3 =34
    print(UnaryPolynomial([1, 2, 3, 1]).calculate(3))  # 1 + 2x + 3x^2|x=3 =61
    print(UnaryPolynomial([1, 2, 3, 2]).calculate(3))  # 1 + 2x + 3x^2|x=3 =88
    2. efficient test
    time_calculator.st("test")
    time_calculator.st("my")
    for i in range(1000000):
        print(UnaryPolynomial([1, 2, 3, 2, 2, 2, 2, 2]).calculate(3))  # 1 + 2x + 3x^2|x=3 =88
    time_calculator.ed("my")

    time_calculator.st("dir")
    for i in range(1000000):
        # UnaryPolynomial([1, 2, 3, 2]).calculate(3)  # 1 + 2x + 3x^2|x=3 =88
        # y = 1 + 2 * 3. + 3 * math.pow(3., 2) + 2 * math.pow(3., 3)  # almost 4 times
        # y = 1 + 2 * 3. + 3 * math.pow(3., 2) + 2 * math.pow(3., 3) + 2 * math.pow(3., 4)  # almost 3 times
        y = 1 + 2 * 3. + 3 * math.pow(3., 2) + 2 * math.pow(3., 3) + 2 * math.pow(3., 4) \
            + 2 * math.pow(3., 5) + 2 * math.pow(3., 6) + 2 * math.pow(3., 7)  # almost 2 times
    time_calculator.ed("dir")

    time_calculator.ed("test")
    """
    """
    二、 Distribution 测试
    #d = UniformDistribution([0, 1], [0, 2])
    d = NormalDistribution([1,1,5,0,0,0,0,2], [1])
    s = []
    eval_times = 20000
    n = eval_times // 40
    bar = tqdm(eval_times)
    for i in range(eval_times):
        s.append(d.sample(2))
        if i % 50 == 49:
            bar.update(50)
    p, x = np.histogram(s, bins=n)
    x = x[:-1] + (x[1] - x[0]) / 2
    f = UnivariateSpline(x, p, s=n)
    bar.close()
    plt.plot(x, f(x))
    plt.show()
    """
