import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 5, 6, 7])

plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.show()

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.0
d = 0.0
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
a = num / d
b = y_mean - a * x_mean
print("a=", a, "b=", b)

y_hat = a * x + b
plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()

x_preict = 2.5
y_predict = a * x_preict + b

plt.scatter(x, y)
plt.scatter(x_preict, y_predict)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()


# m面向对象


class SimpleLinearRegressionl:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据训练数据集和标签,训练SimpleLinearRegression模型的参数'''

        assert x_train.ndim == 1, \
            ''''''
        assert len(x_train) == len(y_train), \
            '''x_train的长度和y_train的长度应该是一致'''

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        '''对于给定的预测数集,'''
        assert x_predict.ndim == 1, \
            ''''''
        assert self.a_ is not None and self.b_ is not None, \
            '''预测之前必须有你和模型的参数'''

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''对单个样本尽心预测'''
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return '''SimpleLinearRegressionl'''


regl = SimpleLinearRegressionl()
regl.fit(x, y)

print(regl.a_)
print(regl.b_)


