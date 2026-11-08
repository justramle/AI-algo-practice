import numpy as np

class Perceptron:
    # function declare:
    def __init__(self, n_features):
        self.weight =  np.zeros(n_features)
        self.bias = 0

    # classify the prediction 1 and -1
    # self (w,b) and input
    # first weight
    def prediction(self, x):
        # weighted sum or w.f
        activation = np.dot(self.weight, x) + self.bias
        if (activation >= 0):
            y_pred = +1
        else:
            y_pred = -1
        return y_pred

    # training where perceptron learn weight and update
    def fit(self, x, y, epoch=10):
        for epoch in range(epoch):
            count_mistake = 0 # reset
            # loop thr train and compare y_pred vs y* and updatae weight
            for x_i, y_i in zip(x, y):
                score = self.prediction(x_i)
                if score != y_i:
                    self.weight = self.weight + y_i * x_i
                    self.bias = self.bias + y_i 
                    count_mistake += 1
            print(f"epoch {epoch + 1}: {count_mistake} mistakes")
            if count_mistake == 0: # no mistake, stop the train
                break;

# 3 input /w 2 features
X = np.array([[1,-2], [1,1], [-1,-2]])
Y = np.array([1,-1,-1])

model = Perceptron(2)
model.fit(X,Y)
for x_i in X:
    model.prediction(x_i)
    print(x_i, "-> y_pred: ",  model.prediction(x_i))
print("w: ", model.weight, " and b: ", model.bias)


