import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from random import shuffle


class Perceptron1:
    # Section 1: define the Perceptron1 as a prototype model
    def __init__(self, alpha, max_iter):
        # initialise the object with preset parameters
        self.w = []
        self.alpha = alpha
        self.max_iter = max_iter

    def inner_product(self, xi):
        # Section 2: define xw() function to determine <x,w>
        xw: float = 0
        for a in range(len(xi)):
            xw += xi[a] * self.w[a]
        return xw

    def training(self, x_training_sample, y_training_sample):
        # Section 3: input x vector where xi is a feature, y is the label, and achieve the optimise w vector
        self.w = [float(0) for _ in range(len(x_training_sample[0]))]
        gd_iter = 0
        for k in range(self.max_iter):
            no_wrong_classify = True
            for c in range(len(x_training_sample)):
                if y_training_sample[c] * self.inner_product(x_training_sample[c]) <= 0:
                    no_wrong_classify = False
                    for e in range(len(self.w)):
                        self.w[e] = self.w[e] + self.alpha * y_training_sample[c] * x_training_sample[c][e]
            gd_iter += 1
            if no_wrong_classify:
                break

    def sign(self, x_test_sample):
        # Section 4: define the sign function as the predictor used for testing
        if self.inner_product(x_test_sample) > 0:
            return 1
        else:
            return -1


class Perceptron2:

    def __init__(self, alpha, max_iter):
        # Section 5: initialise the object with preset parameters
        self.w = []
        self.b = 0
        self.alpha = alpha
        self.max_iter = max_iter

    def inner_product(self, xi):
        xw: float = 0
        for a in range(len(xi)):
            xw += xi[a] * self.w[a]
        return xw

    def training(self, x_training_sample, y_training_sample):
        self.w = [float(0) for _ in range(len(x_training_sample[0]))]
        gd_iter = 0
        for k in range(self.max_iter):
            no_wrong_classify = True
            for c in range(len(x_training_sample)):
                if y_training_sample[c] * (self.inner_product(x_training_sample[c]) + self.b) <= 0:
                    no_wrong_classify = False
                    # Section 6: update bias
                    self.b = self.b + self.alpha * y_training_sample[c]
                    for e in range(len(self.w)):
                        self.w[e] = self.w[e] + self.alpha * y_training_sample[c] * x_training_sample[c][e]
            gd_iter += 1
            if no_wrong_classify:
                break

    def sign(self, x_test_sample):
        if self.inner_product(x_test_sample) + self.b > 0:
            return 1
        else:
            return -1


def raw_data():
    # obtain data from provided dataset file and make it a list containing all subjects
    f = open("diabetes_scale.txt", encoding="utf-8")
    rawdata = []
    for line in f:
        rawdata.append(line.split())
    f.close()
    return rawdata


def add_dimension(feature):
    # add x0 dimension for features of each sample, where x0 = 1
    for a in range(len(feature)):
        feature[a].append(1.0)


if '__main__' == __name__:
    # Run the Perceptron model and test
    # Load the dataset "diabetes_scale" and define x as features, y as labels and sample to draw from dataset
    raw = raw_data()
    x = []
    y = []
    sample = []

    # Run once with one sample
    for i in range(len(raw)):
        for j in range(1, len(raw[i])):
            sample.append(float(raw[i][j][2:]))
        if len(sample) == 8:
            x.append(sample)
            y.append(int(raw[i][0]))
        sample = []

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    y_predict_1 = []
    y_predict_2 = []
    for i in range(608):
        if len(x[i]) == 8:
            x_train.append(x[i])
            y_train.append(y[i])
    for i in range(151):
        if len(x[i+608]):
            x_test.append(x[i+608])
            y_test.append(y[i+608])

    add_dimension(x_train)
    add_dimension(x_test)

    p = Perceptron1(1, 100)
    p.training(x_train, y_train)

    q = Perceptron2(1, 100)
    q.training(x_train, y_train)

    for i in range(len(x_test)):
        y_predict_1.append(p.sign(x_test[i]))

    for i in range(len(x_test)):
        y_predict_2.append(q.sign(x_test[i]))

    error_1 = 0
    for i in range(len(y_test)):
        if y_test[i] != y_predict_1[i]:
            error_1 += 1

    error_2 = 0
    for i in range(len(y_test)):
        if y_test[i] != y_predict_2[i]:
            error_2 += 1

    print('One time running with fixed sample: alpha=1, max_iter=100')
    print('Perceptron I wrong classified:', error_1)
    print('Perceptron I F1 score is:', f1_score(y_test, y_predict_1))
    print('Perceptron I Accuracy score is:', accuracy_score(y_test, y_predict_1))

    print('Perceptron II wrong classified', error_2)
    print('Perceptron II F1 score is:', f1_score(y_test, y_predict_2))
    print('Perceptron II Accuracy score is:', accuracy_score(y_test, y_predict_2))
    print('\n')

    # Find the best iteration times by generating a graph
    iteration = 1
    # Define list to store accuracy and f1 score for p1 and p2 objects
    p1_accuracy = []
    p1_f1_score = []
    p2_accuracy = []
    p2_f1_score = []
    # iter for 100 times
    while iteration <= 100:
        y_predict_1 = []
        y_predict_2 = []
        p = Perceptron1(0.01, iteration)
        p.training(x_train, y_train)

        q = Perceptron2(0.01, iteration)
        q.training(x_train, y_train)

        for i in range(len(x_test)):
            y_predict_1.append(p.sign(x_test[i]))

        for i in range(len(x_test)):
            y_predict_2.append(q.sign(x_test[i]))

        p1_accuracy.append(accuracy_score(y_test, y_predict_1))
        p2_accuracy.append(accuracy_score(y_test, y_predict_2))

        p1_f1_score.append(f1_score(y_test, y_predict_1))
        p2_f1_score.append(f1_score(y_test, y_predict_2))

        iteration = iteration + 1

    print('Perceptron I accuracy:', p1_accuracy)
    print('Perceptron II accuracy:', p2_accuracy)

    print('Perceptron I f1:', p1_f1_score)
    print('Perceptron II f1:', p2_f1_score)

    # Make iteration times 1- 100 as the x-axis
    iteration_times = []
    for i in range(100):
        iteration_times.append(i)

    # plot chart
    linechart = plt.figure()
    up = plt.subplot(3, 1, 1)
    plt.xlabel('Iteration times')
    plt.ylabel('Predicted accuracy')
    plt.title('Accuracy vs. iteration: alpha=0.01')
    plt.plot(iteration_times, p1_accuracy,
             iteration_times, p2_accuracy)
    # plt.plot(iteration_times, p1_accuracy)
    down = plt.subplot(3, 1, 3)
    plt.xlabel('Iteration times')
    plt.ylabel('F1 score')
    plt.title('F1 score vs. iteration: alpha=0.01')
    plt.plot(iteration_times, p1_f1_score,
             iteration_times, p2_f1_score)
    # plt.plot(iteration_times, p1_f1_score)
    plt.show()
    # General performance with 1000 different samples input
    p1_F1 = []
    p1_accuracy = []
    p2_F1 = []
    p2_accuracy = []
    for counter in range(1000):
        x = []
        y = []
        sample = []
        shuffle(raw)
        for i in range(len(raw)):
            for j in range(1, len(raw[i])):
                sample.append(float(raw[i][j][2:]))
            if len(sample) == 8:
                x.append(sample)
                y.append(int(raw[i][0]))
            sample = []

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        y_predict_1 = []
        y_predict_2 = []
        for i in range(608):
            if len(x[i]) == 8:
                x_train.append(x[i])
                y_train.append(y[i])
        for i in range(151):
            if len(x[i + 608]):
                x_test.append(x[i + 608])
                y_test.append(y[i + 608])

        add_dimension(x_train)
        add_dimension(x_test)

        y_predict_1 = []
        y_predict_2 = []
        p = Perceptron1(1, 100)
        p.training(x_train, y_train)

        q = Perceptron2(1, 100)
        q.training(x_train, y_train)

        for i in range(len(x_test)):
            y_predict_1.append(p.sign(x_test[i]))

        for i in range(len(x_test)):
            y_predict_2.append(q.sign(x_test[i]))

        p1_accuracy.append(accuracy_score(y_test, y_predict_1))
        p1_F1.append(f1_score(y_test, y_predict_1))
        p2_accuracy.append(accuracy_score(y_test, y_predict_2))
        p2_F1.append(f1_score(y_test, y_predict_2))

    print('1,000 times running with random training and testing sample: alpha=1, max_iter=100 ')
    print('Perceptron I: accuracy mean', np.mean(p1_accuracy), 'std', np.std(p1_accuracy))
    print('Perceptron I: F1 score mean', np.mean(p1_F1), 'std', np.std(p1_F1))
    print('Perceptron II: accuracy mean', np.mean(p2_accuracy), 'std', np.std(p2_accuracy))
    print('Perceptron II: F1 score mean', np.mean(p2_F1), 'std', np.std(p2_F1))
