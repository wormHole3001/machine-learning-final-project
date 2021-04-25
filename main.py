import pandas as pd
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k):
        self.k = k

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        distances = self.compute_distance(x_test)
        return self.predict_labels(distances)

    def compute_distance(self, x_test):
        # Naive
        num_test = x_test.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum((x_test[i, :] - self.x_train[j, :]) ** 2))

        return distances

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_predict = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_predict[i] = Counter(k_closest_classes).most_common(1)[0][0]
        return y_predict

    def max_heapify(self, x_test):
        distances = self.compute_distance(x_test)
        distances = distances.flatten()
        create = self.create_max_heap(distances)
        return create

    def max_heap(self, distances, num):
        left = self.left(num)
        right = self.right(num)

        if left < len(distances) and distances[left] > distances[num]:
            largest = left
        else:
            largest = num

        if right < len(distances) and distances[right] > distances[largest]:
            largest = right

        if largest != num:
            distances[num], distances[largest] = distances[largest], distances[num]
            self.max_heap(distances, largest)
        return distances

    def create_max_heap(self, distances):
        num = int((len(distances) // 2) - 1)
        for i in range(num, -1, -1):
            self.max_heap(distances, i)
        return distances

    @staticmethod
    def left(num):
        return 2 * num

    @staticmethod
    def right(num):
        return 2 * num + 1


if __name__ == "__main__":
    # Import Data
    train_df = pd.read_csv("TrainData.csv")
    test_df = pd.read_csv("TestData.csv")

    # Split Train Data
    train_x = train_df.iloc[:, 0:14]
    train_y = train_df.iloc[:, 14]

    # Split Test Data
    test_x = test_df.iloc[:, 0:14]
    test_y = test_df.iloc[:, 14]

    # Scale Data
    trainX = train_x.sample(n=1000)
    trainY = train_y.sample(n=1000)
    testX = test_x.sample(n=1000)
    testY = test_y.sample(n=1000)

    # KNN Without Max Heap
    KNN = KNN(k=25)
    KNN.train(train_x.values, test_y.values)
    y_pred = KNN.predict(train_x.values)

    print("K: 25")
    print("Training Data Sample Size:", len(train_df))
    print("Testing Data Sample Size:", len(test_df))
    print(f"Accuracy: {(sum(y_pred == testY.values) / testY.values.shape[0]) * 100}%")
