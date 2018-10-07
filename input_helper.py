import numpy as np

from data_preprocessor import DataPreprocessor

class InputHelper:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()

    def get_dataset(self, input_files, bar_length, assets, time_steps, percent_dev, shuffle=True):
        data = self.data_preprocessor.read(input_files)
        data = self.data_preprocessor.preprocess(data, bar_length)
        data = self.data_preprocessor.select_features(data)
        data, mean, std = self.data_preprocessor.normalize(data)
        np.save("./save/mean-std", np.asarray([mean, std]))
        print(data.shape)
        dataset = self.generate_dataset(data, assets, time_steps)
        return self.validate(dataset, percent_dev)

    def batch_iter(self, data, batch_size, num_epoches, shuffle=True):
        data = np.asarray(data)
        size = len(data)
        num_batches_per_epoch = size // batch_size if size % batch_size == 0 else size // batch_size + 1
        for epoch in range(num_epoches):
            shuffled_data = data
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(size))
                shuffled_data = data[shuffle_indices]
            for batch_no in range(num_batches_per_epoch):
                start_index = batch_no * batch_size
                end_index = min((batch_no + 1) * batch_size, size)
                yield shuffled_data[start_index: end_index]
    
    def generate_dataset(self, data, assets, time_steps):
        X = []
        Y = []
        for i in range(data.shape[0] - time_steps):
            X.append(np.reshape(data[i: i + time_steps, assets], [time_steps, -1]))
            Y.append(data[i + time_steps, assets, 1])
        return np.asarray(X), np.asarray(Y)

    def validate(self, dataset, percent_dev):
        X, Y = dataset
        np.random.seed(5)
        shuffle_indices = np.random.permutation(X.shape[0])
        shuffled_X = X[shuffle_indices]
        shuffled_Y = Y[shuffle_indices]
        dev_index = -1 * X.shape[0] * percent_dev // 100
        X_train, X_dev = shuffled_X[: dev_index], shuffled_X[dev_index: ]
        Y_train, Y_dev = shuffled_Y[: dev_index], shuffled_Y[dev_index: ]
        train_set = X_train, Y_train
        dev_set = X_dev, Y_dev
        return train_set, dev_set

if __name__ == "__main__":
    format2_path = "data_format2_20180916_20180923.h5"
    input_helper = InputHelper()
    # bar_length, assets, time_steps, dev_percent
    train_set, dev_set = input_helper.get_dataset(format2_path, 30, [0], 10, 20)
    print(train_set[0].shape, train_set[1].shape)
    print(dev_set[0].shape, dev_set[1].shape)

