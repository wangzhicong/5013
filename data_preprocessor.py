import h5py
import numpy as np

class DataPreprocessor:
    def read(self, input_files):
        data = []
        for input_file in input_files:
            h5data = h5py.File(input_file)
            for key in list(h5data.keys()):
                data.append(h5data[key][:,:])
            print("read {}".format(input_file))
        return np.asarray(data)

    def generate_bar(self, data):
        close = data[data.shape[0] - 1, :, 0]
        high = np.max(data[:, :, 1], axis=0)
        low = np.min(data[:, :, 2], axis=0)
        open_price = data[0, :, 3]
        avg_volume = np.mean(data[:, :, 4], axis=0)
        return np.transpose(np.asarray([close, high, low, open_price, avg_volume]), [1, 0])

    def preprocess(self, data, bar_length):
        reduced_data = []
        for i in range((data.shape[0] // bar_length)):
            reduced_data.append(self.generate_bar(data[i * bar_length: min((i + 1) * bar_length, data.shape[0])]))
        return np.asarray(reduced_data)

    def select_features(self, data):
        increasing_rate = (data[:, :, 0] - data[:, :, 3]) * 100 / data[:, :, 3]
        return np.transpose(np.asarray([data[:, :, 3], increasing_rate, data[:, :, 4]]), [1, 2, 0])

    def normalize(self, data):
        shape = data.shape
        data = data.reshape(shape[0], -1)
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        data = (data - mean) / std
        return data.reshape(shape), mean, std
