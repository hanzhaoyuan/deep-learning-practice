import csv
from numpy import asarray, reshape


def load_data(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        head_data = next(reader)
        data = []
        for row in reader:
            data.append(row)
    data = asarray(data).astype(float)
    x = data[:, 0:-1]
    num_samples = data.shape[0]
    y = reshape(data[:, -1], (num_samples, 1))
    return (x, y)
