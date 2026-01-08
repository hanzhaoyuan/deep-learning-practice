import csv
from numpy import asarray


def load_1d_csv_data(file_name):
    """load 1D CSV data transfrom to NDArray"""
    with open(file_name) as file:
        reader = csv.reader(file)
        header = next(reader)

        data = []
        for row in reader:
            data.append(row)
        array = asarray(data).astype('float')
    x = array[:, 0]
    y = array[:, 1]
    return (x, y)
