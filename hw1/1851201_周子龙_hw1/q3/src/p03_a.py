import utils
import pandas as pd


def handleMissingValue(data):
    data.fillna(method='ffill',axis=0)
    return data


def main(data_path, ):
    data = utils.load_dataset(data_path)
    data = pd.DataFrame(data)
    data.fillna(method='ffill', axis=0)



if __name__ == '__main__':
    main("../data/missingValue.txt")
