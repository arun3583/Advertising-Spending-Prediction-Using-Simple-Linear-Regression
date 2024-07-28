import pandas as pd

class DataIngestion:
    """
    Class for Ingesting Advertising and sales data from a csv file.
    """
    def __init__(self, file_path):
        """
        Initialize class with the filepath
        :param file_path: str, path to csv file
        """
        self.file_path = file_path

    def load_data(self):
        """
        Load data from csv file
        :return: pandas DataFrame of loaded csv
        """
        data = pd.read_csv(self.file_path)
        return data

    def get_train_test_data(self):
        """
        Get Features (x) and target (y) from data
        :return:tuple, feature (x) and target variable(y)
        """
        data = self.load_data()
        x = data[["TV"]]
        y = data["Sales"]
        df = pd.concat([x, y], axis=1)
        return x, y, df