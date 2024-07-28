import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataProcessing:
    def __init__(self, df):
        self.df = df

    def identify_outliers(self, data: pd.DataFrame):
        """
        Function to identify outliers in the data using a box plot
        :param data:
        :return:
        """
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_title("Box plot of Data")
        ax.set_ylabel("value")
        plt.show()

    def identify_outliers_zscore(self, data: pd.Series, threshold: float=3):
        """
        Function to identify the outliers using zscore
        :param data:
        :param threshold:
        :return:
        """
        mean = np.mean(data)
        std = np.std(data)
        z_score = (data - mean) / std
        outliers = data[np.abs(z_score) > threshold]
        return outliers
