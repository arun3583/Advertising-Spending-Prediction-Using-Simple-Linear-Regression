import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegressionAssumptions:

    def __init__(self, model, X, y):
        self.X = X
        self.y = y
        self.model = model
        self.fitted_values = model.fittedvalues
        self.residuals = model.resid
        self.normalised_residuals = model.get_influence().resid_studentized_internal

    def linearity(self):
        """
        Linearity checking
        :return:
        """
        sns.regplot(x=self.X, y=self.y)
        plt.show()

    def independence(self):
        """
        checking independence of events
        :return:
        """
        durbin_watson = sm.stats.stattools.durbin_watson(self.residuals)
        print("durbin watson statistic:%s", durbin_watson)
        if durbin_watson < 1.5 or durbin_watson > 2.5:
            print("Errors are not Independent")
        else:
            print("Errors are Independent")

    def normality(self):
        """
        Checking Normality of error
        :return:
        """
        sns.displot(self.normalised_residuals)
        plt.title("Normality of Error")
        plt.show()

    def homoscedasticity_assumptions(self):
        """
        Checking if errors exhibit constant variance
        :return:
        """
        sns.regplot(x=self.fitted_values, y=self.residuals)
        plt.title("Homoscedasticity")
        plt.show()

    def run_all(self):
        self.linearity()
        self.independence()
        self.normality()
        self.homoscedasticity_assumptions()