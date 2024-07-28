import statsmodels.api as sm


class SimpleLinearRegression:
    def __init__(self, x, y):
        """
        Initialize class with independent and dependent variables
        :param x:
        :param y:
        """
        self.x = x
        self.y = y
        self.x = sm.add_constant(self.x)

    def fit(self):
        """
        fit Linear regression with dependent and independent variables
        :return: model
        """
        model = sm.OLS(self.y, self.x).fit()
        return model

    def model_summary(self):
        """
        print the summary of linear regression model
        :return:
        """
        model = self.fit()
        # model.summary()
        return model
