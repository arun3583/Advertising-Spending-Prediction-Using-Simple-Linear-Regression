from src.data_ingest import DataIngestion
from src.data_processing import DataProcessing
from src.build_model import SimpleLinearRegression
from src.asuumptions_test import LinearRegressionAssumptions

if __name__ == "__main__":
    # Initialize the data ingestion class with file path
    data_ingest = DataIngestion("./data/advertising.csv")

    # Load training and testing data
    x, y, df = data_ingest.get_train_test_data()
    df.to_csv("./data/simple_df.csv", index=False)

    # Initialize Data Processing class with data
    data_processing = DataProcessing(df)
    # data_processing.identify_outliers(df["TV"])
    outliers = data_processing.identify_outliers_zscore(df["TV"])
    # # Initialize Model building class with input and output data
    lr_model = SimpleLinearRegression(x, y)
    model = lr_model.model_summary()

    # Assumptions
    assumption_test = LinearRegressionAssumptions(model, x, y)
    assumption_test.run_all()
