from datetime import timedelta
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
import warnings

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../classification'))

class GenerateData:
    """Class is used to download or load data, that includes the historical beta for specific stocks

    """

    def get_msci_world_symbols(self, number: int = 100):
        """gets the first x stocks from the msci world

        Args:
            number: int: number of stock-symbols that should be returned

        Returns: list with stock symbols

        """
        data_symbols = pd.read_csv(r'../classification/stocks_msci_world.csv')
        symbols = data_symbols['Symbol']
        symbol_list = []
        for x in symbols:
            symbol_list.append(x)
        return symbol_list[:number]

    def download_data(self, symbol_list: list):
        """Downloads the stock course of all symbols from yahoo finance

        Args:
            symbol_list: list with all symbols

        Returns: tuple: (time of the course, course)

        """
        ticker_names = " ".join(symbol_list)
        # using s&p 500 as comparison index
        ticker_names_with_sandp = ticker_names + " ^GSPC"
        data = yf.download(ticker_names_with_sandp, interval='1d')
        # downloading close prices for each day
        data = data['Adj Close']
        # calculating percentage change in close prices (= Rendite) in order to use it for beta calculation
        data = data.pct_change()
        # drop empty columns of dataframe
        data = data.dropna(axis=1, how='all')
        data = data.dropna(axis=0, how='all')
        time = data.index
        return time, data

    def calculate_beta(self, time, data):
        """Calculates the beta over an interval of 30 days for each stock and day

        Args:
            time: Series attribute including each day of the stock-data-download
            data: end course of specific stocks

        Returns: df with time as index and all companies as columns

        """
        beta_df = pd.DataFrame()
        for j in time:
            data_temp = data[data.index < j]
            time_end = j - timedelta(days=30)
            data_temp = data_temp[data_temp.index > time_end]
            cov = data_temp.cov()
            beta = cov / data_temp['^GSPC'].var()
            beta = beta.iloc[:, -1]
            beta_df = beta_df.append(beta)
        beta_df.index = time
        return beta_df

    def get_beta_for_stocks(self, stock_symbols_for_calculation: list = None, save: bool = True):
        """returns a dataframe with the historical betas for a given list of stock symbols and save it optionally in a
        json file

        Args:
            stock_symbols_for_calculation: optional: list for calculation
            save: bool

        Returns: dataframe with historical betas

        """
        if stock_symbols_for_calculation is None:
            stock_symbols_for_calculation = self.get_msci_world_symbols()
        stock_symbols_for_calculation = list(stock_symbols_for_calculation)
        time, data = self.download_data(stock_symbols_for_calculation)
        data = self.calculate_beta(time, data)
        data = data.drop('^GSPC', axis=1, errors='ignore')
        data = data.set_index(time)
        if save:
            with open("data_msci_world_betas_30day_interval.json", "w") as outfile:
                data_as_json = data.copy().to_json()
                json.dump(data_as_json, outfile)
        return data

    def load_stock_data(self, filename="data_msci_world_betas_30day_interval.json"):
        """
        reads json file with information about the historical beta for better performance

        Returns: pandas.DataFrame with time as the index and the historical beta of each stock + s&p 500 for
                comparison

        """
        with open(filename) as outfile:
            all_stocks0 = json.load(outfile)
            df = pd.read_json(all_stocks0)

        # Preprocessing of data
        df = df.drop("^GSPC", axis=1)
        df.index = pd.to_datetime(df.index, unit="ms")
        print(df.tail())
        return df


warnings.filterwarnings("ignore")


class CompanyBetaPredictor:
    """Class for train model to forecast beta, evaluate the model or predict future beta for a stock

    """

    def __init__(self, symbol: str, df: pd.DataFrame, existing_model=False) -> object:
        """
        Initialize object of class company_beta_predictor in order to predict or evaluate the beta of one specific
        company

        Args:
            symbol: (str) Symbol of the company
        """
        self.df = df
        self.symbol = symbol
        self.path = f"..//classification//trained_models//{self.symbol}"
        self.figurepath = f"..//classification//trained_models//{self.symbol}//figures//"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.figurepath):
            os.makedirs(self.figurepath)
        self.existing_model = existing_model
        if existing_model:
            try:
                self.model = load_model(f"{self.path}//testRun//model.pkl")
            except:
                print(f"Can't find model for path: {self.path}")
                self.model = "EMPTY"
        else:
            self.model = None

    def print_beta_history(self):
        """
        prints historical betas of the company

        Returns: plot with historical betas

        """
        symbol = self.symbol
        plt.plot(self.df[symbol])
        plt.xlabel("Time")
        plt.ylabel("Beta")
        plt.title(f"Betas des Unternehmen {symbol} über die Vergangenheit")
        # plt.show()
        plt.savefig(f"{self.figurepath}//historical_betas.png")
        plt.close()

    def train_models(self, test_run: bool = True):
        """
        Train model of the company for predicting prospective betas, safe model as pkl file

        Args:
            test_run: (bool)
                if true: execute one test run and takes the beta of the last 60 days working days for testing
                else: predicts the beta for the next 60 days

        Returns:

        """

        data = self.df.filter([self.symbol])
        data = data.dropna()
        self.data = data
        dataset = data.values
        self.dataset = dataset

        # Get the number of rows to train the model on
        if test_run:
            training_data_len = int(np.ceil(len(dataset) * .95))  # len(dataset) - 60 #
        else:
            training_data_len = int(np.ceil(len(dataset) * 1))  # - 60
            #             training_data_len = int(len(dataset))

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(dataset)

        # Create the training data set
        # Create the scaled training data set
        self.train_data = self.scaled_data[0:int(training_data_len), :]
        self.train_data_len = training_data_len
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []
        # x can be set to any wanted value, despite on how many days should be forecasted
        x = 60
        for i in range(360, len(self.train_data) - x):
            # takes last 60 days for training and the actual day as label
            x_train.append(self.train_data[i - 360:i, 0])
            y_train.append(sum(self.train_data[i:i + x, 0]) / x)

        # Convert the x_train and y_train to numpy arrays
        x_train, self.y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        if self.existing_model:
            pass
        else:
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(x_train, self.y_train, batch_size=1, epochs=1)

            if test_run:
                path1 = f"{self.path}//testRun/model.pkl"
            else:
                path1 = f"{self.path}//model.pkl"
            model.save(path1)
            self.model = model

    def prediction_new(self) -> list:
        self.predictions = []
        for i in range(60):
            temp = 360 - i
            test_data = self.scaled_data[self.train_data_len - temp:, 0]
            self.predictions
            test_data = np.append(test_data, self.predictions)
            # Convert the data to a numpy array
            x_test = np.array(test_data)

            # Reshape the data
            x_test = np.reshape(x_test, (1, x_test.shape[0], 1))

            # Get the models predicted price values
            predictions = self.model.predict(x_test)
            # print(predictions[0])
            self.predictions.extend(predictions[0])
        self.predictions = self.scaler.inverse_transform(np.array(self.predictions).reshape(-1, 1))
        return self.predictions

    ##################
    # # Evaluation # #
    ##################
    def evaluation(self) -> tuple:
        """
        Create the testing data set
        Create a new array containing scaled values

        Returns: tuple including predictions and y_test with real values

        """
        test_data = self.scaled_data[self.train_data_len - 360:, :]
        # Create the data sets x_test and y_test
        x_test = []
        y_test = self.dataset[self.train_data_len:, :]
        for i in range(360, len(test_data)):
            x_test.append(test_data[i - 360:i, 0])
        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get the models predicted price values
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        self.predictions = predictions
        # Get the root mean squared error (RMSE)
        # rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        # mse = np.mean(((predictions - y_test) ** 2))

        return predictions, y_test

    def print_model_evaluation(self):
        """
        predicting F-Score from model and compares it to a naive one

        Returns:

        """
        predictions, y_test = self.evaluation()
        y_naive = [[self.y_train[-1]] for i in range(len(y_test))]
        F_score, rsme, output_string = self.model_F_score(predictions, y_test, title="Predictions")
        F_score_naive, rsme_naive, output_string_naive = self.model_F_score(y_naive, y_test, title="Naive Forecasting")

        with open(f"{self.figurepath}//test_information", "w") as outfile:
            outfile.write(output_string + '\n' + output_string_naive)

    def model_F_score(self, newp, newy_test, title: str = None) -> tuple:
        """
        calculates F-score and rmse (root mean squared error) of a model and the correlating test data

        Args:
            newp: predicted data of a model
            newy_test: real data
            title: title of the diagram

        Returns: (tuple) Fscore of the model and rmse of the model + plot including predictions and actual data

        """
        newp = np.array(newp)
        newy_test = np.array(newy_test)
        TP = 0
        FN = 0
        FP = 0
        for i in range(len(newp) - 1):
            test_prof = newy_test[i + 1] - newy_test[i]
            p_prof = newp[i + 1] - newp[i]

            # logical functionality: if the model predicted, that the beta will increase the next day and it increases
            # than it will be count as true positive
            # fn: beta increases on beta but decrease on predicted data
            if ((test_prof >= 0) and (p_prof >= 0)):
                TP = TP + 1
            if ((test_prof >= 0) and (p_prof < 0)):
                FN = FN + 1
            if ((test_prof < 0) and (p_prof >= 0)):
                FP = FP + 1

        Precision = float(TP) / float(TP + FP)
        Recall = float(TP) / float(TP + FN)

        Fscore = 2.0 * Precision * Recall / (Precision + Recall)

        if title is not None:
            print(title)

        output_string = f'Title: {title}' + f'\n\ Precision: {Precision}' + '\n' + 'Recall' + str(Recall) + '\n' + \
                        'classification F score: %.5f' % (Fscore) + '\n'
        print('classification F score: %.5f' % (Fscore))

        rmse = np.sqrt(np.mean(((newp - newy_test) ** 2)))

        import matplotlib.pyplot as plt2

        if title is not None:
            plt.title(title)
        plt2.plot(newp, color='red', label='Prediction')
        plt2.plot(newy_test, color='blue', label='Actual')
        print('Test Score: %.2f RMSE' % (rmse))
        plt2.legend(loc='best')
        # plt.show()
        plt2.savefig(f"{self.figurepath}//Test_vs_Classification_{title}.png")
        plt2.close()
        output_string = output_string + 'Test Score: %.2f RMSE' % (rmse)

        return Fscore, rmse, output_string

    def plot_train_test_prediction(self):
        """
        Plots train data, test data and the predictions

        Returns: diagram

        """
        train = self.data[:self.train_data_len]
        valid = self.data[self.train_data_len:]  # self.train_data_len+60]
        # valid = valid.reset_index()
        valid['Predictions'] = self.predictions
        # Visualize the data
        plt.figure(figsize=(16, 6))
        plt.title(f'Model Prediction for {self.symbol}')
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Beta ', fontsize=18)
        plt.plot(train)
        plt.plot(valid[[self.symbol, 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        # plt.show()
        plt.savefig(f"{self.figurepath}//Train_Test_Prediction.png")
        plt.close()

    def get_further_company_information(self) -> tuple:
        """
        downloads further information for the company

        Returns: tuple including country, sector and names of the company

        """

        ticker = yf.Ticker(self.symbol)
        company_profile = ticker.info
        country = company_profile['country']
        # print(company_profile)
        sector = company_profile['sector']
        name = company_profile['longName']
        return country, sector, name

    def test_run(self):
        """
        Executes all necessary steps for a test run

        Returns:

        """
        self.print_beta_history()
        self.train_models()
        self.evaluation()
        self.print_model_evaluation()
        self.plot_train_test_prediction()

    def forecast_beta(self):
        """
        Executes all necessary steps to forecast the beta of a company

        Returns:

        """
        if self.model == 'EMPTY':
            return []

        self.train_models(test_run=False)
        prediction = self.prediction_new()
        latest_prediction = prediction[-1, -1]
        try:
            country, sector, name = self.get_further_company_information()
            return [country, sector, name, latest_prediction, self.symbol]
        except:
            return []


def classifier(wert):
    if wert >= 1.5:
        return 5
    elif wert >= 1.15:
        return 4
    elif wert >= 0.85:
        return 3
    elif wert >= 0.5:
        return 2
    elif wert >= 0:
        return 1
    else:
        return 0


def aktualisieren():
    """

    Returns:

    """
    obj_generate_data = GenerateData()
    df = obj_generate_data.get_beta_for_stocks(save=True)
    # df = obj_generate_data.load_stock_data()
    beta_predictions = pd.DataFrame(columns=["country", "sector", "name", "prediction", "symbol"])
    # do the forecast for all companies included in df
    for counter, symbol in enumerate(df.columns):
        print(f"Prediction based on new data {counter}/{len(df.columns)} Done", end="\r")
        beta_predictor = CompanyBetaPredictor(symbol, df, existing_model=True)
        company_information_list_temp = beta_predictor.forecast_beta()
        if company_information_list_temp != []:
            beta_predictions = beta_predictions.append(pd.DataFrame([company_information_list_temp],
                                                                    columns=["country", "sector", "name",
                                                                             "prediction", "symbol"]),
                                                       ignore_index=True)
    beta_predictions["prediction"] = beta_predictions["prediction"].apply(classifier)
    beta_predictions.to_csv(rf"..//FrontendFallstudie//Data//predicted_daframe.csv")


if __name__ == "__main__":
    print("Which symbol do you want to calculate? (please use the symbols according to yahoo finance)")
    another_symbol_bool = True
    symbol_list = []
    while(another_symbol_bool != False):
        symbol_temp = input("symbol: ")
        another_symbol_bool = input("Do you want to calculate another symbol? [y, n] ")
        if another_symbol_bool.lower() != 'y':
            another_symbol_bool = False
        symbol_list.append(symbol_temp)
    train_new_model = input("Do you want to train an new model (this may take some time)? [y, n] ")
    obj_generate_data = GenerateData()
    # df = obj_generate_data.get_beta_for_stocks(save=True)
    df = obj_generate_data.get_beta_for_stocks(symbol_list, save=False)
    beta_predictions = pd.DataFrame(columns=["country", "sector", "name", "beta_following_period", "symbol"])
    # do the forecast for all companies included in df
    for counter, symbol in enumerate(df.columns):
        if counter == 0:
            print(f"{counter}/{len(df.columns)} Done")
        else:
            print(f"{counter}/{len(df.columns)} Done", end="\r")
        if train_new_model:
            beta_predictor = CompanyBetaPredictor(symbol, df, existing_model=False)
            company_information_list_temp = beta_predictor.forecast_beta()
            print(f"You can find a test report an graphics here: {beta_predictor.figurepath}")
            print(company_information_list_temp)
        else:
            beta_predictor = CompanyBetaPredictor(symbol, df, existing_model=True)
            result = beta_predictor.test_run()
            # print(result)
    input("Please press enter to leave application")
