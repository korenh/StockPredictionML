import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


class StockPrediction():
    def __init__(self, symbol, provider):
        self.provider = provider
        self.symbol = symbol
        self.start = dt.datetime(2014, 1, 1)
        self.end = dt.datetime.now()
        self.data = web.DataReader(
            self.symbol, self.provider, self.start, self.end)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = Sequential()

    def train_data(self, prediction_days):
        scaled_data = self.scaler.fit_transform(
            self.data['Close'].values.reshape(-1, 1))

        x_train = []
        y_train = []

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.model.add(LSTM(units=50, return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, epochs=25, batch_size=32)

    def future(self, prediction_days):
        test_start = dt.datetime(2020, 1, 1)
        test_end = dt.datetime.now()
        test_data = web.DataReader(
            self.symbol, self.provider, test_start, test_end)

        total_dataset = pd.concat(
            (self.data['Close'], test_data['Close']), axis=0)
        model_inupts = total_dataset[len(
            total_dataset) - len(test_data) - prediction_days:].values
        model_inupts = model_inupts.reshape(-1, 1)
        model_inupts = self.scaler.transform(model_inupts)

        real_data = [
            model_inupts[len(model_inupts) + 1 - prediction_days:len(model_inupts+1), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(
            real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = self.model.predict(real_data)
        prediction = self.scaler.inverse_transform(prediction)
        return prediction[0][0]

    def show(self, prediction_days, prediction, since):
        test_start = dt.datetime(since, 1, 1)
        test_end = dt.datetime.now()
        test_data = web.DataReader(
            self.symbol, self.provider, test_start, test_end)
        actual_prices = test_data['Close'].values

        total_dataset = pd.concat(
            (self.data['Close'], test_data['Close']), axis=0)
        model_inupts = total_dataset[len(
            total_dataset) - len(test_data) - prediction_days:].values
        model_inupts = model_inupts.reshape(-1, 1)
        model_inupts = self.scaler.transform(model_inupts)

        x_test = []

        for x in range(prediction_days, len(model_inupts)):
            x_test.append(model_inupts[x-prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = self.model.predict(x_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices)

        plt.plot(actual_prices, color='black', label='Price')
        plt.plot(predicted_prices, color='red', label='Prediction')
        plt.title(self.symbol + ' | Tomorrow Prediction - ' + str(prediction))
        plt.xlabel('Range')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


stock = StockPrediction("PYPL", "yahoo")
stock.train_data(120)
prediction = stock.future(120)
stock.show(120, prediction, 2019)
