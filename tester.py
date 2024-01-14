import pandas as pd
from threading import Thread
pd.options.mode.chained_assignment = None  # default='warn'

c1 = -42.379
c2 = 2.04901523
c3 = 10.14333127
c4 = -0.22475541
c5 = -6.83783e-03
c6 = -5.481717e-02
c7 = 1.22874e-03
c8 = 8.5282e-04
c9 = -1.99e-06


class Algorithm:
    def format_x(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x['Date'] = pd.to_datetime(x['Date'])
        x.sort_values(by='Date', ascending=True)
        x['Date'] = x['Date'] - pd.Timedelta(hours=2)
        x['timestamp'] = x['Date'].apply(lambda x: int(x.timestamp()))
        x['day'] = x['Date'].dt.day
        x['month'] = x['Date'].dt.month
        x['hour'] = x['Date'].dt.hour
        x['minute'] = x['Date'].dt.minute
        x['timeofday'] = x['minute'] + x['hour'] * 60
        x['timofyear'] = (x['Date'].dt.dayofyear * 24 + x['hour']) * 60 + x['minute']

        x['heat_index'] = c1 + c2 * x['temp'] + c3 * x['humidity'] + c4 * x['temp'] * x['humidity'] + c5 * x[
            'temp'] ** 2 + c6 * x['humidity'] ** 2 + c7 * x['temp'] ** 2 * x['humidity'] + c8 * x['temp'] * x[
                              'humidity'] ** 2 + c9 * x['temp'] ** 2 * x['humidity'] ** 2

        return x

    def predict(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        x_train = self.format_x(x_train)
        x_test = self.format_x(x_test)
        return self.algo(x_train, y_train, x_test)

    def algo(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        return y_train


def thread_predict(algo: Algorithm, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame,
                   result: [pd.DataFrame], index: int, debug: bool = True) -> None:
    result[index] = algo.algo(x_train, y_train, x_test)
    if debug:
        print(f"Thread: {index} done !")


class Tester:
    def __init__(self):
        self.training_set = pd.read_csv("training.csv")
        self.test_set = pd.read_csv("test_students.csv")

    def evaluate(self, algo: Algorithm, k: int = 10, debug: bool = True, use_thread: bool = False) -> [pd.Series,
                                                                                                       pd.Series,
                                                                                                       pd.Series]:
        x = self.training_set.drop(['solar_production'], axis=1).copy()
        y = self.training_set['solar_production'].copy()
        x = algo.format_x(x)

        n = len(x)
        nk = int(n / k)

        res = pd.Series(dtype='float64')
        y_true_tot = pd.Series(dtype='float64')
        y_pred_tot = pd.Series(dtype='float64')
        if use_thread:
            result = [None] * k
            threads = [None] * k
        for k_split in range(k):
            if debug:
                print(f"K: {k_split} start !")
            first_split = n - (k_split + 1) * nk
            second_split = first_split + nk

            x_train = pd.concat((x[:first_split], x[second_split:n]))
            y_train = pd.concat((y[:first_split], y[second_split:n]))
            x_test = x[first_split:second_split]
            y_true = y[first_split:second_split]

            if use_thread:
                threads[k_split] = Thread(target=thread_predict,
                                          args=(algo, x_train, y_train, x_test, result, k_split, debug))
                threads[k_split].start()
            else:
                y_pred = algo.algo(x_train, y_train, x_test)
                y_pred_tot = pd.concat((y_pred_tot, pd.DataFrame(y_pred)))
                if debug:
                    print(f"K: {k_split} done !")
            y_true_tot = pd.concat((y_true_tot, y_true))

        if use_thread:
            for i in range(len(threads)):
                if debug:
                    print(f"Waiting for thread {i} results...")
                threads[i].join()
                if debug:
                    print(f"Get thread {i} results...")
                y_pred = result[i]
                y_pred_tot = pd.concat((y_pred_tot, pd.DataFrame(y_pred)))

            if debug:
                print(f"Done getting results !")

        return [y_pred_tot, y_true_tot]

    def generate_result(self, algo: Algorithm, csv_name: str = "RESULT.csv") -> pd.DataFrame:
        x_train = self.training_set.drop(['solar_production'], axis=1).copy()
        y_train = self.training_set['solar_production'].copy()
        x_test = self.test_set.copy()
        id_col = x_test['Unnamed: 0']

        y_pred = algo.predict(x_train, y_train, x_test)

        # création du DataFrame à partir des colonnes "id" et "predicted"
        df = pd.DataFrame({'id': id_col, 'predicted': y_pred})

        # enregistrement du DataFrame dans un fichier CSV
        df.to_csv(csv_name, index=False)
        return df
