import os
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

class PullParameters:

    class January:
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet'
        loc = './data/jan_greencab.parquet'
    class February:
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet'
        loc = './data/feb_greencab.parquet'


class PullFunctions:

    def data_pull(location, url):

        os.system(f'wget {url} -O {location}')

        if location[-2:] == 'gz':
            df = pd.read_csv(location, compression='gzip')
        elif location[-3:] == 'csv':
            df = pd.read_csv(location)
        elif location[-7:] == 'parquet':
            df = pd.read_parquet(location)
        else:
            raise Exception('Unrecognized file type')

        return df

    def data_clean_timing(data, month, year):

        data = data[
            (data[AnalysisFunctions.pickup].dt.month == month) &
            (data[AnalysisFunctions.pickup].dt.year == year) & 
            (data[AnalysisFunctions.dropoff].dt.month == month) &
            (data[AnalysisFunctions.dropoff].dt.year == year)
        ]

        return data

    def ohe_universe(data, categorical_vars):

        data[categorical_vars] = data[categorical_vars].astype(str)

        data_dicts = data[categorical_vars].astype(str).to_dict(orient='records')
        dv = DictVectorizer()

        return dv.fit(data_dicts)

    def trip_duration(data):

        data['duration'] = (
            data[AnalysisFunctions.dropoff] - data[AnalysisFunctions.pickup]
        ).dt.seconds / 60

        return data
    
class AnalysisFunctions:

    pickup = 'lpep_pickup_datetime'
    dropoff = 'lpep_dropoff_datetime'
    categorical = ['PULocationID', 'DOLocationID']

    def question_1(data):

        return data.shape[1]

    def question_2(data):

        PullFunctions.trip_duration(data)

        return np.std(data['duration'])

    def question_3(data):

        return data[data['duration'].between(1.00, 60.00)]

    def regression_questions(data, ohe, model = None):

        try:
            X = ohe.transform(
                data[AnalysisFunctions.categorical]
                .astype(str)
                .to_dict(orient='records')
            )
        except:
            raise Exception('No categorical mapping provided')
        
        target = 'duration'
        y_true = data[target].values

        if not model:
            model = LinearRegression()
            model.fit(X, y_true)

            y_pred = model.predict(X)
        else:
            y_pred = model.predict(X)

        return y_pred, y_true, model

def main():

    jan_data = PullFunctions.data_pull(
        PullParameters.January.loc,
        PullParameters.January.url
    )

    jan_data = PullFunctions.data_clean_timing(
        jan_data,
        1, 
        2023
    )

    feb_data = PullFunctions.data_pull(
        PullParameters.February.loc,
        PullParameters.February.url
    )

    feb_data = PullFunctions.data_clean_timing(
        feb_data,
        2, 
        2023
    )
    
    feb_data = PullFunctions.trip_duration(feb_data)
    feb_data = AnalysisFunctions.question_3(feb_data)

    all_data = pd.concat([jan_data, feb_data], axis = 'rows')
    ohe = PullFunctions.ohe_universe(all_data, AnalysisFunctions.categorical)

    print('The Answer to Question 1 is: ', AnalysisFunctions.question_1(jan_data))
    print('The Answer to Question 2 is: ', AnalysisFunctions.question_2(jan_data))
    print('The Answer to Question 3 is: ', len(AnalysisFunctions.question_3(jan_data)) / len(jan_data))

    jan_data = PullFunctions.trip_duration(jan_data)
    jan_data = AnalysisFunctions.question_3(jan_data)

    print(
        'The Answer to Question 4 is: ', 
        ohe.transform(
            jan_data[AnalysisFunctions.categorical]
            .astype(str).to_dict(orient='records'))
            .shape[1]
    )

    y_pred, y_train, lr = AnalysisFunctions.regression_questions(jan_data, ohe)

    print('The Answer to Question 5 is: ', root_mean_squared_error(y_train, y_pred))

    y_pred, y_test, _ = AnalysisFunctions.regression_questions(feb_data, ohe, lr)

    print('The Answer to Question 6 is: ', root_mean_squared_error(y_test, y_pred))


if __name__=='__main__':
    
    main()
