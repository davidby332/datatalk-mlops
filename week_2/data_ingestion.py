import os
import pandas as pd

class PullParameters:

    class January:
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet'
        loc = './data/green_tripdata_2023-01.parquet'
    class February:
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet'
        loc = './data/green_tripdata_2023-02.parquet'
    class March:
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-03.parquet'
        loc = './data/green_tripdata_2023-03.parquet'


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

def main():

    PullFunctions.data_pull(
        PullParameters.January.loc,
        PullParameters.January.url
    )

    PullFunctions.data_pull(
        PullParameters.February.loc,
        PullParameters.February.url
    )

    PullFunctions.data_pull(
        PullParameters.March.loc,
        PullParameters.March.url
    )

    
if __name__=='__main__':
    
    main()
