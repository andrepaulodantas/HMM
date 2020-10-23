import urllib.request
import pandas as pd
import json
import ast
import datetime
from datetime import timedelta

def getBTCData(start_date: str = None, end_date: str = None):
    '''
    Função que obtem os dados históricos do Bitcoin. 
    Args -
    start_date : Str - String ('Ano-Mês-dia') contendo a data de início da série histórica. 
                 Default - 5 anos a menos que end_date;
    end_date : Str - String ('Ano-Mês-dia') contendo a data de fim da série histórica. 
               Default Data atual
    
    Returns:
            Pandas DataFrame. Columns = [Open, High, Low, Close, Volumne], Index = Datetime Object
    '''
    now = datetime.datetime.now().date()
    end = end_date if end_date else now.strftime('%Y-%m-%d')
    start = start_date if start_date else (now - timedelta(days=1826)).strftime('%Y-%m-%d')
    url = f"https://data.messari.io/api/v1/assets/bitcoin/metrics/price/time-series?start={start}&end={end}&interval=1d"
    data_dict = urllib.request.urlopen(url).read()
    data_dict = json.loads(data_dict.decode("utf-8"))
    df = pd.DataFrame(data_dict['data']['values'], columns = data_dict['data']['parameters']['columns'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit = "ms")
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    dados_bitcoin = getBTCData()
    print(dados_bitcoin.sample(5))