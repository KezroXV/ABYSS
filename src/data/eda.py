import pandas as pd 
import numpy as np
from pathlib import Path

class DataExplorer:
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        print(f"Data loaded : {len(df)} rows")
        return df

    def basic_stats(self,df):
      df = pd.DataFrame(df)
      print(f"Basic stats : {df.describe()}")