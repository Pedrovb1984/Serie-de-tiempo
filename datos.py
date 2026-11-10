import pandas as pd

def cargar_datos():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    local_path = "datos/airline-passengers.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(local_path)
        
    return df
