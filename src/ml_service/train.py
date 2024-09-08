import os
import psycopg2
import pandas as pd
import time
from db_utils import DatabaseManager
from utils import *

def train_model(df: pd.DataFrame):
    """
    Train a simple linear regression model using the data.
    """
    if df is not None:
        dataset = feature_engineer(df)
        X_train, X_test, Y_train, Y_test = split(dataset)

        ohe = OneHotEncoder(handle_unknown='ignore')
        mms = MinMaxScaler()

        X_train_encoded = encode(X_train, ohe, mms, fit=True)
        X_test_encoded = encode(X_test, ohe, mms, fit=False)

        # Train the model
        model = select_model()
        model, pred = train_predict(model, X_train_encoded, Y_train, X_test_encoded)

        metadict = {
            "train_metrics": evaluate(Y_test, pred),
            "ohe": ohe,
            "mms": mms,
        }

        return model, metadict
    else:
        print("No data available to train the model.")

def main():
    """
    Main function to execute the machine learning service.
    """
    # Load data from the PostgreSQL database
    db_manager = DatabaseManager()
    
    while not db_manager.is_table_populated():
        print("Transactions table not populated. Sleeping for 30 secs...")
        time.sleep(30)
    
    data_df = db_manager.load_data_from_db()

    #data_df.to_csv('data.csv', index=None)

    # Train the machine learning model
    model, metadict = train_model(data_df)
    if model:
        print(metadict["train_metrics"])
        import pickle
        with open('model/model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        
        with open('model/meta.pkl', 'wb') as metadict_file:
            pickle.dump(metadict, metadict_file)

    db_manager.close()

if __name__ == "__main__":
    main()
