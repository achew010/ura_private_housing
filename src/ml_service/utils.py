import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression

def find_agg_price_neighbours(df, num_neighbours=2, agg_method='mean'):
    """
    Finds the average price of the closest neighbors for each row in the DataFrame, 
    restricted to transactions within the same year.
    
    Parameters:
        df (pd.DataFrame): A DataFrame containing columns ['x-axis', 'y-axis', 
                                                           'transaction_month', 
                                                           'transaction_year', 'price'].
    Returns:
        pd.Series: A Series containing the average price of theclosest neighbors 
                   for each row.
    """
    # Initialize an empty list to store the aggregated price of nearest neighbors for each row
    agg_prices = []

    # Group transactions by year
    for year, group in df.groupby(['tx_month', 'tx_year']):
        # Extract coordinates and prices from the same year group
        coords = group[['x', 'y']].values
        prices = group['price_per_sqm'].values
        
        # Use NearestNeighbors to find the closest neighbors
        nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='auto').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Calculate the median price of the nearest neighbors for each point
        agg_price_per_row = []
        for i in range(len(group)):
            # Get indices of the closest neighbors in group
            nearest_indices = indices[i]
            # get their price per sqm
            nearest_prices = prices[nearest_indices]
            # calculate median of neighbours
            if agg_method == 'median':
                agg_price = int(np.median(nearest_prices))
            elif agg_method == 'mean':
                agg_price = int(np.mean(nearest_prices))
            agg_price_per_row.append(agg_price)
        
        # Append results for this year to the main list
        agg_prices.extend(agg_price_per_row)
    
    # Convert the list into a pandas Series and return it
    return pd.Series(agg_prices, index=df.index)

def feature_engineer(df:pd.DataFrame, predict=False):
    raw = df
    if 'propertytype' in df.columns:
        raw = df[df['propertytype']=='Executive Condominium']
    if predict:
        raw["lease_commencement"] = str(raw["tenure"])[-4:]
        raw["tx_month"] = int(raw["contractdate"].iloc[0,0][:-2])
        raw["tx_year"] = int(raw["contractdate"].iloc[0,0][-2:])
        raw['num_years_from_tenure'] = (2000 + int(raw["tx_year"].iloc[0,0])) - int(raw["lease_commencement"].iloc[0,0])
        return raw[['street', 'project', 'marketsegment',
            'area', 'floorrange',
            'typeofsale', 'district',
            "lease_commencement", "tx_month", "tx_year", "num_years_from_tenure"]]
    else:
        raw["lease_commencement"] = raw["tenure"].astype(str).str[-4:]
        raw["tx_month"] = raw["contractdate"].astype(str).str[:-2].astype(int)
        raw["tx_year"] = raw["contractdate"].astype(str).str[-2:].astype(int)
        raw["price_per_sqm"] = raw['price'].astype(int) // raw['area'].astype(int)
        raw['neighbour_median_price_per_sqm'] = find_agg_price_neighbours(raw)
        raw['num_years_from_tenure'] = (2000 + raw["tx_year"].astype(int)) - raw["lease_commencement"].astype(int)
    
        # Remove unimportant columns like noofunits, typeofarea (no variance),tenure, contractdate
        return raw[['street', 'project', 'marketsegment',
            'area', 'floorrange',
            'typeofsale', 'district',
            "lease_commencement", "tx_month", "tx_year", "num_years_from_tenure", "price_per_sqm"]] #"neighbour_median_price_per_sqm"

def split(dataset):
    X = dataset[[col for col in dataset.columns if col != "price_per_sqm"]]
    Y = dataset["price_per_sqm"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def predict_encode(X, ohe, mms, fit=False):
    continuous_vars = ['area'] #, 'neighbour_median_price_per_sqm'
    tmp = X[[col[0] for col in X.columns if col[0] not in continuous_vars]].values[0].reshape(1, -1)
    X_ohe = ohe.transform(tmp).toarray()
    X_scaled = mms.transform(X[[col[0] for col in X.columns if col[0] in continuous_vars]])
    return np.concatenate([X_ohe, X_scaled], axis=-1)


def encode(X, ohe, mms, fit=False):
    continuous_vars = ['area'] #, 'neighbour_median_price_per_sqm'
    if fit:
        X_ohe = ohe.fit_transform(X[[col for col in X.columns if col not in continuous_vars]]).toarray()
        X_scaled = mms.fit_transform(X[[col for col in X.columns if col in continuous_vars]])
    else:
        X_ohe = ohe.transform(X[[col for col in X.columns if col not in continuous_vars]]).toarray()
        X_scaled = mms.transform(X[[col for col in X.columns if col in continuous_vars]])
    return np.concatenate([X_ohe, X_scaled], axis=-1)

def train_predict(model, X_train, Y_train, X_test):
    model.fit(X_train, Y_train)    
    return model, model.predict(X_test)

def evaluate(test, pred):
    return {
        "mean_absolute_error": "%.2f" % mean_absolute_error(test, pred),
        "root_mean_squared_error": "%.2f" % root_mean_squared_error(test, pred),
    }    


def select_model():
    # from sklearn.ensemble import RandomForestRegressor
    # model = RandomForestRegressor(n_estimators = 1000, max_depth=3, random_state=0)

    # from sklearn.ensemble import GradientBoostingRegressor
    # params = {
    #     "n_estimators": 300,
    #     "max_depth": 3,
    #     "min_samples_split": 3,
    #     "learning_rate": 0.3,
    #     "loss": "squared_error",
    # }
    # model = GradientBoostingRegressor(**params)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    return model
