## Description

This is a draft service to predict private housing prices from URA's data API

Basic Pipeline
![image](https://github.com/user-attachments/assets/f468d8bb-de94-4d5a-90f7-804ea71b8fc3)

Run Service
```
1. docker-compose build

2. docker-compose up
```


Send Request, predicted response will be in $/sqm
```
    import requests

    # URL of the FastAPI prediction endpoint
    url = "http://0.0.0.0:8000/predict/"

    # Example data to send to the model (the JSON payload)
    data = {
        "street": "TAMPINES STREET 73",
        "project": "PINEVALE",
        "marketsegment": "OCR",
        "x": "37836.26604",
        "y": "39121.60695",
        "area": "120.0",
        "floorrange": "06-10",
        "noofunits": "1",
        "contractdate": "125",
        "typeofsale": "3",
        "district": "18",
        "typeofarea": "Strata",
        "tenure": "99 yrs lease commencing from 1997"
    }

    # Send a POST request with the data as JSON
    response = requests.post(url, json=data)

    # Check the response status code and content
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Failed to connect:", response.status_code)
```
