import requests
import json
import os
from dotenv import load_dotenv
from db_utils import DatabaseManager

# Load environment variables from the .env file
load_dotenv()

def get_api_key():
    """
    Load the API key from the environment variables.
    """
    access_key = os.getenv('URA_ACCESS_KEY')
    token = os.getenv('URA_API_TOKEN')
    if not access_key or not token:
        raise Exception("API keys not found. Make sure you have set it in the .env file.")
    return {'access_key': access_key, 'token': token}

def get_api_response(api_keys, batch_number=1):
    """
    Send a GET request to the URA API and return the response data.
    
    Parameters:
        api_key (str): The API key for authenticating the request.
        batch_number (int): The batch number for the request, default is 1.
    
    Returns:
        dict: Parsed JSON response from the API.
    """
    # Define the API URL with the batch parameter
    url = f"https://www.ura.gov.sg/uraDataService/invokeUraDS"
    access_key = api_keys["access_key"]
    token = api_keys["token"]
    
    # Define the headers
    params = {
        "service": "PMI_Resi_Transaction",
        "batch": batch_number
    }

    headers = {
        "User-Agent": 'PostmanRuntime/7.28.4',
        "AccessKey": access_key,
        "Token": token,
    }

    # Make the GET request
    response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the parsed JSON response
    else:
        raise Exception(f"Failed to retrieve data. HTTP Status code: {response.status_code}")

def main():
    """
    Main function to execute the script.
    """
    try:
        # Get the API key
        api_key = get_api_key()

        db_manager = DatabaseManager()
        # Get the API response
        for batch_idx in range(1, 5):
            data = get_api_response(api_key, batch_number=batch_idx)
            print(data['Status'])
            # Save data to the PostgreSQL database
            db_manager.save_data_to_db(data)

        db_manager.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
