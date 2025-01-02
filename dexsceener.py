# write python code to make an API call
import requests

# Define the API endpoint
url1 = "https://api.dexscreener.com/token-profiles/latest/v1"

token_url = "https://api.dexscreener.com/latest/dex/tokens/"


def getTokenInfo(tokenAddress):
    
    # Make the API request (POST method in this example)
    try:
        response = requests.post(token_url+tokenAddress)
        
        # Check if the request was successful
        if response.status_code == 200:
            tokenDetails = response.json()  # Return the JSON response from the API
            # print(tokenDetails)
            for pairs in tokenDetails["pairs"]:
                print(pairs["chainId"]+"---"+pairs["priceNative"])
        else:
            return {"error": f"API request failed with status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred: {e}"}



# Make the API call
response = requests.get(url1,headers={})

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()
    print(data)

    for item in data:
        tokenAddress = item["tokenAddress"]
        print(tokenAddress)
        getTokenInfo(tokenAddress)
else:
    print(f"Failed to retrieve data: {response.status_code}")

