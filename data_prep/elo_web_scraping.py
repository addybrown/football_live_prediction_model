import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
from fuzzywuzzy import fuzz, process
from statsbombpy import sb

def get_country_list():
    api_url = "https://restcountries.com/v2/all"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        # Parse the JSON response
        countries_data = response.json()

    # Extract the names of all countries into a list
        country_names = [country["name"] for country in countries_data]

    # Print the list of country names
    #print(country_names)
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

    country_names.append("England")
    country_names.append("South Korea")
    country_names.append("Russia")

    return country_names

def remove_country_names(country_list, text):
    
    pattern = '|'.join([re.escape(country) for country in country_list])
    output_string = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return output_string
    
def scrape_ranking_elo(page_number, country_list):
    
    # Define the URL of the API endpoint
    api_url = f"https://footballdatabase.com/ranking/world/{page_number}"  # Replace with the actual API URL

    # Make an HTTP GET request to the API
    response = requests.get(api_url)

    elo_rating_df = pd.read_html(response.text)
    elo_rating_df = elo_rating_df[0]
    
    elo_rating_df["Rank"] = pd.to_numeric(elo_rating_df["Rank"],errors='coerce')
    elo_rating_df["Club / Country"] = elo_rating_df["Club / Country"].apply(lambda x: remove_country_names(country_list,x))
    elo_rating_df = elo_rating_df.dropna()
    
    rename_cols = {
        "Rank":"rank",
        "Club / Country":"club",
        "Points":"points",
        "1-yr change":"year_change"
    }

    elo_rating_df = elo_rating_df.rename(columns = rename_cols)
    elo_rating_df["rank"] = elo_rating_df["rank"].astype(int)
    elo_rating_df = elo_rating_df.reset_index(drop=True)
    
    return elo_rating_df

def find_closest_match(name, choices):
    match, score = process.extractOne(name, choices)
    return match if score >= 95 else None


if __name__ == "__main__":
    all_elo_ratings_dfs = []
    country_list = get_country_list()
    for value in range(0,10):
        elo_rating_df = scrape_ranking_elo(value+1, country_list)
        all_elo_ratings_dfs.append(elo_rating_df)
    
    all_clubs_elo_rating_df = pd.concat(all_elo_ratings_dfs).reset_index(drop=True)
    matches_df = sb.matches(competition_id = 2, season_id = 27)
    sb_teams = matches_df["home_team"].unique().tolist()
    all_clubs_elo_rating_df['sb_closest_match'] = all_clubs_elo_rating_df['club'].apply(find_closest_match, args=(sb_teams,))
    
    premier_league_clubs = all_clubs_elo_rating_df.dropna()
    premier_league_clubs = premier_league_clubs.query("club!='Everton CD' ")
    
    premier_league_clubs.to_csv(r"C:\Users\adams\Documents\Courses\Monte_Carlo_Methods\Live_Prediction_Football_Model\data_files\scraped_elo_data.csv")