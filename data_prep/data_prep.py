import pandas as pd
import numpy as np

from statsbombpy import sb


pd.options.mode.chained_assignment = None


if __name__ == "__main__":
    matches_df = sb.matches(competition_id = 2, season_id = 27)
    train_data = pd.read_csv(r"C:\Users\adams\Documents\Courses\Monte_Carlo_Methods\Live_Prediction_Football_Model\data_files\soccer_training_data.csv", index_col=0)
    elo_data = pd.read_csv(r"C:\Users\adams\Documents\Courses\Monte_Carlo_Methods\Live_Prediction_Football_Model\data_files\scraped_elo_data.csv", index_col=0)
    shortened_match_df = matches_df[["match_id","home_team","away_team"]]
    
    temp_elo_data = elo_data
    temp_elo_data.index = temp_elo_data["sb_closest_match"]
    elo_mapping_dict = temp_elo_data.to_dict()["points"]
    
    shortened_match_df["home_elo"] = shortened_match_df["home_team"].map(elo_mapping_dict)
    shortened_match_df["away_elo"] = shortened_match_df["away_team"].map(elo_mapping_dict)

    shortened_match_df.loc[:,"home_elo_diff"] = shortened_match_df["home_elo"] - shortened_match_df["away_elo"]
    shortened_match_df.loc[:,"away_elo_diff"] = shortened_match_df["away_elo"] - shortened_match_df["home_elo"]
    
    test_train_data = train_data.merge(shortened_match_df, on = "match_id")
    test_train_data["elo_diff"] = np.where(test_train_data["home_away"]==1, test_train_data["home_elo_diff"], test_train_data["away_elo_diff"])
    
    main_cols = train_data.columns.tolist() + ["elo_diff"]
    train_data = test_train_data[main_cols]
    
    train_data.to_csv(r"C:\Users\adams\Documents\Courses\Monte_Carlo_Methods\Live_Prediction_Football_Model\data_files\final_train_data.csv")