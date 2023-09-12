import pandas as pd
import numpy as np
from scipy.stats import poisson


def invlogit(x):
    return 1 / (1 + np.exp(-x))


def get_model_probability_dataframe(trace, summary_df, home_data, away_data):
    
    x_list = range(0, 5)
    
    final_combined_df = get_model_setup_dataframe(trace, summary_df, home_data, away_data)
    
    final_combined_df["home_probabilities"] = final_combined_df["home_theta"].apply(lambda y: [poisson.pmf(x, y) for x in x_list])
    final_combined_df["away_probabilities"] = final_combined_df["away_theta"].apply(lambda y: [poisson.pmf(x, y) for x in x_list])
    
    model_probability_df = get_result_matrix_predictions_dataframe(final_combined_df, x_list)
    
    return model_probability_df

def get_model_setup_dataframe(trace, summary_df, home_data, away_data):
    
    # summary_predictions = pm.summary(posterior_predictive["posterior_predictive"])
    # temp_summary_df = summary_predictions.reset_index()

    home_data["mean"] = summary_df[summary_df["index"].str.contains("home_points")]["mean"].tolist()
    home_data["sd"] = summary_df[summary_df["index"].str.contains("home_points")]["sd"].tolist()

    away_data["mean"] = summary_df[summary_df["index"].str.contains("away_points")]["mean"].tolist()
    away_data["sd"] = summary_df[summary_df["index"].str.contains("away_points")]["sd"].tolist()

    home_rename_cols = {
        "team_id":"home_id",
        "team":"home_name",
        "mean":"home_mean",
        "sd":"home_sd",
        "final_goals":"home_final_goals",
        "team_goals":"home_team_goals"
    }

    away_rename_cols = {
        "team_id":"away_id",
        "team":"away_name",
        "mean":"away_mean",
        "sd":"away_sd",
        "final_goals":"away_final_goals",
        "team_goals":"away_team_goals"
    }

    home_cols = ["match_id","home_name","home_mean","home_sd","home_final_goals","home_team_goals"]
    away_cols = ["match_id","away_name","away_mean","away_sd","away_final_goals","away_team_goals"]

    temp_home = home_data.rename(columns = home_rename_cols)[home_cols]
    temp_away = away_data.rename(columns = away_rename_cols)[away_cols]

    # combined_data = pd.merge(temp_home, temp_away, on=["match_id"])

    df_value = trace["posterior"].to_dataframe().reset_index()

    mean_vals = []
    mean_vals_dict = {}
    for value in df_value.columns.tolist():
        if value not in ["chain","draw"]:
            mean_vals.append(df_value[value].mean())
            mean_vals_dict[value]=df_value[value].mean()
            
    rename_home_data_cols = {
        "team_id":"home_id",
        "team":"home_team",
        "team_goals":"home_team_goals",
        "yellow_cards":"home_yellow_cards",
        "shots_generated":"home_shots_generated",
        "red_cards":"home_red_cards",
        "score_differential":"home_score_diff",
        "outcome":"home_outcome",
        "elo_diff_better_or_worse":"home_elo_diff",
        "home_away":"home",
        "final_goals":"home_final_goals"
    }

    home_cols = [value for value in rename_home_data_cols.values()]

    rename_away_cols = {

        "team_id":"away_id",
        "team":"away_team",
        "team_goals":"away_team_goals",
        "yellow_cards":"away_yellow_cards",
        "shots_generated":"away_shots_generated",
        "red_cards":"away_red_cards",
        "score_differential":"away_score_diff",
        "outcome":"away_outcome",
        "elo_diff_better_or_worse":"away_elo_diff",
        "home_away":"away",
        "final_goals":"away_final_goals"
    }

    away_cols = [value for value in rename_away_cols.values()]

    temp_home_df = home_data.rename(columns = rename_home_data_cols)
    temp_away_df = away_data.rename(columns = rename_away_cols)


    columns = ["match_id","minute","time_remaining_percentage","minute"] + home_cols + away_cols

    final_combined_df = pd.merge(temp_home_df,temp_away_df, on =["match_id","minute","time_remaining_percentage"])[columns]


    combo_dict = {

        "beta_team_goals":"team_goals",
        "beta_reds":"red_cards",
        "beta_yellows":"yellow_cards",
        "beta_shots_generated":"shots_generated",
        "beta_score_diff":"score_diff",
        "beta_elo_diff":"elo_diff"
    }

    #for home
    for key,value in mean_vals_dict.items():
        
        if key == 'home':
            home_value=value
            
        elif key == "alpha":
            home_value = home_value+value
            
        else:
            col_value = combo_dict[key]
            
            home_value = home_value + value*final_combined_df[f"home_{col_value}"]
            

    for key,value in mean_vals_dict.items():
        
        if key == 'home':
            away_value=0
            
        elif key == "alpha":
            away_value = away_value+value
            
        else:
            col_value = combo_dict[key]
            
            away_value = away_value + value*final_combined_df[f"away_{col_value}"]

    final_combined_df["home_logit_val"] = home_value.apply(lambda x: invlogit(x))
    final_combined_df["away_logit_val"] = away_value.apply(lambda x: invlogit(x))

    final_combined_df["home_theta"] = final_combined_df["time_remaining_percentage"]*final_combined_df["home_logit_val"] 
    final_combined_df["away_theta"] = final_combined_df["time_remaining_percentage"]*final_combined_df["away_logit_val"]
    
    return final_combined_df

def get_probabilities_dataframe(matrix_df, home_team, away_team,time_remaining_percentage,score_diff, x_list):
    home_team = home_team.replace("'", "")
    away_team = away_team.replace("'", "")

    temp_val_df = pd.DataFrame()
    temp_val_df["goals"] = x_list
    temp_val_df["home_probabilities"] = matrix_df.query(f"home_team == '{home_team}' and time_remaining_percentage == {time_remaining_percentage} ")["home_probabilities"].iloc[
        0
    ]

    temp_val_df_2 = pd.DataFrame()
    temp_val_df_2["goals"] = x_list
    temp_val_df_2["away_probabilities"] = matrix_df.query(f"away_team == '{away_team}' and time_remaining_percentage == {time_remaining_percentage} ")[
        "away_probabilities"
    ].iloc[0]

    temp_val_df.index = temp_val_df["goals"]
    temp_val_df_2.index = temp_val_df_2["goals"]

    temp_val_df.drop(columns=["goals"], inplace=True)
    temp_val_df_2.drop(columns=["goals"], inplace=True)

    arr1 = temp_val_df.values
    arr2 = temp_val_df_2.values

    result_arr = arr1[:, np.newaxis, :] * arr2[np.newaxis, :, :]

    all_dfs = []

    for value in range(0, len(temp_val_df.index.values)):
        temp_df = pd.DataFrame()
        temp_df["combined_probabilities"] = list(result_arr[value].flatten())
        temp_df[f"home_goals"] = [temp_val_df.index.values[value]] * len(temp_val_df)
        temp_df["home_probabilities"] = [temp_val_df["home_probabilities"].tolist()[value]] * len(temp_val_df)
        temp_df[f"away_goals"] = temp_val_df_2.index.values
        temp_df["away_probabilities"] = temp_val_df_2["away_probabilities"]

        all_dfs.append(temp_df)

    final_matrix_df = pd.concat(all_dfs, axis=0)

    if score_diff == 0:
        home_win = np.sum(final_matrix_df.query("home_goals > away_goals")["combined_probabilities"])
        away_win = np.sum(final_matrix_df.query("home_goals < away_goals")["combined_probabilities"])
        draw = np.sum(final_matrix_df.query("home_goals == away_goals")["combined_probabilities"])
    
        return home_win, away_win, draw

    elif score_diff > 0:

        away_amount_to_win = score_diff + 1
        home_win = np.sum(final_matrix_df.query("home_goals > away_goals")["combined_probabilities"]) + np.sum(final_matrix_df.query("home_goals == away_goals")["combined_probabilities"])
        #away_win = np.sum(final_matrix_df.query(f"home_goals < away_goals + {away_amount_to_win}")["combined_probabilities"])
        draw = np.sum(final_matrix_df.query(f"away_goals - home_goals == {score_diff}")["combined_probabilities"])
        away_win = 1- (home_win+draw)

        return home_win, away_win, draw
    
    else:

        home_amount_to_win = score_diff + 1
        away_win = np.sum(final_matrix_df.query("home_goals < away_goals")["combined_probabilities"]) + np.sum(final_matrix_df.query("home_goals == away_goals")["combined_probabilities"])
        #home_win = np.sum(final_matrix_df.query(f"home_goals + {home_amount_to_win}> away_goals")["combined_probabilities"])
        draw = np.sum(final_matrix_df.query(f"home_goals - away_goals == {score_diff}")["combined_probabilities"])
        home_win = 1- (away_win+draw)

        return home_win, away_win, draw
    

def get_result_matrix_predictions_dataframe(matrix_df, x_list):
    matrix_df["home_team"] = matrix_df["home_team"].str.replace("'", "")
    matrix_df["away_team"] = matrix_df["away_team"].str.replace("'", "")
    all_dfs = []
    for row in range(0, len(matrix_df)):
        # value_df = pd.DataFrame()

        match_id = matrix_df.loc[row, "match_id"]
        home_team = matrix_df.loc[row, "home_team"]

        away_team = matrix_df.loc[row, "away_team"]

        #time_remaining = matrix_df.loc[row,"time_remaining"]
        time_remaining_percentage = matrix_df.loc[row,"time_remaining_percentage"]
        home_team_goals = matrix_df.loc[row,"home_team_goals"]
        away_team_goals = matrix_df.loc[row,"away_team_goals"]

        home_score_diff = matrix_df.loc[row,"home_score_diff"]

        score_diff = home_score_diff

        home_win, away_win, draw = get_probabilities_dataframe(matrix_df, home_team, away_team,time_remaining_percentage, score_diff, x_list)

        dict_value = {
            "match_id":[match_id],
            "home_team": [home_team],
            "away_team": [away_team],
            "time_remaining_percentage":[time_remaining_percentage],
            "home_team_goals":[home_team_goals],
            "away_team_goals":[away_team_goals],
            "home_win": [home_win],
            "draw": [draw],
            "away_win": [away_win]
        }

        df_temp = pd.DataFrame(dict_value)

        all_dfs.append(df_temp)

    result_matrix_predictions_df = pd.concat(all_dfs).reset_index(drop=True)

    return result_matrix_predictions_df