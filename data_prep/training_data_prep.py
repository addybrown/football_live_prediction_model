import pandas as pd
import numpy as np
from statsbombpy import sb

pd.options.mode.chained_assignment = None


def get_goals_minute_by_minute_data(event_df):
    df = event_df

    match_id = event_df["match_id"].iloc[0]
    df.sort_values(by="timestamp", inplace=True)
    df["is_goal"] = np.where(df["shot_outcome"] == "Goal", 1, 0)

    groupby_data = df.groupby(["team_id", "team", "minute"]).sum().reset_index()
    team_ids = set(groupby_data["team_id"])

    all_dfs = []
    for team_id in team_ids:
        temp_team_df = groupby_data.query(f"team_id == {team_id}")
        temp_team_df.loc[:, "cul_sum"] = temp_team_df["is_goal"].cumsum()

        all_dfs.append(temp_team_df)

    final_play_by_play_df = pd.concat(all_dfs, axis=0)
    final_play_by_play_df["match_id"] = match_id

    final_play_by_play_df.rename(columns={"cul_sum": "team_goals"}, inplace=True)

    main_cols = ["match_id", "team_id", "team", "minute", "team_goals"]

    return final_play_by_play_df[main_cols]


def get_yellow_card_data(event_df):
    df = event_df

    match_id = event_df["match_id"].iloc[0]
    df.sort_values(by="timestamp", inplace=True)

    df["is_yellow"] = np.where(df["foul_committed_card"] == "Yellow Card", 1, 0)

    groupby_data = df.groupby(["team_id", "team", "minute"]).sum().reset_index()
    team_ids = set(groupby_data["team_id"])

    all_dfs = []
    for team_id in team_ids:
        temp_team_df = groupby_data.query(f"team_id == {team_id}")
        temp_team_df.loc[:, "cul_sum"] = temp_team_df["is_yellow"].cumsum()

        all_dfs.append(temp_team_df)

    final_play_by_play_df = pd.concat(all_dfs, axis=0)
    final_play_by_play_df["match_id"] = match_id

    final_play_by_play_df.rename(columns={"cul_sum": "yellow_cards"}, inplace=True)

    main_cols = ["match_id", "team_id", "team", "minute", "yellow_cards"]

    yellow_cards = final_play_by_play_df[main_cols]

    rename_cols = {"team_id": "opp_id", "team": "opp_name", "yellow_cards": "opp_yellow_cards"}

    df1 = yellow_cards
    df2 = yellow_cards.rename(columns=rename_cols)

    df3 = pd.merge(df1, df2, on=["match_id", "minute"]).query("team != opp_name")
    df3["yellow_cards"] = df3["opp_yellow_cards"] - df3["yellow_cards"]

    return df3[main_cols]


def get_red_card_data(event_df):
    df = event_df

    match_id = event_df["match_id"].iloc[0]
    df.sort_values(by="timestamp", inplace=True)
    df["is_red"] = np.where(df["foul_committed_card"] == "Red Card", 1, 0)

    groupby_data = df.groupby(["team_id", "team", "minute"]).sum().reset_index()
    team_ids = set(groupby_data["team_id"])

    all_dfs = []
    for team_id in team_ids:
        temp_team_df = groupby_data.query(f"team_id == {team_id}")
        temp_team_df.loc[:, "cul_sum"] = temp_team_df["is_red"].cumsum()

        all_dfs.append(temp_team_df)

    final_play_by_play_df = pd.concat(all_dfs, axis=0)
    final_play_by_play_df["match_id"] = match_id

    final_play_by_play_df.rename(columns={"cul_sum": "red_cards"}, inplace=True)

    main_cols = ["match_id", "team_id", "team", "minute", "red_cards"]

    red_cards = final_play_by_play_df[main_cols]

    rename_cols = {"team_id": "opp_id", "team": "opp_name", "red_cards": "opp_red_cards"}

    df1 = red_cards
    df2 = red_cards.rename(columns=rename_cols)

    df3 = pd.merge(df1, df2, on=["match_id", "minute"]).query("team != opp_name")
    df3["red_cards"] = df3["red_cards"] - df3["opp_red_cards"]

    return df3[main_cols]


def get_shots_generated(event_df):
    df = event_df

    match_id = event_df["match_id"].iloc[0]
    df.sort_values(by="timestamp", inplace=True)

    df["is_shot_generated"] = np.where(df["type"] == "Shot", 1, 0)

    groupby_data = df.groupby(["team_id", "team", "minute"]).sum().reset_index()
    team_ids = set(groupby_data["team_id"])

    all_dfs = []
    for team_id in team_ids:
        temp_team_df = groupby_data.query(f"team_id == {team_id}")
        temp_team_df.loc[:, "cul_sum"] = temp_team_df["is_shot_generated"].cumsum()

        all_dfs.append(temp_team_df)

    final_play_by_play_df = pd.concat(all_dfs, axis=0)
    final_play_by_play_df["match_id"] = match_id

    final_play_by_play_df.rename(columns={"cul_sum": "shots_generated"}, inplace=True)

    main_cols = ["match_id", "team_id", "team", "minute", "shots_generated"]

    return final_play_by_play_df[main_cols]


def get_outcome_data(event_df):
    match_id = event_df["match_id"].iloc[0]
    goals_df = get_goals_minute_by_minute_data(event_df)
    temp_df = goals_df.groupby(["team_id", "team"]).agg({"team_goals": "max"}).reset_index()
    next_val = temp_df.shift(1).dropna()
    prev_val = temp_df.shift(-1).dropna()

    rename_cols = {"team_id": "opp_team_id", "team": "opp_team_name", "team_goals": "opp_goals"}

    opp_df = pd.concat([prev_val, next_val], axis=0).rename(columns=rename_cols)

    next_val = temp_df.shift(1).dropna()
    prev_val = temp_df.shift(-1).dropna()

    rename_cols = {"team_id": "opp_team_id", "team": "opp_team_name", "team_goals": "opp_goals"}

    opp_df = pd.concat([prev_val, next_val], axis=0).rename(columns=rename_cols)

    new_df = pd.concat([temp_df, opp_df], axis=1)
    new_df["outcome"] = np.where(new_df["team_goals"] == new_df["opp_goals"], 2, 0)
    new_df["outcome"] = np.where(new_df["team_goals"] > new_df["opp_goals"], 1, 0)
    new_df["match_id"] = match_id
    main_cols = ["match_id", "team_id", "team", "outcome"]

    return new_df[main_cols]


def get_score_differential_data(event_df):
    rename_cols = {"team_id": "opp_team_id", "team": "opp_team_name", "team_goals": "opp_team_goals"}

    main_cols = ["match_id", "team_id", "team", "minute", "score_differential"]

    list_set = ([0, 1], [1, 0])

    goals_df = get_goals_minute_by_minute_data(event_df)
    team_ids = goals_df["team_id"].unique().tolist()

    all_dfs = []
    for team_id in team_ids:
        temp_event_df = goals_df.query(f"team_id == {team_id}")
        all_dfs.append(temp_event_df)

    all_team_dfs = []
    for first, second in list_set:
        team_df = pd.merge(all_dfs[first], all_dfs[second].rename(columns=rename_cols), on=["match_id", "minute"])
        team_df["score_differential"] = team_df["team_goals"] - team_df["opp_team_goals"]
        all_team_dfs.append(team_df[main_cols])

    final_score_diff_df = pd.concat(all_team_dfs, axis=0).reset_index(drop=True)

    return final_score_diff_df


def get_final_third_percentage_data(event_df):
    df = event_df

    def get_coordinate(value, index):
        if type(value) != list:
            return None
        else:
            return value[index]

    df["x_value"] = df["location"].apply(lambda x: get_coordinate(x, 0))
    df["y_value"] = df["location"].apply(lambda x: get_coordinate(x, 1))

    final_third_query_string = "(x_value > = 90 and x_value < 120) and (y_value > 0 and y_value < 120)"
    final_third_df = df.query(final_third_query_string)

    # this sums up the final third data amount
    groupby_data = final_third_df.groupby(["team_id", "team", "minute"]).agg({"duration": "sum"}).reset_index()
    team_ids = list(set(final_third_df["team_id"]))

    all_dfs = []
    max_minute = df["minute"].max()

    for team_id in team_ids:
        dataframe_dict = {
            "team_id": [team_id] * (max_minute + 1),
            "team": [df.query(f"team_id == {team_id} ")["team"].iloc[0]] * (max_minute + 1),
            "minute": list(range(0, max_minute + 1)),
        }

        time_dataframe_df = pd.DataFrame(dataframe_dict)
        all_dfs.append(time_dataframe_df)

    empty_minute_df = pd.concat(all_dfs)

    groupby_data = pd.merge(groupby_data, empty_minute_df, how="right", on=["team_id", "team", "minute"])
    groupby_data["duration"].fillna(0, inplace=True)

    window_size = 10
    averages = groupby_data["duration"].rolling(window=window_size).sum()
    groupby_data["rolling_mean_duration"] = averages
    groupby_data["final_third_percentage"] = (groupby_data["rolling_mean_duration"] / 60) / 10
    groupby_data["final_third_percentage"].fillna(0, inplace=True)
    groupby_data["match_id"] = df["match_id"].iloc[0]
    main_cols = ["match_id", "team_id", "team", "minute", "final_third_percentage"]

    return groupby_data[main_cols]


def get_final_training_data_table(match_ids):
    all_data_dfs = []
    for match_id in match_ids:
        if match_id not in [3754136, 3754045, 3754053, 3754226, 3754042, 3754058, 3754245]:
            events_df = sb.events(match_id=match_id)

            goals = get_goals_minute_by_minute_data(events_df)
            yellow_card = get_yellow_card_data(events_df)
            red_card = get_red_card_data(events_df)
            score_differential = get_score_differential_data(events_df)
            shots_generated = get_shots_generated(events_df)
            outcome = get_outcome_data(events_df)
            final_third_percentage = get_final_third_percentage_data(events_df)

            merged_df = goals.merge(
                yellow_card, on=["match_id", "team_id", "team", "minute"], how="inner"
            )  # Merge df1 and df2
            merged_df = merged_df.merge(shots_generated, on=["match_id", "team_id", "team", "minute"], how="inner")
            merged_df = merged_df.merge(
                red_card, on=["match_id", "team_id", "team", "minute"], how="inner"
            )  # Merge with df3
            merged_df = merged_df.merge(score_differential, on=["match_id", "team", "team_id", "minute"], how="inner")
            merged_df = merged_df.merge(outcome, on=["match_id", "team_id", "team"], how="inner")  # Merge with df4
            merged_df = merged_df.merge(
                final_third_percentage, on=["match_id", "team_id", "team", "minute"], how="inner"
            )

            data = merged_df

            all_data_dfs.append(data)

    data = pd.concat(all_data_dfs, axis=0)
    minute_df = (
        data.groupby("match_id").agg({"minute": "max"}).reset_index().rename(columns={"minute": "max_minute_game"})
    )
    data = pd.merge(data, minute_df, on="match_id")
    data["minute_percentage"] = 100 * np.round(data["minute"] / data["max_minute_game"], 2)

    # Getting home_away_value for column
    data_columns = data.columns.tolist() + ["home_away"]
    temp_data_df = pd.merge(data, matches_df[["match_id", "home_team", "away_team"]], on="match_id")
    temp_data_df["home_away"] = np.where(temp_data_df["team"] == temp_data_df["home_team"], 1, 0)
    data = temp_data_df[data_columns]

    data_columns = data.columns.tolist() + ["final_goals"]
    temp_data_df = pd.merge(
        data, matches_df[["match_id", "home_team", "away_team", "home_score", "away_score"]], on="match_id"
    )
    temp_data_df["final_goals"] = np.where(
        temp_data_df["team"] == temp_data_df["home_team"], temp_data_df["home_score"], temp_data_df["away_score"]
    )
    data = temp_data_df[data_columns]

    return data


if __name__ == "__main__":
    matches_df = sb.matches(competition_id=2, season_id=27)
    match_ids = matches_df["match_id"].unique().tolist()
    match_ids = match_ids[:20]
    # TRAINING DATA

    data = get_final_training_data_table(match_ids)
    data.to_csv(
        r"C:\Users\adams\Documents\Courses\Monte_Carlo_Methods\Live_Prediction_Football_Model\data_files\soccer_training_data.csv"
    )
