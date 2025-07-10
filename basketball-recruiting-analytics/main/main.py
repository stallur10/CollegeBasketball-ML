import pandas as pd
import numpy as np
import tensorflow as tf
import itertools

# --- Config ---
MODEL_PATH = './models/predict-conf-wins'
PLAYER_DATA_CSV = './data/player_data.csv'
TEAM_NAME = 'Illinois'
MAX_PLAYERS = 8

# --- Features used in training ---
PLAYER_FEATURES = [
    "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per", "blk_per", "stl_per",
    "ftr", "porpag", "adjoe", "pfr", "drtg", "adrtg", "dporpag", "bpm", "obpm", "dbpm", 
    "gbpm", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "3p/100?"
]

# --- Load model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Load and filter player data ---
stats = pd.read_csv(PLAYER_DATA_CSV)
stats.columns = [  # Reapply correct headers
    "player_name", "team", "conf", "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", 
    "AST_per", "TO_per", "FTM", "FTA", "FT_per", "twoPM", "twoPA", "twoP_per", "TPM", "TPA", "TP_per", 
    "blk_per", "stl_per", "ftr", "yr", "ht", "num", "porpag", "adjoe", "pfr", "year", "pid", "type", 
    "Rec Rank", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
    "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade", 
    "dunksmade/(dunksmade+dunksmiss)", "pick", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", 
    "dbpm", "gbpm", "mp", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "role", 
    "3p/100?"
]

# --- Filter Illinois top 8 players ---
illinois_players = stats[(stats['team'] == TEAM_NAME) & (stats['Min_per'] >= 50)]
illinois_players = illinois_players.sort_values(by='Min_per', ascending=False).dropna(subset=PLAYER_FEATURES)

illinois_top = illinois_players[PLAYER_FEATURES].values[:MAX_PLAYERS]
if illinois_top.shape[0] < MAX_PLAYERS:
    pad = np.zeros((MAX_PLAYERS - illinois_top.shape[0], len(PLAYER_FEATURES)))
    illinois_top = np.vstack([illinois_top, pad])

# --- Filter eligible portal players (non-freshmen, Min_per > 50, not Illinois) ---
portal_candidates = stats[
    (stats['yr'].str.lower() != 'fr') & 
    (stats['Min_per'] >= 70) & 
    (stats['team'] != TEAM_NAME)
]

portal_candidates = portal_candidates.dropna(subset=PLAYER_FEATURES)
top_portal = portal_candidates.sort_values(by='Min_per', ascending=False).head(50)

# --- Try each candidate by adding to Illinois's top 8 ---
results = []

for _, portal_player in top_portal.iterrows():
    candidate_stats = portal_player[PLAYER_FEATURES].values.reshape(1, -1)

    # Replace the last player in Illinois's 8 with this portal player
    modified_team = np.vstack([
        illinois_top[:MAX_PLAYERS - 1],
        candidate_stats
    ])

    input_vector = modified_team.flatten().astype(np.float32).reshape(1, -1)

    if np.isnan(input_vector).any():
        continue  # skip players with bad stats

    prediction = model.predict(input_vector, verbose=0)[0][0]

    results.append({
        "player_name": portal_player["player_name"],
        "team": portal_player["team"],
        "year": portal_player["yr"],
        "pts": portal_player["pts"],
        "predicted_conf_win_pct": prediction
    })

# --- Sort by predicted Conf Win% and get top 5 ---
top_recommendations = sorted(results, key=lambda x: x["predicted_conf_win_pct"], reverse=True)[:5]

print("\nTop 5 Fits for Illinois (Boost Conf Win%):")
for i, player in enumerate(top_recommendations, 1):
    print(f"{i}. {player['player_name']} ({player['team']}, {player['year']}) — "
          f"pts: {player['pts']:.1f}, Pred Conf Win%: {player['predicted_conf_win_pct']:.3f}")
