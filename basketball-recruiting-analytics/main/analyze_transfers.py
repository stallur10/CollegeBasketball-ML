import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import r2_score

# pre-proccess player stats
stats = pd.read_csv("./data/player_data.csv")

stats.columns = [
    "player_name", "team", "conf", "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", 
    "AST_per", "TO_per", "FTM", "FTA", "FT_per", "twoPM", "twoPA", "twoP_per", "TPM", "TPA", "TP_per", 
    "blk_per", "stl_per", "ftr", "yr", "ht", "num", "porpag", "adjoe", "pfr", "year", "pid", "type", 
    "Rec Rank", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
    "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade", 
    "dunksmade/(dunksmade+dunksmiss)", "pick", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", 
    "dbpm", "gbpm", "mp", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "role", 
    "3p/100?"
]
df_filtered = stats[stats['Min_per'] >= 50]

# group players by team
players_by_team = df_filtered.groupby('team')

# pre-proccess team Data
team_df = pd.read_csv('./data/team_data.csv')
team_df.dropna(subset=['Conf Win%'], inplace=True)


# map should be team to Conf Win%, but csv is incorrect and shifted headers
team_conf_win_pct = dict(zip(team_df['rank'], team_df['ConSOSRemain']))

# start setting up model
player_features = [
    "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per", "blk_per", "stl_per",
    "ftr", "porpag", "adjoe", "pfr", "drtg", "adrtg", "dporpag", "bpm", "obpm", "dbpm", 
    "gbpm", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "3p/100?"
]

# config vars
MAX_PLAYERS = 8
INPUT_SIZE = MAX_PLAYERS * len(player_features)

X = []
y = []
team_names = []

# normalize

# iterate through each team and keep top players
for team_name, group in players_by_team:

    if team_name not in team_conf_win_pct:
        continue

    team_players = group.sort_values(by="Min_per", ascending=False)
    player_data = team_players[player_features].values

    if player_data.shape[0] < MAX_PLAYERS:
        pad_size = MAX_PLAYERS - player_data.shape[0]
        pad = np.zeros((pad_size, len(player_features)))
        player_data = np.vstack([player_data, pad])
    else:
        player_data = player_data[:MAX_PLAYERS]

    team_vector = player_data.flatten()
    if len(team_vector) == INPUT_SIZE:
        X.append(team_vector)
        y.append(team_conf_win_pct[team_name])
        team_names.append(team_name)

X = np.array(X)
y = np.array(y)

# split into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train model
model.fit(X_train, y_train, epochs=300, batch_size=6, validation_split=0.1, verbose=1)


### OUTPUT METRICS AND PREDICTIONS ###
predictions = model.predict(X_test).flatten()

for i in range(len(predictions)):
    print(f"Predicted: {predictions[i]:.3f}, Actual: {y_test[i]:.3f}")

# metrics
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss:.4f}")

r2 = r2_score(y_test, predictions)
print(f"R² Score (accuracy): {r2:.4f}")

# save Model
if r2 > 0.3:
    save_dir = './models/predict-conf-wins'
    model.save(save_dir)
    print(f"Model saved to {save_dir}")








    