# CollegeBasketball-ML

## Model Metrics
MSE: 0.0235 <br />
R² Score: 0.4399

## Write Up
To help Basketball Programs during the increasingly complex transfer portal process, I built a prototype machine learning model designed to identify high-impact D1 transfer prospects. The model predicts the expected conference win rate for any lineup configuration, helping staff identify the top players to recruit from the transfer portal. 

I developed the model in Python using Pandas, NumPy, and TensorFlow. The input data came from BartTorvik’s 2024–2025 player statistics, which I preprocessed by manually labeling columns, filtering out low-minutes-per-game players, and grouping athletes by team. I then joined this with a separate dataset mapping each team to its actual conference win rate, using that as the regression target.

To reflect actual game impact, I limited each team’s player vector to its top 8 players by minutes played. Each player’s features such as offensive/defensive efficiency, usage rate, and “KenPom” metrics were used to train the model.

The model architecture included 7 fully connected layers with ReLU activations, followed by a single-node sigmoid output layer to produce a win-rate prediction between 0-1. Early runs showed high variance and poor generalization (MSE and R²), meaning the model was overfitting.  To fix this, I introduced a dropout layer and increased batch size to improve regularization.

Though developed within a short timeframe, the prototype already demonstrates practical utility in streamlining recruitment analysis. With further hyperparameter tuning, regularization, and potentially better datasets, it has the ability to transform the transfer scouting process.
