# Advanced_time_series_forecasting

Advanced Time Series Forecasting with LSTM + Attention and Bayesian Optimization

This project investigates how far we can push modern neural network methods for multivariate time-series forecasting when we move beyond “standard LSTMs” and incorporate more deliberate engineering choices—specifically attention mechanisms and Bayesian hyperparameter optimization. The goal was not only to build a model that predicts well, but also to show clearly why a more advanced approach performs better than a simple baseline.

Dataset
For the experiments, I worked with a multivariate time-series dataset containing several correlated environmental variables. The dataset includes continuous sensor readings collected at regular time intervals. Since real-world time-series data often contains noise, missing values, and drift, part of the effort went into cleaning, aligning, and scaling the features. I also engineered additional lagged features and simple rolling statistics to give the models richer temporal context.

Baseline Models
Before experimenting with complex neural networks, I built two baselines:

Persistence Model – predicts that the next value equals the last observed value.

Standard LSTM – a single-layer LSTM without heavy tuning.

ARIMA (univariate) – used mainly as a point of comparison for the primary target variable.

These models helped establish a realistic performance floor. The persistence model performed better than expected (as it often does in short-range forecasting), which made it a meaningful reference.

Advanced Model Architecture
The main model combines a Bidirectional LSTM encoder with a custom attention layer. The reason for adding attention was straightforward: plain LSTMs compress an entire sequence into a single state, which can cause the model to “forget” important time steps. Attention allows the network to assign different weights to different points in the input window, effectively letting it focus on the most relevant parts of the history.

Hyperparameter Optimization
Instead of manual guessing or grid search, I used Optuna for Bayesian hyperparameter optimization. The search space included:

Hidden units

Dropout

Learning rate

Batch size

Sequence length

Number of LSTM layers

Optuna’s pruning mechanism helped stop bad trials early, which kept the overall tuning process efficient. The optimization phase revealed that the model was very sensitive to dropout and learning rate, and far less sensitive to network depth than I initially expected.

Evaluation
Evaluation was performed on a held-out test set using:

RMSE

MAE

In addition to raw metrics, I also checked the distribution of residuals and inspected attention weights to understand what parts of the sequence the model relied on. This helped verify that the model wasn't overfitting or simply memorizing patterns.

Results
The optimized LSTM-Attention model consistently outperformed all baselines. The improvement over the naive LSTM was especially clear once the best hyperparameters were applied. The persistence model remained surprisingly competitive for very short-range predictions, but the attention-based model pulled ahead as soon as the forecast task depended on longer temporal dependencies.

Practical Takeaways
Attention noticeably improves the model’s ability to leverage longer historical patterns.

Hyperparameter tuning is essential; the “default LSTM” performs significantly worse.

Even simple baselines like persistence and ARIMA are important because they highlight whether the neural network is genuinely learning patterns rather than noise.

Feature engineering (lags, rolling stats) contributed more than I initially expected.

The best model ended up being only moderately complex; quality tuning mattered more than stacking layers.

Project Deliverables
Fully modular, documented Python code for:

Data loading and preprocessing

Baseline methods

Advanced LSTM-Attention model

Optuna optimization

Evaluation and reporting

Final report summarizing dataset characteristics, modeling decisions, hyperparameter search results, and metric comparisons

A saved record of the best hyperparameters discovered by Optuna
