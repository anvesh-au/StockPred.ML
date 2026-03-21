# StockPred.ML

StockPred.ML is an exploratory project focused on understanding how different deep learning architectures can be applied to stock price time-series prediction.

The goal of this repository is not only to build predictive models, but also to experiment with and study how different neural network architectures learn patterns from financial time-series data.

The project explores several sequence-based neural networks and generative models to analyze their effectiveness in modeling stock price movements.

---

# Learning Objectives

This project was built to explore:

- How stock price data can be prepared for machine learning
- How sequential neural networks process financial time-series data
- Differences between RNN, LSTM, and GRU architectures
- How generative models like GANs can learn market data distributions
- Practical challenges in stock prediction using deep learning

---

# Prediction Pipeline

The prediction workflow in this project follows four main stages:
Data Sourcing → Feature Engineering → Model Training → Evaluation

Each stage helps transform raw stock data into meaningful predictions.

---

# 1. Data Sourcing

Handled in: data_creation.ipynb

This stage prepares the dataset used for training the models.

Key steps include:

- Creating or loading historical stock price data
- Structuring data as time-series
- Normalizing the data
- Preparing training and testing splits

Why this matters:

Stock prices are sequential and noisy. Proper preparation ensures models receive structured inputs that allow them to learn meaningful temporal patterns.

---

# 2. Feature Engineering

Time-series models cannot directly learn from raw price values without proper formatting.

Feature engineering converts the stock price data into sequences that neural networks can process.

Typical transformation:
Input : [p1, p2, p3, ..., pn]
Target : next price

A sliding window approach is used where past prices are used to predict future values.

This allows models to learn:

- short-term trends
- momentum patterns
- temporal dependencies

---

# 3. Models Used

This repository experiments with the following neural network architectures:

- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- GAN (Generative Adversarial Network)

Each model is implemented in its own notebook to allow independent experimentation and learning.

---

# 3.1 RNN (Recurrent Neural Network)

File: rnn.ipynb

RNNs are one of the earliest neural network architectures designed for sequential data.

They process inputs step-by-step while maintaining a hidden state that captures information from previous time steps.

Why RNNs are useful for stock prediction:

- Stock prices depend on previous prices
- Sequential processing captures temporal relationships

However, standard RNNs struggle with long-term dependencies due to the **vanishing gradient problem**.

---

# 3.2 LSTM (Long Short-Term Memory)

File: lstm.ipynb

LSTMs are an improved form of RNN designed to handle long-term dependencies in sequential data.

They introduce **gating mechanisms** that control what information is remembered or forgotten.

Advantages for stock prediction:

- Can capture long-term market trends
- More stable training than basic RNNs
- Effective for complex time-series patterns

Because financial markets often show long-term dependencies, LSTMs are widely used in financial forecasting tasks.

---

# 3.3 GRU (Gated Recurrent Unit)

File: gru.ipynb

GRUs are a simplified version of LSTMs.

They combine certain gating mechanisms, reducing the number of parameters while maintaining strong performance.

Benefits:

- Faster training
- Less computational overhead
- Often similar performance to LSTMs

GRUs are useful when computational efficiency is important while still modeling sequential dependencies.

---

# 3.4 GAN (Generative Adversarial Network)

File: gan.ipynb

GANs consist of two competing neural networks:

- Generator  
- Discriminator

The generator tries to create realistic sequences of stock price data, while the discriminator attempts to distinguish real data from generated data.

Through this adversarial process, the generator learns the underlying data distribution.

Why GANs are interesting for financial data:

- Financial markets are stochastic and complex
- GANs can learn realistic data distributions
- They can generate synthetic market scenarios for experimentation

This makes GANs useful for exploring how financial time-series behave.

---

# 4. Evaluation

After training, model predictions are compared against actual stock prices.

Evaluation typically includes:

- Comparing predicted vs actual values
- Monitoring training loss
- Visualizing predictions

The goal is to understand:

- how well each architecture captures market behavior
- how different models learn time-series patterns

Since financial markets are inherently noisy, evaluation focuses on understanding model behavior rather than expecting perfect prediction accuracy.

| Model | Split | Train(R2/RMSE) | Test(R2/RMSE) |
|:---|:---:|---:|---:|
| RNN | 50:50 | 0.93/0.04 | 0.61/0.27 | |
| RNN | 80:20 | na | na | |
| GRU | 50:50 | 0.94/0.04 | 0.81/0.19 | |
| GRU | 80:20 | 0.93/0.035 | 0.65/0.35 | |
| LSTM | 50:50 | 0.92/0.05 | 0.73/0.26 | |
| LSTM | 80:20 | 0.98/0.02 | 0.56/0.37 | |
| GAN | 50:50 | 0.97/0.03 | 0.8/0.2 | |
| GAN | 80:20 | 0.88/0.048 | 0.79/0.26 | |

---

# Project Philosophy

This project is primarily designed for:

- learning deep learning for time-series
- experimenting with financial data
- understanding sequence models
- exploring generative modeling

Rather than building a production trading system, the repository focuses on **education, experimentation, and exploration of machine learning techniques in finance**.

---

# Possible Future Extensions

Future improvements could include:

- Transformer-based time-series models
- Attention mechanisms
- Volatility prediction
- Backtesting strategies
- Multi-asset prediction
- Reinforcement learning for trading
