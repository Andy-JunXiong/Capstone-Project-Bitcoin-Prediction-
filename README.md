# Bitcoin-predictions-and-visualisations-capstone

*This repo was forked from my University of Sydney's repo https://github.sydney.edu.au/mili2539/capstone-btc-group-23-2*

#### UPDATE: I trained a Multilayer Neural Network classifier to predict the direction of change in 15-minute Bitcoin Price which achieved 94% accuracy ####

This project analyses the time-series of Bitcoin process and macro-economic factors that influence Bitcoin Price using data from 2011 to 2018. The datasets contain Bitcoin trading data (high,low,open,close,volume,marketcap) and 33 other features inlcluding sentiment (Google Trends), currency exchange rates, stock and commodity prices, internal Blockchain data, Bitcoin price technical indicators.

Relevant features are then then applied to train and compare the performance of the 14 models on predicting the latest Bitcoin Price in 2018 using Python libraries: 

Machine Learning (Scikit-learn, XGBoost): 
* Parametric: OLS Linear Regression, Lasso, Ridge, Bayesian Ridge, ElasticNet
* Non-parametric: KNN, Support Vector Regression, Gradient Boosting Trees, Extremely Randomised Trees, Decision Tree

Deep Learning (Keras): 
* Recurrent Neural Networks: LSTM, GRU 
* Multilayer Perceptron Network
    
Time-Series (Statsmodels): 
* Autoregressive Integrated Moving Average model 
    
Visualisations for predicted Bitcoin Price and model comparison were developed using Python's plotly, D3.js, Tableu and deployed to AWS S3 Buckets at https://s3-ap-southeast-2.amazonaws.com/capstone-bitcoin/prices/index.html#actualprice

