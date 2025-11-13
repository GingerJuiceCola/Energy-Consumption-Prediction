# Energy-Consumption-Prediction  
CDS535 Group Work  

## Energy Consumption Prediction for Data Servers  
A comparison of three models (Linear Regression, Random Forest, XGBoost) for predicting data server energy use.  

## Project Overview  
We tried to predict energy consumption of HP Z440 workstations. And we compared three models, and XGBoost performed a bit better than the others by using data like hardware status (CPU/GPU load/temp), power parameters (voltage, current), and environmental info (temp, humidity).

## Dataset Information  
### 1. Data Sources  
| Data Type       | Details                                                                 |
|-----------------|-------------------------------------------------------------------------|
| Server Energy   | From IEEE DataPort (HP Z440) – includes voltage, current, CPU/GPU data |
| Weather         | From Open-Meteo API – includes temp, humidity, wind speed (matched to server time/location) |

### 2. Data Processing  
We used 3 scripts to get the final dataset (`finaldata1_no0temp.csv`):  
1. `combine.py`: Turned second-level energy data into hourly averages, removed useless columns, merged with weather data  
2. `CleanMissing.py`: Fixed missing values  
3. `No0℃.py`: Deleted samples with 0°C (likely sensor errors)  
Final data: 918 rows, 17 features  

## Model Comparison  
We split data into 80% (train) and 20% (test). 3 metrics are used to compare the performances of the three models:
- MAE/RMSE: Lower = better (smaller prediction error)
- R²: Closer to 1 = better (explains more energy use variation)

### Performance Results
| Model            | MAE    | RMSE   | R²    | Notes                                  |
|------------------|--------|--------|-------|----------------------------------------|
| XGBoost          | 3.3390 | 7.5614 | 0.9862| Slightly best – small errors, explains 98.6% variation |
| Random Forest    | 3.4330 | 8.1994 | 0.9837| Very good – close to XGBoost           |
| Linear Regression| 21.6024|27.9089 | 0.8117| Less good – bigger errors              |

## Key Takeaways
1. XGBoost and Random Forest are more suitable for this energy prediction task than Linear Regression  
2. Using clean, processed data (like hourly averages) helps improve model results  
3. Hardware and weather data together are useful for predicting server energy use
