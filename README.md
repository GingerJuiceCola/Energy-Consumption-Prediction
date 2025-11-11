# Energy-Consumption-Prediction
CDS535 group work
# Energy Consumption Prediction for Data Servers
A comparison of Random Forest and Linear Regression models for data server energy consumption prediction.

## Project Overview
This project focuses on predicting the energy consumption of data servers. We compared two popular machine learning models: **Random Forest Regression** and **Linear Regression**. The results show that Random Forest performs significantly better in terms of prediction accuracy and robustness.

## Dataset Information
### 1. Main Energy Consumption Dataset
The core dataset comes from the [Data Server Energy Consumption Dataset](https://ieee-dataport.org/open-access/data-server-energy-consumption-dataset) on IEEE DataPort.
- Collected from an HP Z440 workstation over 245 days (35 weeks)
- Sampling rate: 1 value per second
- Includes variables like voltage (V), current (A), active power (W), CPU/GPU consumption (%), temperatures (°C), and more
- Includes "1mayo - agosto 2021.csv" and "2agosto -dic 2021.csv"

### 2. Weather Data
Weather data (e.g., temperature, humidity) was obtained from the public API of [open-meteo.com](https://open-meteo.com/). It corresponds to the location of the data server for accurate correlation analysis.

### 3. Data Processing Steps
We processed the raw data through the following steps to get the final dataset:
1. Run `combine.py`:
   - Convert the original second-level energy data to **hourly average values**
   - Remove 3 less useful attributes: "MAC", "weekday", "fecha_esp32"
   - Combine the processed energy data with matching weather data
   - Output: `finaldata1.csv`
2. Run `CleanMissing.py`: Handle missing values in the dataset
   - Output: `finaldata1_nomissing.csv`
3. Run `No0℃.py`: Remove samples with 0°C temperature (as needed for the task)
   - Output: `finaldata1_no0temp.csv`

## Model Comparison
We trained and tested both models using the processed dataset:
| Model               | Key Finding                  |
|---------------------|------------------------------|
| Random Forest Regression | More accurate and robust. It explains ~98% of the variance in energy consumption and has much smaller prediction errors. |
| Linear Regression   | Less accurate. It only explains ~81% of the variance, as it cannot capture nonlinear relationships between features (e.g., CPU load vs. energy use). |

<img width="660" height="447" alt="image" src="https://github.com/user-attachments/assets/dfba956a-9df6-4981-9771-bccd5b4ec520" />
