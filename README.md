## 📥 Setup

1. Download the datasets:

- Home Credit Default Risk Dataset:  Download application_train.csv 
  https://www.kaggle.com/competitions/home-credit-default-risk/data

- German Credit Dataset:  
  https://www.kaggle.com/datasets/uciml/german-credit

2. Place the CSV files inside the `data/` folder:

```
drift-oracle/
│
├── data/
│   ├── application_train.csv
│   ├── german_data.csv
├── data_preprocess.py
├── train_model.py
├── drift_detection.py
├── german_credit.py
├── .gitignore
```

## 🚀 Execution Order

Run the following scripts in order:

1. Data Preprocessing  
```
python data_preprocess.py
```

2. Train Models  
```
python train_model.py
```

3. Drift Detection (PSI)  
```
python drift_detection.py
```

4. Evaluation & Retraining  
```
python german_credit.py
```
