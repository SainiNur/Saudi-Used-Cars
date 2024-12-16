# Data Analysis & Modeling [Saudi Used Cars]

This project aims to analyze the price of used cars in Saudi by considering price rate, car type, car age, car mileage, car fuel type, car transmission type, and car brand. The model will be trained to predict used car prices based on these features. Modeling becomes more accurate using several techniques, such as feature scaling, selection, and cross-validation. Ultimately, serving a good model will help businesses improve operational performance by increasing efficiency and cost reduction.

## Requirements:

- Python >= 3

## Installation

1. Clone or download de repository:

```
git clone https://github.com/SainiNur/Capstone-3-Saudi-Used-Cars.git
```

1. Open the console inside the project directory and create a virtual environment (You can skip this step if you have docker installed).

```git bash
python -m venv venv
source venv/Scripts/activate
```

3. Install the app

```git bash
(venv) pip install -r requirements.txt
```

## Run Streamlit App

streamlit run dashboard/dashboard.py

📦capstone 3
 ┣ 📂dashboard
 ┃ ┗ 📜dashboard.py
 ┣ 📂dataset
 ┃ ┣ 📜cleaned_data.csv
 ┃ ┣ 📜data_saudi_used_cars.csv
 ┃ ┗ 📜Saudi Arabia Used Cars.docx
 ┣ 📂GCP_file
 ┃ ┣ 📜GCP_file.ipynb
 ┃ ┗ 📜trial_bigq.json
 ┣ 📂notebook
 ┃ ┣ 📜eda_and_visualization.ipynb
 ┃ ┣ 📜model.ipynb
 ┃ ┗ 📜Saudi-Used-Cars-XGB-ML-Regression-Model.pkl
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┗ 📜requirements.txt
