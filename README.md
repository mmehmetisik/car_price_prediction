# 🚗 Car Price Prediction System

This project is a **machine learning-based web application** that predicts the **market value of used cars** based on user input. Built with LightGBM and Streamlit, it provides accurate predictions based on real market data.

🔗 Live Demo: [Car Price Predictor App](https://miuul-car-price-predictor.streamlit.app/)

## 📊 Overview

A robust machine learning system that analyzes 50+ vehicle features to provide instant and accurate car price predictions. The application offers a user-friendly interface for easy interaction.

## ⭐ Features

* Real-time price predictions based on market data
* User-friendly web interface
* Analysis of 50+ vehicle features
* Detailed result reporting
* Advanced ML model (LightGBM)
* Instant car price calculation
* Comprehensive vehicle feature analysis

## 💻 Technologies Used

* **Python** - Core programming language
* **Streamlit** - Web framework for interactive UI
* **Scikit-learn** - Model training & preprocessing
* **LightGBM, CatBoost** - Advanced regression models
* **Pandas & NumPy** - Data processing & transformation
* **Pickle** - Model & feature storage

## 🔧 Installation & Setup

### Download Required Model Files
The following files must be present in the `models/` directory:
```bash
models/
├── final_model.pkl    # Trained ML Model
├── feature_columns.pkl # Feature columns used during training
└── scaler.pkl        # Scaler for normalizing input data
```
# Installation Steps
# Clone the repository
git clone https://github.com/your-username/car_price_prediction.git

# Navigate to project directory
cd car_price_prediction

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run app.py

📂 Project Structure
```bash
Car Price Prediction
├── models/
│   ├── final_model.pkl        # Trained ML model
│   ├── feature_columns.pkl    # Feature set used for training
│   └── scaler.pkl            # Scaler for numerical features
├── data/
│   └── USA_cars_datasets.csv  # Original dataset
├── app.py                     # Streamlit web application
├── car_pred.py               # Machine Learning model training script
├── requirements.txt          # Required Python packages
└── README.md                # Documentation
```

🎯 How to Use

Enter vehicle details (brand, model year, mileage, color, state, vehicle condition, etc.)
Click the "Predict Price" button
View the predicted car price instantly! 🚗 💰

# 📚 Dataset
The model is trained on the USA Cars Dataset, featuring:

2,500+ car records
Comprehensive vehicle information
Real market data

# 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
# 🤝 Contributing & Contact
Interested in contributing? Follow these steps:

# Fork the repository
Create a new branch (feature-addition)
Make your changes and commit them
Submit a Pull Request (PR) for review
