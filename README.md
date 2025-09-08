🩺 DiaPredictPro – Diabetes Prediction System

📖 Project Overview

DiaPredictPro is an end-to-end Machine Learning project that predicts the likelihood of diabetes using healthcare data.
It features both:

🌐 Flask API – for backend model deployment

📊 Streamlit Dashboard – for interactive visualizations and predictions

This project showcases the complete ML pipeline from data preprocessing → model training → deployment in a production-ready environment.

🚀 Features

🧹 Data Preprocessing (scaling, cleaning, feature engineering)

📊 Exploratory Data Analysis (Jupyter notebooks with visualizations)

🤖 Model Training (Decision Trees, Random Forests, Logistic Regression)

🌐 Flask Web App (API endpoint for predictions)

📈 Streamlit Dashboard (user-friendly prediction interface)

📦 Dockerized Deployment (easy to run anywhere)

📂 Project Structure
.
├── Dockerfile                # Docker setup
├── docker-compose.yml        # Multi-service setup
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
│
├── app_flask/                # Flask API
│   ├── app.py
│   ├── static/style.css
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── app_streamlit/            # Streamlit dashboard
│   └── app.py
│
├── data/                     # Dataset (raw & processed)
│   └── raw/diabetes.csv
│
├── models/                   # Saved ML models
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── notebooks/                # Jupyter Notebooks
│   └── diabetes_analysis.ipynb
│
└── src/                      # Source code
    ├── data_preprocessing.py
    ├── model_training.py
    └── utils.py

⚙️ Installation & Usage
🔹 1. Clone the Repository
git clone https://github.com/Nasir54/DiaPredictPro-.git
cd DiaPredictPro-

🔹 2. Install Dependencies
pip install -r requirements.txt

🔹 3. Run Flask App
cd app_flask
gunicorn --bind 0.0.0.0:5001 --workers 4 app:app


App will be available at: 👉 http://localhost:5001

🔹 4. Run Streamlit Dashboard
cd app_streamlit
streamlit run app.py --server.port=8501


Dashboard available at: 👉 http://localhost:8501

🔹 5. Run with Docker (Recommended)
docker build -t diabetes-prediction .
docker run -p 5001:5001 -p 8501:8501 diabetes-prediction

📊 Dataset

The project uses the Pima Indians Diabetes Dataset (data/raw/diabetes.csv) – a popular dataset for medical prediction tasks.

🧠 Models Used

Decision Tree Classifier

Random Forest Classifier

Logistic Regression

StandardScaler for normalization

🌟 Future Improvements

Deploy on Heroku/AWS/GCP

Add model retraining pipeline

Implement Explainable AI (XAI) for medical insights

👨‍💻 Author

Muhammad Nasir
📌 Data Science & AI Enthusiast

⭐ Contribute & Support

If you like this project, give it a ⭐ on GitHub!
Pull requests and feedback are welcome 🙌
