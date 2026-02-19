# 🎾 Tennis Play Predictor

A machine learning-based application that predicts whether weather conditions are suitable for playing tennis. This project utilizes the **Naive Bayes** algorithm to analyze various weather parameters and provide a recommendation.

## 📂 Project Structure

The project is organized into two main directories:

- **`Project/`**: Contains the complete source code, dataset, and machine learning models.
    - `app.py`: The main Streamlit application for the user interface.
    - `train_model.py`: Script to train the Naive Bayes model.
    - `generate_dataset.py`: Script used to generate the dataset.
    - `play_tennis_2400.csv`: The dataset containing 2400 weather records.
    - `requirements.txt`: List of Python dependencies.
    - `*.pkl`: Serialized model and encoder files.

- **`Presentation/`**: Contains documentation and presentation materials.
    - `Naive Baye's Algarithom.pdf`: A detailed presentation on the algorithm and project concepts.

## 🧠 Algorithm: Naive Bayes

This project implements the **Naive Bayes Classifier**, a probabilistic machine learning model based on Bayes' Theorem. It is particularly effective for classification tasks.

**How it works:**
The model calculates the probability of an event (Playing Tennis: Yes/No) based on prior knowledge of conditions that might be related to the event (Outlook, Temperature, Humidity, Wind). It assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature (hence "Naive").

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### 2. Installation
Navigate to the `Project` directory and install the required dependencies:

```bash
cd Project
pip install -r requirements.txt
```

### 3. Training the Model (Optional)
If you want to retrain the model with the dataset:

```bash
python train_model.py
```
This will generate new `tennis_model.pkl` and `encoders.pkl` files.

### 4. Running the Application
Launch the web interface using Streamlit:

```bash
streamlit run app.py
```

The application will open in your default web browser. You can then select the weather conditions (Outlook, Temperature, Humidity, Wind) and click **PREDICT NOW** to see the result.

## 👥 Team Members

- **Partha Sarathi R**
- **Ayyapparaja VJ**
- **Thirupathi Yaswanth**

---
*Class of 2026 - Machine Learning Project*
