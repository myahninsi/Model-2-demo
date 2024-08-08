import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os
import yfinance as yf
import ta
import time

# Load company data
company_data_path = "final_v2.csv"  # Adjust the path as needed

if not os.path.exists(company_data_path):
    st.error(f"The file {company_data_path} does not exist. Please ensure the file is uploaded correctly.")
else:
    company_data = pd.read_csv(company_data_path)

    # Remove duplicates and NaN values
    company_data.dropna(subset=['Sector', 'Company'], inplace=True)
    company_data = company_data.drop_duplicates(subset=['Company'])

    # Streamlit app title
    st.title("Stock Trend Prediction Model")

    # Allow users to filter by sector and select a company
    sector = st.selectbox("Select Sector", company_data['Sector'].dropna().unique())
    companies_in_sector = company_data[company_data['Sector'] == sector]['Company'].unique()
    ticker = st.selectbox("Select Company", companies_in_sector)

    # Allow users to select technical indicators
    technical_indicators = ['EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
    selected_indicators = st.multiselect("Select technical indicators to include:", technical_indicators, default=technical_indicators)

    if st.button('Predict'):
        # Load the stock data
        @st.cache_data
        def load_data(ticker):
            try:
                data = yf.download(ticker, period='5y', progress=False)
                data.reset_index(inplace=True)
                data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
                data['EMA_200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()
                data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
                macd = ta.trend.MACD(data['Close'])
                data['MACD'] = macd.macd()
                data['MACD_Signal'] = macd.macd_signal()
                data['MACD_Hist'] = macd.macd_diff()
                data['Close_1_days_ahead'] = data['Close'].shift(-1)
                data.dropna(inplace=True)  # Drop rows with NaN values
                return data
            except Exception as e:
                st.error(f"Error loading data for ticker {ticker}: {e}")
                return pd.DataFrame()

        start_time = time.time()
        data = load_data(ticker)
        end_time = time.time()
        st.write(f"Data loaded in {end_time - start_time:.2f} seconds.")

        # Check if data is loaded correctly
        if data.empty:
            st.error("No data available for the selected ticker.")
        else:
            st.write(f"Data loaded for ticker {ticker}:")
            st.write(data.head())

            # Create a target variable for classification
            def classify_target(row):
                change = (row['Close_1_days_ahead'] - row['Close']) / row['Close']
                if change > 0.02:
                    return 1  # Up
                elif change < -0.02:
                    return 0  # Down
                else:
                    return 2  # Neutral

            data['Target'] = data.apply(classify_target, axis=1)
            data = data[:-1]  # Remove the last row with NaN target

            # Features and target
            X = data[selected_indicators]
            y = data['Target']

            # Handle missing values by filling with mean
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Define classifiers to compare for best model selection
            classifiers = {
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000, class_weight='balanced'),
                'SVM': SVC(random_state=42, class_weight='balanced'),
                'XGBoost': xgb.XGBClassifier(random_state=42, scale_pos_weight=1)
            }

            # Use TimeSeriesSplit for time series cross-validation
            tscv = TimeSeriesSplit(n_splits=2)  # Reduced the number of splits for faster results

            # Compare classifiers using time series cross-validation
            best_classifier = None
            best_score = 0
            results = {}

            for name, clf in classifiers.items():
                scores = []
                for train_index, test_index in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Check for NaN values and handle them
                    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
                        st.error("NaN values detected in training data. Please ensure there are no missing values.")
                        break

                    # Ensure there are enough samples for SMOTE
                    if len(np.unique(y_train)) < 2:
                        st.warning(f"Not enough samples to apply SMOTE for class {np.unique(y_train)}")
                        continue

                    # Oversample the minority classes (0 and 1) in the training set
                    smote = SMOTE(random_state=42, k_neighbors=min(2, len(y_train) - 1))
                    try:
                        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    except ValueError as e:
                        st.error(f"Error during SMOTE resampling: {e}")
                        continue

                    clf.fit(X_train_resampled, y_train_resampled)
                    y_pred = clf.predict(X_test)
                    scores.append(accuracy_score(y_test, y_pred))

                avg_score = np.mean(scores)
                results[name] = avg_score
                if avg_score > best_score:
                    best_score = avg_score
                    best_classifier = clf

            # Display results
            st.write("### Models & Accuracies")
            for name, score in results.items():
                st.write(f"{name}: {score:.4f}")

            st.write(f"\nBest Classifier: {best_classifier.__class__.__name__} with score: {best_score:.4f}")

            # Display confusion matrix and classification report for the best classifier
            y_pred_best = best_classifier.predict(X_scaled)
            st.write("### Classification Report")
            st.text(classification_report(y, y_pred_best))
            st.write("### Confusion Matrix")
            confusion_mtx = confusion_matrix(y, y_pred_best)
            sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=['Down', 'Up', 'Neutral'], yticklabels=['Down', 'Up', 'Neutral'])
            st.pyplot(plt)

            # Accuracy Score
            accuracy = accuracy_score(y, y_pred_best)
            st.write(f"Accuracy Score: {accuracy:.4f}")

