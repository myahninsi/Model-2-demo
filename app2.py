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
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os
import yfinance as yf
import ta

# Load company data
company_data_path = r"final_v2.csv"  # Use relative path

if not os.path.exists(company_data_path):
    st.error(f"The file {company_data_path} does not exist. Please ensure the file is uploaded correctly.")
else:
    company_data = pd.read_csv(company_data_path)

    # Streamlit app title
    st.title("Stock Trend Prediction Model")

    # Allow users to filter by sector and select a company
    sector = st.selectbox("Select Sector", company_data['Sector'].unique())
    companies_in_sector = company_data[company_data['Sector'] == sector]['Company']
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

        data = load_data(ticker)

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

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Check class distribution
            st.write("Class distribution before resampling:")
            st.write(y.value_counts())

            # Define classifiers
            classifiers = {
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000, class_weight='balanced'),
                'SVM': SVC(random_state=42, class_weight='balanced'),
                'XGBoost': xgb.XGBClassifier(random_state=42, scale_pos_weight=1)  # Adjust scale_pos_weight as needed
            }

            # Use TimeSeriesSplit for time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Oversample the minority classes in the training set
            oversampler = SMOTE(random_state=42, k_neighbors=2)  # Reduce k_neighbors to handle small class sizes

            # Compare classifiers using time series cross-validation
            best_score = 0
            results = {}

            for name, clf in classifiers.items():
                scores = []
                for train_index, test_index in tscv.split(X_scaled):
                    # Convert train and test indices to lists
                    train_index = list(train_index)
                    test_index = list(test_index)

                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

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

            # Train the best classifier on the full training set
            train_index, test_index = list(tscv.split(X_scaled))[-1]
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

            best_classifier.fit(X_train_resampled, y_train_resampled)

            # Predict the model on the last split
            y_pred = best_classifier.predict(X_test)

            # Evaluate the model
            classification_report_result = classification_report(y_test, y_pred, output_dict=True)
            confusion_matrix_result = confusion_matrix(y_test, y_pred)
            accuracy_result = accuracy_score(y_test, y_pred)

            # Convert classification report to DataFrame
            classification_report_df = pd.DataFrame(classification_report_result).transpose()

            # Print the evaluation metrics
            st.write("### Classification Report")
            st.table(classification_report_df)
            st.write("### Confusion Matrix")
            st.write(confusion_matrix_result)
            st.write(f"### Accuracy Score: {accuracy_result:.4f}")

            # Visualize the confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up', 'Neutral'], yticklabels=['Down', 'Up', 'Neutral'], ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(fig)

            # Make predictions for the entire dataset
            data['Predicted'] = best_classifier.predict(scaler.transform(data[selected_indicators]))

            st.write("### Data with Predictions")
            st.write(data[['Close', 'Target', 'Predicted']].tail(15))
