import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Title of the app
st.title("Malaria Cases Prediction")

# Sidebar for model and developer descriptions
st.sidebar.header("About the Model")
st.sidebar.write("""
This application uses the Seasonal Autoregressive Integrated Moving Average (SARIMA) model to forecast positive malaria cases based on historical data.
The SARIMA model is well-suited for time series data that exhibit trends and seasonality, making it a powerful tool for predicting disease outbreaks.
""")

st.sidebar.header("Developers")
st.sidebar.write("""
- **ANNRITA MUKAMI**
- **MARQULINE OPIYO**
""")

# File upload for the dataset
uploaded_file = st.file_uploader("/content/KISUMU COUNTY MALARIA CASES.csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Check and trim whitespace from column names
    data.columns = data.columns.str.strip()

    # Add a year to the 'PERIOD NAME' to create a complete date
    data['PERIOD NAME'] = data['PERIOD NAME'] + '-2022'  # Change year as needed

    # Convert 'PERIOD NAME' to datetime
    data['date'] = pd.to_datetime(data['PERIOD NAME'], format='%d-%b-%Y')
    data.set_index('date', inplace=True)

    # Aggregate positive cases by month
    monthly_data = data.resample('M').sum()['POSITIVE MALARIA CASES']

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Forecasting", "Model Evaluation", "Predict"])

    # Tab 1: Forecasting
    with tab1:
        st.subheader("Monthly Positive Malaria Cases")
        st.line_chart(monthly_data)

        # Fit the SARIMA model
        st.subheader("Model Fitting")
        order = st.text_input("Enter the order (p,d,q):", "1,1,1")
        seasonal_order = st.text_input("Enter the seasonal order (P,D,Q,S):", "1,1,1,12")

        if st.button("Fit and Save Model"):
            # Convert input to tuples
            order = tuple(map(int, order.split(',')))
            seasonal_order = tuple(map(int, seasonal_order.split(',')))

            # Fit the model
            model = sm.tsa.SARIMAX(monthly_data, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit()

            # Save the model
            joblib.dump(model_fit, 'sarima_model.joblib')
            st.success("Model saved as 'sarima_model.joblib'")

            # Forecast for the next 10 years (120 months)
            forecast = model_fit.get_forecast(steps=120)
            forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.offsets.MonthBegin(1), periods=120, freq='M')

            # Get forecast values and confidence intervals
            forecast_values = forecast.predicted_mean
            confidence_intervals = forecast.conf_int()

            # Plot forecast results
            plt.figure(figsize=(12, 6))
            plt.plot(monthly_data, label='Historical Data')
            plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
            plt.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.5)
            plt.title('10-Year Forecast of Positive Malaria Cases')
            plt.xlabel('Date')
            plt.ylabel('Positive Cases')
            plt.legend()
            st.pyplot(plt)

    # Tab 2: Model Evaluation
    with tab2:
        st.subheader("Model Performance Metrics")

        if st.button("Evaluate Model"):
            # Load the saved model
            model_fit = joblib.load('sarima_model.joblib')

            # Split data for evaluation
            train_size = int(len(monthly_data) * 0.8)
            train, test = monthly_data[:train_size], monthly_data[train_size:]

            # Refit model on training data for evaluation
            model = sm.tsa.SARIMAX(train, order=model_fit.specification['order'], seasonal_order=model_fit.specification['seasonal_order'])
            model_fit = model.fit()
            forecast_test = model_fit.get_forecast(steps=len(test))
            forecast_test_values = forecast_test.predicted_mean

            # Calculate performance metrics
            mae = mean_absolute_error(test, forecast_test_values)
            rmse = np.sqrt(mean_squared_error(test, forecast_test_values))
            mape = np.mean(np.abs((test - forecast_test_values) / test)) * 100

            st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
            st.write(f'Root Mean Square Error (RMSE): {rmse:.2f}')
            st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Tab 3: Predict
    with tab3:
        st.subheader("Make Predictions")
        prediction_steps = st.number_input("Enter number of months to predict:", min_value=1, max_value=120, value=12)

        if st.button("Predict"):
            # Load the saved model
            model_fit = joblib.load('sarima_model.joblib')

            # Forecast for the specified number of months
            forecast = model_fit.get_forecast(steps=prediction_steps)
            forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.offsets.MonthBegin(1), periods=prediction_steps, freq='M')

            # Get forecast values and confidence intervals
            forecast_values = forecast.predicted_mean
            confidence_intervals = forecast.conf_int()

            # Display forecast results
            st.write("Forecasted Positive Malaria Cases:")
            forecast_df = pd.DataFrame({
                'Date': forecast_index,
                'Forecast': forecast_values,
                'Lower CI': confidence_intervals.iloc[:, 0],
                'Upper CI': confidence_intervals.iloc[:, 1]
            })
            st.dataframe(forecast_df)

            # Plot forecast results
            plt.figure(figsize=(12, 6))
            plt.plot(monthly_data, label='Historical Data')
            plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
            plt.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.5)
            plt.title(f'Forecast for Next {prediction_steps} Months')
            plt.xlabel('Date')
            plt.ylabel('Positive Cases')
            plt.legend()
            st.pyplot(plt)

# Footer for additional information
st.sidebar.header("Additional Information")
st.sidebar.write("""
For any issues or feedback regarding this application, please reach out to the developers listed above.
""")
