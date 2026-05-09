import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bin.Analyse.Traffic import Traffic
from bin.Analyse.Weather import Weather
from bin.Analyse.Sales import Sales


st.set_page_config(page_title="Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")


def _infer_freq(index):
    """Infer a safe pandas frequency for future date generation."""
    idx = pd.to_datetime(pd.Index(index)).sort_values()
    if len(idx) >= 3:
        guessed = pd.infer_freq(idx)
        if guessed:
            return guessed
    if len(idx) >= 2:
        delta = idx.to_series().diff().dropna().median()
        if pd.notna(delta) and delta > pd.Timedelta(0):
            return delta
    return "D"


def _make_forecast_features(length, seasonal_period=7):
    """Create trend + seasonality features for a small NumPy ML regression model."""
    t = np.arange(length, dtype=float)
    scale = max(length - 1, 1)
    x = t / scale

    features = [
        np.ones(length),
        x,
        x ** 2,
        np.sin(2 * np.pi * t / seasonal_period),
        np.cos(2 * np.pi * t / seasonal_period),
    ]

    # Add longer seasonality when there is enough data.
    if length >= 24:
        long_period = min(30, max(12, seasonal_period * 4))
        features.extend([
            np.sin(2 * np.pi * t / long_period),
            np.cos(2 * np.pi * t / long_period),
        ])

    return np.column_stack(features)


def _fit_ml_forecast(series, forecast_steps=7, seasonal_period=7):
    """
    Forecast with NumPy-based regression using trend + seasonal features.
    This avoids the old flat/straight prediction problem for data with trend or seasonality.
    """
    clean_series = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_series) < 4:
        return None

    y = clean_series.values.astype(float)
    n = len(y)

    # Smooth noisy data slightly, but keep real movements.
    smooth_window = min(5, max(2, n // 8))
    y_train = pd.Series(y).rolling(smooth_window, min_periods=1).mean().values

    X = _make_forecast_features(n + forecast_steps, seasonal_period=seasonal_period)
    X_train = X[:n]
    X_future = X[n:]

    # Small ridge value improves stability for small datasets.
    ridge = 1e-6
    identity = np.eye(X_train.shape[1])
    identity[0, 0] = 0  # do not regularize intercept
    coef = np.linalg.pinv(X_train.T @ X_train + ridge * identity) @ X_train.T @ y_train

    fitted = X_train @ coef
    future_y = X_future @ coef

    # Add a recent momentum adjustment so future points follow the latest movement, not only the full-period average.
    recent_count = min(8, n - 1)
    if recent_count >= 2:
        recent_slope = np.polyfit(np.arange(recent_count), y[-recent_count:], 1)[0]
        model_slope = future_y[1] - future_y[0] if len(future_y) > 1 else 0
        momentum = 0.35 * (recent_slope - model_slope)
        future_y = future_y + momentum * np.arange(1, forecast_steps + 1)

    # Keep predictions realistic: non-negative for common dashboard quantities.
    if np.nanmin(y) >= 0:
        future_y = np.maximum(future_y, 0)
        fitted = np.maximum(fitted, 0)

    residual = y - fitted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {"fitted": fitted, "future_y": future_y, "rmse": rmse, "r2": r2}


def _future_dates(last_date, periods, freq):
    last_date = pd.to_datetime(last_date)
    return pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]


def _plot_recent_and_forecast(st_obj, title, historical, forecast_steps=7, freq=None, seasonal_period=7):
    historical = pd.Series(historical).sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(historical) < 4:
        st_obj.warning(f"Not enough data for {title} prediction. Need at least 4 time points.")
        return ""

    model = _fit_ml_forecast(historical, forecast_steps, seasonal_period=seasonal_period)
    if model is None:
        st_obj.warning(f"Not enough data for {title} prediction.")
        return ""

    if freq is None:
        freq = _infer_freq(historical.index)
    forecast_index = _future_dates(historical.index[-1], forecast_steps, freq)
    recent_window = min(45, len(historical))
    recent_data = historical.tail(recent_window)
    fitted_recent = pd.Series(model["fitted"], index=historical.index).tail(recent_window)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(recent_data.index, recent_data.values, marker="o", label="Recent actual trend")
    ax.plot(fitted_recent.index, fitted_recent.values, linestyle="--", label="ML fitted trend")
    ax.plot(forecast_index, model["future_y"], marker="o", linestyle="--", label="Future prediction")
    ax.set_title(title)
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    st_obj.pyplot(fig)
    plt.close(fig)

    current_value = float(historical.iloc[-1])
    predicted_value = float(model["future_y"][-1])
    change = predicted_value - current_value
    change_percent = (change / current_value * 100) if current_value != 0 else 0
    direction = "increase" if change > 0 else "decrease" if change < 0 else "remain stable"

    return (
        f"{title}: current value {current_value:.2f}; predicted value after "
        f"{forecast_steps} step(s) {predicted_value:.2f}; expected to {direction} "
        f"by {change:.2f} ({change_percent:.2f}%). RMSE: {model['rmse']:.2f}, R²: {model['r2']:.2f}.\n"
    )


def advanced_analysis_and_predictions(df, st_obj, base, selected_analysis):
    selected_analysis = list(selected_analysis)

    if not selected_analysis:
        st_obj.info("Select at least one analysis type above to show ML-based recent trend and future prediction.")
        return

    st_obj.caption(
        "ML prediction uses NumPy regression with trend, curve, seasonality, and recent momentum. "
        "It appears only for the analysis types you select."
    )

    forecast_steps = st_obj.slider("Future prediction period", min_value=3, max_value=30, value=7, step=1)
    prediction_report = ""

    try:
        if base == "Traffic":
            work = df.copy()
            work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
            work = work.dropna(subset=["DateTime"])
            work["Date"] = work["DateTime"].dt.date

            if "Traffic Trend Analysis" in selected_analysis:
                daily = work.groupby("Date")["Vehicles"].sum()
                daily.index = pd.to_datetime(daily.index)
                prediction_report += _plot_recent_and_forecast(st_obj, "Traffic Vehicles: Recent Trend + Future Prediction", daily, forecast_steps, "D", 7)

            if "Junction Comparison" in selected_analysis and "Junction" in work.columns:
                st_obj.markdown("#### Junction-wise Traffic Prediction")
                for junction, part in work.groupby("Junction"):
                    daily = part.groupby("Date")["Vehicles"].sum()
                    daily.index = pd.to_datetime(daily.index)
                    prediction_report += _plot_recent_and_forecast(st_obj, f"Junction {junction} Vehicles Forecast", daily, forecast_steps, "D", 7)

            if "Peak Hour Detection" in selected_analysis:
                hourly = work.groupby(work["DateTime"].dt.floor("H"))["Vehicles"].sum()
                prediction_report += _plot_recent_and_forecast(st_obj, "Hourly Traffic: Recent Trend + Future Prediction", hourly, forecast_steps, "H", 24)

            if "Day-wise / Weekly Analysis" in selected_analysis:
                daily = work.groupby("Date")["Vehicles"].sum()
                daily.index = pd.to_datetime(daily.index)
                prediction_report += _plot_recent_and_forecast(st_obj, "Weekly/Daily Traffic Forecast", daily, forecast_steps, "D", 7)

        elif base == "Weather":
            work = df.copy()
            work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
            work = work.dropna(subset=["DateTime"])
            work["Date"] = work["DateTime"].dt.date

            metric_map = {
                "Temperature Analysis": ("Temperature", "mean", 7),
                "Humidity Analysis": ("Humidity", "mean", 7),
                "Rainfall Analysis": ("Rainfall", "sum", 7),
                "Wind Speed Analysis": ("WindSpeed", "mean", 7),
                "Time-Based Analysis": ("Temperature", "mean", 7),
            }

            for option in selected_analysis:
                if option in metric_map:
                    column, agg, season = metric_map[option]
                    if column not in work.columns:
                        continue
                    daily = work.groupby("Date")[column].sum() if agg == "sum" else work.groupby("Date")[column].mean()
                    daily.index = pd.to_datetime(daily.index)
                    prediction_report += _plot_recent_and_forecast(st_obj, f"{column}: Recent Trend + Future Prediction", daily, forecast_steps, "D", season)

            if "City Comparison" in selected_analysis and "City" in work.columns:
                st_obj.markdown("#### City-wise Temperature Prediction")
                for city, part in work.groupby("City"):
                    daily = part.groupby("Date")["Temperature"].mean()
                    daily.index = pd.to_datetime(daily.index)
                    prediction_report += _plot_recent_and_forecast(st_obj, f"{city} Temperature Forecast", daily, forecast_steps, "D", 7)

        elif base == "Sales":
            work = df.copy()
            work.columns = work.columns.str.strip()
            work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
            work = work.dropna(subset=["Date"])

            if "Sales Revenue Analysis" in selected_analysis or "Time-Based Trends" in selected_analysis:
                daily = work.groupby("Date")["Revenue"].sum()
                prediction_report += _plot_recent_and_forecast(st_obj, "Sales Revenue: Recent Trend + Future Prediction", daily, forecast_steps, "D", 7)

            if "Product Performance" in selected_analysis and "Product" in work.columns:
                st_obj.markdown("#### Product-wise Revenue Prediction")
                for product, part in work.groupby("Product"):
                    daily = part.groupby("Date")["Revenue"].sum()
                    prediction_report += _plot_recent_and_forecast(st_obj, f"{product} Revenue Forecast", daily, forecast_steps, "D", 7)

            if "Region Analysis" in selected_analysis and "Region" in work.columns:
                st_obj.markdown("#### Region-wise Revenue Prediction")
                for region, part in work.groupby("Region"):
                    daily = part.groupby("Date")["Revenue"].sum()
                    prediction_report += _plot_recent_and_forecast(st_obj, f"{region} Region Revenue Forecast", daily, forecast_steps, "D", 7)

            if "Top & Bottom Analysis" in selected_analysis and "Product" in work.columns:
                product_sales = work.groupby("Product")["Revenue"].sum().sort_values()
                selected_products = list(product_sales.head(2).index) + list(product_sales.tail(2).index)
                st_obj.markdown("#### Top & Bottom Product Prediction")
                for product in dict.fromkeys(selected_products):
                    daily = work[work["Product"] == product].groupby("Date")["Revenue"].sum()
                    prediction_report += _plot_recent_and_forecast(st_obj, f"{product} Top/Bottom Revenue Forecast", daily, forecast_steps, "D", 7)

        else:
            st_obj.info("Advanced predictions are currently configured for Traffic, Weather, and Sales datasets.")

    except Exception as e:
        st_obj.error(f"Prediction error: {e}")

    if prediction_report.strip():
        st_obj.markdown("#### Prediction Summary")
        st_obj.success(prediction_report)
    else:
        st_obj.info("No prediction chart is available for the selected analysis type yet. Choose a trend-based option such as Traffic Trend, Temperature, Sales Revenue, Product, Region, or Time-Based Trends.")


file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

df = None
dataset_index = 0

if file is not None:
    try:
        
        if file.name.endswith(".csv"):
             df = pd.read_csv(file)
             st.success("✅ File uploaded successfully!")
        elif file.name.endswith((".xlsx", ".xls")):
             df = pd.read_excel(file)
             st.success("✅ File uploaded successfully!")

        else:
             st.error("❌ Unsupported file type!")


        if "traffic" in file.name.lower():
            st.info("Loaded Traffic dataset.")
            dataset_index = 0
        elif "weather" in file.name.lower():
            st.info("Loaded Weather dataset.")
            dataset_index = 1
        elif "sales" in file.name.lower():
            st.info("Loaded Sales dataset.")
            dataset_index = 2
        else:
            dataset_index = 3

    except Exception as e:
        st.error(f"Error loading file: {e}")

base = st.selectbox(
    "Select a base for your analysis",
    ["Traffic", "Weather", "Sales", "others"],
    index=dataset_index
)

left, right = st.columns([1, 2])

with left:
    st.subheader("📂 Data Panel")

    if df is not None:
        st.write("**Rows:**", df.shape[0])
        st.write("**Columns:**", df.shape[1])

        selected_columns = st.multiselect(
            "Select columns",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

        st.markdown("### Preview")
        st.dataframe(df[selected_columns].head(3600) if selected_columns else df.head())
    else:
        st.info("Upload a dataset to begin.")

with right:
    st.subheader("📊 Analytics Panel")

    if df is not None:

        if base == "Traffic":
            analysis_type = st.multiselect("Select analysis type", Traffic.analysis_options)
        elif base == "Weather":
            analysis_type = st.multiselect("Select analysis type", Weather.analysis_options)
        elif base == "Sales":
            analysis_type = st.multiselect("Select analysis type", Sales.analysis_options)
        else:
            analysis_type = st.multiselect(
                "Select analysis type",
                ["Summary Statistics", "Correlation Matrix", "Distribution Plots", "Time Series Analysis", "Custom Analysis"]
            )

        st.markdown("### Charts")
        if base == "Traffic":
            Traffic(df, st, analysis_type)
        elif base == "Weather":
            Weather(df, st, analysis_type)
        elif base == "Sales":
            Sales(df, st, analysis_type)

        st.markdown("### Insights")
        if base == "Traffic":
            st.info(Traffic.insights if Traffic.insights.strip() else "Select traffic analysis options to generate insights.")
        elif base == "Weather":
            st.info(Weather.insights if hasattr(Weather, "insights") and Weather.insights.strip() else "Select weather analysis options to generate insights.")
        elif base == "Sales":
            st.info(Sales.insights if hasattr(Sales, "insights") and Sales.insights.strip() else "Select sales analysis options to generate insights.")
        else:
            st.info("Insights are not available for this dataset type yet.")

        st.markdown("### Advanced Analysis & Predictions")
        advanced_analysis_and_predictions(df, st, base, analysis_type)

        st.markdown("### Results")
        if base == "Traffic":
            st.success(Traffic.results if Traffic.results.strip() else "No traffic results yet.")
        elif base == "Weather":
            st.success(Weather.results if Weather.results.strip() else "No weather results yet.")
        elif base == "Sales":
            st.success(Sales.results if Sales.results.strip() else "No sales results yet.")
        else:
            st.success("No specific analysis results available for this dataset.")
    else:
        st.info("Upload data to view analytics.")
