import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Weather:
    results = ""
    insights = ""

    analysis_options = [
        "Temperature Analysis",
        "Humidity Analysis",
        "Rainfall Analysis",
        "Wind Speed Analysis",
        "Time-Based Analysis",
        "City Comparison",
        "Correlation Analysis"
    ]





    def __init__(self, df, st, analysis_type):
        Weather.results = ""
        Weather.insights = ""

        self.df = df.copy()
        self.st = st

        self.df["DateTime"] = pd.to_datetime(self.df["DateTime"], errors="coerce")

        if self.df["DateTime"].isna().all():
            self.st.error("Could not convert 'DateTime' column into datetime format.")
            return

        self.df["Date"] = self.df["DateTime"].dt.date
        self.df["Hour"] = self.df["DateTime"].dt.hour
        self.df["Day"] = self.df["DateTime"].dt.day_name()

        left, right = st.columns([1, 1])
        self.analysis_type = np.array(analysis_type)

        for x in self.analysis_type:
            if x == "Temperature Analysis":
                with left:
                    result, insight = self.Temperature_Analysis()
                    Weather.results += result
                    Weather.insights += insight

            elif x == "Humidity Analysis":
                with right:
                    result, insight = self.Humidity_Analysis()
                    Weather.results += result
                    Weather.insights += insight

            elif x == "Rainfall Analysis":
                with left:
                    result, insight = self.Rainfall_Analysis()
                    Weather.results += result
                    Weather.insights += insight

            elif x == "Wind Speed Analysis":
                with right:
                    result, insight = self.WindSpeed_Analysis()
                    Weather.results += result
                    Weather.insights += insight

            elif x == "Time-Based Analysis":
                with left:
                    result, insight = self.TimeBased_Analysis()
                    Weather.results += result
                    Weather.insights += insight

            elif x == "City Comparison":
                with right:
                    result, insight = self.City_Comparison()
                    Weather.results += result
                    Weather.insights += insight

            elif x == "Correlation Analysis":
                with left:
                    result, insight = self.Correlation_Analysis()
                    Weather.results += result
                    Weather.insights += insight







    def Temperature_Analysis(self):
        self.st.subheader("Temperature Analysis")

        temp_trend = self.df.groupby("Date")["Temperature"].mean()

        plt.figure(figsize=(8, 4))
        plt.plot(temp_trend.index, temp_trend.values)
        plt.title("Temperature Trend")
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        if len(temp_trend) > 1:
            trend = "increasing" if temp_trend.values[-1] > temp_trend.values[0] else "decreasing"
        else:
            trend = "stable"

        result = f"Temperature Analysis shows an {trend} average temperature trend over time.\n"
        insight = f"Temperature is {trend} over time, which may indicate changing weather conditions across the selected period.\n"
        return result, insight
    









    def Humidity_Analysis(self):
        self.st.subheader("Humidity Analysis")

        humidity = self.df.groupby("Date")["Humidity"].mean()

        plt.figure(figsize=(8, 4))
        plt.plot(humidity.index, humidity.values)
        plt.title("Humidity Trend")
        plt.xlabel("Date")
        plt.ylabel("Humidity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        avg_humidity = humidity.mean()

        result = f"Humidity Analysis shows average humidity of {avg_humidity:.2f} over the selected period.\n"
        insight = f"Overall humidity remains around {avg_humidity:.2f}, which helps describe how moist or dry the environment is.\n"
        return result, insight
    











    def Rainfall_Analysis(self):
        self.st.subheader("Rainfall Analysis")

        rainfall = self.df.groupby("Date")["Rainfall"].sum()

        plt.figure(figsize=(8, 4))
        plt.bar(rainfall.index, rainfall.values)
        plt.title("Rainfall Distribution")
        plt.xlabel("Date")
        plt.ylabel("Rainfall")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        max_day = rainfall.idxmax()
        max_value = rainfall.max()

        result = f"Rainfall Analysis shows the highest rainfall on {max_day} with total rainfall {max_value}.\n"
        insight = f"The wettest day was {max_day}, so that date experienced the most rainfall pressure.\n"
        return result, insight
    









    def WindSpeed_Analysis(self):
        self.st.subheader("Wind Speed Analysis")

        wind = self.df.groupby("Date")["WindSpeed"].mean()

        plt.figure(figsize=(8, 4))
        plt.plot(wind.index, wind.values)
        plt.title("Wind Speed Trend")
        plt.xlabel("Date")
        plt.ylabel("Wind Speed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        avg_wind = wind.mean()

        result = f"Wind Speed Analysis shows an average wind speed of {avg_wind:.2f}.\n"
        insight = f"Average wind speed is {avg_wind:.2f}, which gives a general idea of how windy the conditions were.\n"
        return result, insight
    













    def TimeBased_Analysis(self):
        self.st.subheader("Time-Based Analysis")

        hourly_temp = self.df.groupby("Hour")["Temperature"].mean()

        plt.figure(figsize=(8, 4))
        plt.plot(hourly_temp.index, hourly_temp.values)
        plt.title("Hourly Temperature Pattern")
        plt.xlabel("Hour")
        plt.ylabel("Temperature")
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()




        peak_hour = hourly_temp.idxmax()
        peak_temp = hourly_temp.max()

        result = f"Time-Based Analysis shows the highest average temperature at {peak_hour}:00 with value {peak_temp:.2f}.\n"
        insight = f"The hottest period is around {peak_hour}:00, showing when temperature usually peaks during the day.\n"
        return result, insight
















    def City_Comparison(self):
        self.st.subheader("City Comparison")

        city_temp = self.df.groupby("City")["Temperature"].mean()

        plt.figure(figsize=(8, 4))
        plt.bar(city_temp.index, city_temp.values)
        plt.title("City-wise Temperature")
        plt.xlabel("City")
        plt.ylabel("Temperature")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        hottest_city = city_temp.idxmax()
        hottest_temp = city_temp.max()

        result = f"City Comparison shows that {hottest_city} has the highest average temperature of {hottest_temp:.2f}.\n"
        insight = f"{hottest_city} is the hottest city in the dataset and may experience stronger heat conditions than others.\n"
        return result, insight
    











    def Correlation_Analysis(self):
        self.st.subheader("Correlation Analysis")

        corr = self.df[["Temperature", "Humidity", "Rainfall", "WindSpeed"]].corr()

        plt.figure(figsize=(6, 5))
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Weather Correlation Matrix")
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        result = "Correlation Analysis shows the relationship among Temperature, Humidity, Rainfall, and WindSpeed.\n"
        insight = "This helps explain how weather variables move together and whether one factor changes with another.\n"
        return result, insight