import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Traffic:
    results = ""
    insights = ""

    analysis_options = [
        "Traffic Trend Analysis",
        "Peak Hour Detection",
        "Junction Comparison",
        "Day-wise / Weekly Analysis",
        "Heatmap Analysis",
        "Anomaly Detection"
    ]






    def __init__(self, df, st, analysis_type):
        Traffic.results = ""
        Traffic.insights = ""

        self.df = df.copy()
        self.st = st

        self.df["DateTime"] = pd.to_datetime(self.df["DateTime"])
        self.df["Date"] = self.df["DateTime"].dt.date
        self.df["Hour"] = self.df["DateTime"].dt.hour
        self.df["Day"] = self.df["DateTime"].dt.day_name()

        left, right = st.columns([1, 1])
        self.analysis_type = np.array(analysis_type)

        for x in self.analysis_type:
            if x == "Traffic Trend Analysis":
                with left:
                    result, insight = self.TrafficTrendAnalyze()
                    Traffic.results += result
                    Traffic.insights += insight

            elif x == "Peak Hour Detection":
                with right:
                    result, insight = self.Peak_Hour_Detection()
                    Traffic.results += result
                    Traffic.insights += insight

            elif x == "Junction Comparison":
                with left:
                    result, insight = self.Junction_Comparison()
                    Traffic.results += result
                    Traffic.insights += insight

            elif x == "Day-wise / Weekly Analysis":
                with right:
                    result, insight = self.Daywise_Weekly_Analysis()
                    Traffic.results += result
                    Traffic.insights += insight

            elif x == "Heatmap Analysis":
                with left:
                    result, insight = self.Heatmap_Analysis()
                    Traffic.results += result
                    Traffic.insights += insight

            elif x == "Anomaly Detection":
                with right:
                    result, insight = self.Anomaly_Detection()
                    Traffic.results += result
                    Traffic.insights += insight







    def TrafficTrendAnalyze(self):
        self.st.subheader("Traffic Trend Analysis")
        daily_trend = self.df.groupby("Date")["Vehicles"].sum()

        plt.figure(figsize=(8, 4))
        plt.plot(daily_trend.index, daily_trend.values)
        plt.title("Daily Traffic Trend")
        plt.xlabel("Date")
        plt.ylabel("Total Vehicles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        trend = "increasing" if daily_trend.values[-1] > daily_trend.values[0] else "decreasing"
        fluctuation = "high" if np.std(daily_trend.values) > 100 else "low"

        result = f"Traffic trend analysis shows a {trend} traffic pattern.\n"
        insight = f"Traffic volume is {trend} over time, and day-to-day fluctuation is {fluctuation}."

        return result, insight
    










    def Peak_Hour_Detection(self):
        hourly_traffic = self.df.groupby("Hour")["Vehicles"].sum()
        peak_hour = hourly_traffic.idxmax()
        peak_value = hourly_traffic.max()

        self.st.subheader("Peak Hour Detection")
        self.st.info(f"The peak hour is {peak_hour}:00 with {peak_value} vehicles.")

        plt.figure(figsize=(8, 4))
        plt.bar(hourly_traffic.index, hourly_traffic.values)
        plt.title("Peak Hour Detection")
        plt.xlabel("Hour")
        plt.ylabel("Total Vehicles")
        self.st.pyplot(plt)
        plt.close()

        result = f"The peak hour is {peak_hour}:00 with {peak_value} vehicles.\n"
        insight = f"Traffic is highest around {peak_hour}:00, so that period may need better signal control or road management."

        return result, insight

    def Junction_Comparison(self):
        junction_traffic = self.df.groupby("Junction")["Vehicles"].sum()

        self.st.subheader("Junction Comparison")

        plt.figure(figsize=(8, 4))
        plt.bar(junction_traffic.index.astype(str), junction_traffic.values)
        plt.title("Junction Comparison")
        plt.xlabel("Junction")
        plt.ylabel("Total Vehicles")
        self.st.pyplot(plt)
        plt.close()

        top_junction = junction_traffic.idxmax()
        top_value = junction_traffic.max()

        result = f"Junction {top_junction} has the highest traffic with {top_value} vehicles.\n"
        insight = f"Junction {top_junction} is the busiest junction and may be the main congestion point."

        return result, insight
    














    def Daywise_Weekly_Analysis(self):
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daywise = self.df.groupby("Day")["Vehicles"].sum().reindex(day_order)

        self.st.subheader("Day-wise / Weekly Analysis")

        plt.figure(figsize=(9, 4))
        plt.bar(daywise.index, daywise.values)
        plt.title("Day-wise Traffic Analysis")
        plt.xlabel("Day")
        plt.ylabel("Total Vehicles")
        plt.xticks(rotation=45)
        self.st.pyplot(plt)
        plt.close()

        busiest_day = daywise.idxmax()
        busiest_value = daywise.max()

        result = f"{busiest_day} has the highest traffic with {busiest_value} vehicles.\n"
        insight = f"Weekly traffic is highest on {busiest_day}, so that day may experience the most road pressure."

        return result, insight

    def Heatmap_Analysis(self):
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = self.df.pivot_table(values="Vehicles", index="Day", columns="Hour", aggfunc="sum").reindex(day_order)

        self.st.subheader("Heatmap Analysis")

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        plt.title("Traffic Heatmap")
        plt.xlabel("Hour")
        plt.ylabel("Day")
        plt.xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns)
        plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
        self.st.pyplot(plt)
        plt.close()

        stacked = heatmap_data.stack()
        busiest_day, busiest_hour = stacked.idxmax()
        highest_value = stacked.max()

        result = f"Highest traffic occurs on {busiest_day} at {busiest_hour}:00 with {highest_value} vehicles.\n"
        insight = f"The busiest traffic window is {busiest_day} at {busiest_hour}:00, which is likely the critical congestion slot."

        return result, insight
    














    def Anomaly_Detection(self):
        self.st.subheader("Anomaly Detection")

        vehicle_mean = np.mean(self.df["Vehicles"])
        vehicle_std = np.std(self.df["Vehicles"])
        threshold_high = vehicle_mean + 2 * vehicle_std
        threshold_low = vehicle_mean - 2 * vehicle_std

        anomalies = self.df[(self.df["Vehicles"] > threshold_high) | (self.df["Vehicles"] < threshold_low)]

        traffic_time = self.df.groupby("DateTime")["Vehicles"].sum()
        mean_val = np.mean(traffic_time.values)
        std_val = np.std(traffic_time.values)
        upper = mean_val + 2 * std_val
        anomaly_points = traffic_time[traffic_time > upper]

        plt.figure(figsize=(8, 5))
        plt.plot(traffic_time.index, traffic_time.values, label="Traffic")
        plt.scatter(anomaly_points.index, anomaly_points.values, label="Anomalies")
        plt.title("Traffic Anomaly Detection")
        plt.xlabel("Time")
        plt.ylabel("Vehicles")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        result = f"Anomaly detection identified {len(anomalies)} anomalies in the traffic data.\n"
        insight = f"There are {len(anomalies)} unusual traffic records, which may indicate special events, incidents, or data spikes."

        return result, insight