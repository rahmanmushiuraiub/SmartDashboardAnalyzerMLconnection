import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





class Sales:
    results = ""
    insights = ""

    analysis_options = [
        "Sales Revenue Analysis",
        "Product Performance",
        "Region Analysis",
        "Time-Based Trends",
        "Correlation Analysis",
        "Top & Bottom Analysis",
        "Distribution Analysis"
    ]






    def __init__(self, df, st, analysis_type):
        Sales.results = ""
        Sales.insights = ""

        self.df = df.copy()
        self.st = st

        self.df.columns = self.df.columns.str.strip()

        if "Date" not in self.df.columns:
            self.st.error(f"'Date' column not found. Available columns: {list(self.df.columns)}")
            return

        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")

        if self.df["Date"].isna().all():
            self.st.error("Could not convert 'Date' column into datetime format.")
            return

        self.df["Hour"] = self.df["Date"].dt.hour
        self.df["Day"] = self.df["Date"].dt.day_name()
        self.df["Month"] = self.df["Date"].dt.month_name()

        left, right = st.columns([1, 1])
        self.analysis_type = np.array(analysis_type)

        for x in self.analysis_type:
            if x == "Sales Revenue Analysis":
                with left:
                    result, insight = self.Sales_Revenue_Analysis()
                    Sales.results += result
                    Sales.insights += insight

            elif x == "Product Performance":
                with right:
                    result, insight = self.Product_Performance()
                    Sales.results += result
                    Sales.insights += insight

            elif x == "Region Analysis":
                with left:
                    result, insight = self.Region_Analysis()
                    Sales.results += result
                    Sales.insights += insight

            elif x == "Time-Based Trends":
                with right:
                    result, insight = self.Time_Based_Trends()
                    Sales.results += result
                    Sales.insights += insight

            elif x == "Correlation Analysis":
                with left:
                    result, insight = self.Correlation_Analysis()
                    Sales.results += result
                    Sales.insights += insight

            elif x == "Top & Bottom Analysis":
                with right:
                    result, insight = self.Top_Bottom_Analysis()
                    Sales.results += result
                    Sales.insights += insight

            elif x == "Distribution Analysis":
                with left:
                    result, insight = self.Distribution_Analysis()
                    Sales.results += result
                    Sales.insights += insight






    def Sales_Revenue_Analysis(self):
        self.st.subheader("Sales Revenue Analysis")

        daily_sales = self.df.groupby("Date")["Revenue"].sum()

        plt.figure(figsize=(8, 4))
        plt.plot(daily_sales.index, daily_sales.values, marker="o")
        plt.title("Daily Sales Revenue")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        if len(daily_sales) > 1:
            trend = "increasing" if daily_sales.values[-1] > daily_sales.values[0] else "decreasing"
        else:
            trend = "stable"

        result = f"Sales Revenue Analysis shows a {trend} daily revenue trend.\n"
        insight = f"Daily revenue is {trend}, which helps indicate whether sales performance is improving or declining.\n"
        return result, insight





    def Product_Performance(self):
        self.st.subheader("Product Performance")

        product_sales = self.df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)

        plt.figure(figsize=(8, 4))
        plt.bar(product_sales.index.astype(str), product_sales.values)
        plt.title("Product Performance")
        plt.xlabel("Product")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        top_product = product_sales.idxmax()
        top_value = product_sales.max()

        result = f"Product Performance shows that {top_product} generated the highest revenue with {top_value}.\n"
        insight = f"{top_product} is the strongest product and may deserve more stock, promotion, or business focus.\n"
        return result, insight















    def Region_Analysis(self):
        self.st.subheader("Region Analysis")

        region_sales = self.df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)

        plt.figure(figsize=(8, 4))
        plt.bar(region_sales.index.astype(str), region_sales.values)
        plt.title("Region-wise Sales")
        plt.xlabel("Region")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        best_region = region_sales.idxmax()
        best_value = region_sales.max()

        result = f"Region Analysis shows that {best_region} has the highest sales with revenue {best_value}.\n"
        insight = f"{best_region} is the best-performing region and may be your most valuable market.\n"
        return result, insight






    def Time_Based_Trends(self):
        self.st.subheader("Time-Based Trends")

        monthly_sales = self.df.groupby("Month")["Revenue"].sum()

        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_sales = monthly_sales.reindex(month_order).dropna()

        plt.figure(figsize=(9, 4))
        plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
        plt.title("Monthly Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        best_month = monthly_sales.idxmax()
        best_value = monthly_sales.max()

        result = f"Time-Based Trends indicate that {best_month} recorded the highest monthly revenue of {best_value}.\n"
        insight = f"{best_month} appears to be the strongest sales month, suggesting possible seasonality in customer demand.\n"
        return result, insight

    def Correlation_Analysis(self):
        self.st.subheader("Correlation Analysis")

        numeric_cols = ["Quantity", "Price", "Revenue"]
        corr_data = self.df[numeric_cols].corr()

        plt.figure(figsize=(6, 5))
        plt.imshow(corr_data, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.xticks(np.arange(len(corr_data.columns)), corr_data.columns, rotation=45)
        plt.yticks(np.arange(len(corr_data.index)), corr_data.index)
        plt.title("Sales Correlation Matrix")
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        result = "Correlation Analysis shows the relationship among Quantity, Price, and Revenue.\n"
        insight = "This helps identify whether revenue changes more with product quantity, pricing, or both.\n"
        return result, insight
    





    def Top_Bottom_Analysis(self):
        self.st.subheader("Top & Bottom Analysis")

        product_sales = self.df.groupby("Product")["Revenue"].sum().sort_values()

        bottom_5 = product_sales.head(5)
        top_5 = product_sales.tail(5)

        self.st.write("Top 5 Products")
        plt.figure(figsize=(8, 4))
        plt.bar(top_5.index.astype(str), top_5.values)
        plt.title("Top 5 Products by Revenue")
        plt.xlabel("Product")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        self.st.write("Bottom 5 Products")
        plt.figure(figsize=(8, 4))
        plt.bar(bottom_5.index.astype(str), bottom_5.values)
        plt.title("Bottom 5 Products by Revenue")
        plt.xlabel("Product")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        best_product = top_5.idxmax()
        lowest_product = bottom_5.idxmin()

        result = f"Top & Bottom Analysis shows best product {best_product} and lowest product {lowest_product}.\n"
        insight = f"{best_product} is your top performer, while {lowest_product} may need improvement, discounting, or review.\n"
        return result, insight

    def Distribution_Analysis(self):
        self.st.subheader("Distribution Analysis")

        plt.figure(figsize=(8, 4))
        plt.hist(self.df["Revenue"], bins=20)
        plt.title("Revenue Distribution")
        plt.xlabel("Revenue")
        plt.ylabel("Frequency")
        plt.tight_layout()
        self.st.pyplot(plt)
        plt.close()

        mean_revenue = self.df["Revenue"].mean()

        result = f"Distribution Analysis shows the revenue distribution with an average revenue of {mean_revenue:.2f}.\n"
        insight = f"The average revenue is {mean_revenue:.2f}, which gives a central view of your typical sales performance.\n"
        return result, insight