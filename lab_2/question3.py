import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# fetch stock data from excel
def read_excel_data(path):
    return pd.read_excel(path, sheet_name="IRCTC Stock Price")

# user-defined mean calculation
def calc_mean(arr):
    s = 0
    for item in arr:
        s += item
    return s / len(arr)

# user-defined variance calculation
def calc_variance(arr):
    m = calc_mean(arr)
    s = 0
    for item in arr:
        s += (item - m) ** 2
    return s / len(arr)

# measure average execution time
def measure_time(fn, arr):
    duration = []
    for _ in range(10):
        t1 = time.time()
        fn(arr)
        t2 = time.time()
        duration.append(t2 - t1)
    return sum(duration) / 10


def main():
    excel_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_excel_data(excel_path)

    stock_price = df.iloc[:, 3].values
    week_day = df.iloc[:, 2]
    month_name = df.iloc[:, 1]
    price_change = df.iloc[:, 8].values

    # numpy statistics
    mean_np = np.mean(stock_price)
    var_np = np.var(stock_price)

    # manual statistics
    mean_manual = calc_mean(stock_price)
    var_manual = calc_variance(stock_price)

    # timing analysis
    time_numpy = measure_time(np.mean, stock_price)
    time_manual = measure_time(calc_mean, stock_price)

    # Wednesday average
    wed_prices = stock_price[week_day == "Wed"]
    wed_avg = calc_mean(wed_prices)

    # April average
    april_prices = stock_price[month_name == "Apr"]
    april_avg = calc_mean(april_prices)

    # probability calculations
    prob_loss = len([x for x in price_change if x < 0]) / len(price_change)

    prob_profit_wed = len([i for i in range(len(price_change))
                           if week_day.iloc[i] == "Wed" and price_change[i] > 0]) / len(price_change)

    wed_changes = price_change[week_day == "Wed"]
    cond_profit_wed = len([x for x in wed_changes if x > 0]) / len(wed_changes)

    # results
    print("Mean using NumPy:", mean_np)
    print("Mean using Custom Logic:", mean_manual)

    print("Variance using NumPy:", var_np)
    print("Variance using Custom Logic:", var_manual)

    print("Avg Time (NumPy):", time_numpy)
    print("Avg Time (Custom):", time_manual)

    print("Average Price on Wednesday:", wed_avg)
    print("Average Price in April:", april_avg)

    print("Chance of Loss:", prob_loss)
    print("Chance of Profit on Wednesday:", prob_profit_wed)
    print("Conditional Profit Chance on Wednesday:", cond_profit_wed)

    # visualization
    plt.scatter(week_day, price_change)
    plt.xlabel("Week Day")
    plt.ylabel("Percentage Change")
    plt.title("Stock Change Percentage vs Week Day")
    plt.show()


if __name__ == "__main__":
    main()
