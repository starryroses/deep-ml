
import math

PI = math.pi

def power_grid_forecast(consumption_data):
    # 1. Remove the daily fluctuation
    og_days = []

    for i in range(len(consumption_data)):
        fluctuation = 10 * math.sin((2 * PI * i) / 10)
        og_days.append(consumption_data[i] - fluctuation)

    # 2. Perform linear regression
    days = list(range(len(consumption_data)))
    n = len(days)

    sum_x = sum(days)
    sum_y = sum(og_days)
    sum_xy = sum(x * y for x, y in zip(days, og_days))
    sum_x2 = sum(x**2 for x in days)

    denominator = n * sum_x2 - sum_x**2
    m = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - m * sum_x) / n

    # 3. Predict day 15's base consumption
    base_day15 = m * 14 + b

    # 4. Add day 15's fluctuation back
    fluctuation_day15 = 10 * math.sin((2 * PI * 15) / 10)
    predicted = base_day15 + fluctuation_day15

    # 5. Round, then add 5% safety margin rounded up
    rounded_prediction = round(predicted)
    final_value = math.ceil(rounded_prediction * 1.05)

    # 6. Return the final integer
    return final_value
