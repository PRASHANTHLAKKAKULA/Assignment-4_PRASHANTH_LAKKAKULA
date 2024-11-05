from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 1
    # Hint: Calculate scheduled vs actual travel time, then find the largest discrepancies using window functions.

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 2
    # Hint: Calculate standard deviation of departure delays, filter airlines with more than 100 flights.

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # consistent_airlines.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # TODO: Implement the SQL query for Task 3
    # Hint: Calculate cancellation rates for each route, then join with airports to get airport names.

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # canceled_routes.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 4
    # Hint: Create time of day groups and calculate average delay for each carrier within each group.

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # carrier_performance_time_of_day.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
