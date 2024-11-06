from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, when, stddev, count, expr, avg, hour, lit, unix_timestamp
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("/workspaces/Assignment-4_PRASHANTH_LAKKAKULA/flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("/workspaces/Assignment-4_PRASHANTH_LAKKAKULA/airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("/workspaces/Assignment-4_PRASHANTH_LAKKAKULA/carriers.csv", header=True, inferSchema=True)

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
    flights_with_time = flights_df.withColumn(
        "ScheduledTravelTime", unix_timestamp("ScheduledArrival") - unix_timestamp("ScheduledDeparture")
    ).withColumn(
        "ActualTravelTime", unix_timestamp("ActualArrival") - unix_timestamp("ActualDeparture")
    ).withColumn(
        "Discrepancy", abs(col("ScheduledTravelTime") - col("ActualTravelTime"))
    )
    
    # Define window for ranking discrepancies by carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(col("Discrepancy").desc())
    
    largest_discrepancy = flights_with_time.withColumn(
        "Rank", expr("row_number() over (PARTITION BY CarrierCode ORDER BY Discrepancy DESC)")
    ).filter(col("Rank") == 1).join(
        carriers_df, flights_with_time["CarrierCode"] == carriers_df["CarrierCode"], "inner"
    ).select(
        flights_with_time["FlightNum"].alias("FlightNumber"),
        carriers_df["CarrierName"].alias("CarrierName"),
        flights_with_time["Origin"],
        flights_with_time["Destination"],
        "ScheduledTravelTime",
        "ActualTravelTime",
        "Discrepancy"
    )
    
    largest_discrepancy.write.csv(task1_output, header=True, mode="overwrite")
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Convert timestamps to seconds for calculation
    departure_delays = flights_df.withColumn(
        "DepartureDelay", unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture")
    )
    
    consistent_airlines = departure_delays.groupBy("CarrierCode").agg(
        count("FlightNum").alias("NumFlights"),
        stddev("DepartureDelay").alias("DelayStdDev")
    ).filter(
        col("NumFlights") > 100
    ).join(
        carriers_df, "CarrierCode", "inner"
    ).select(
        carriers_df["CarrierName"].alias("CarrierName"), "NumFlights", "DelayStdDev"
    ).orderBy("DelayStdDev")
    
    consistent_airlines.write.csv(task2_output, header=True, mode="overwrite")
    print(f"Task 2 output written to {task2_output}")

# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
def task3_canceled_routes(flights_df, airports_df):
    # Add a cancellation flag in the flights dataframe
    canceled_flights = flights_df.withColumn(
        "IsCanceled", when(col("ActualDeparture").isNull(), lit(1)).otherwise(lit(0))
    )
    
    # Calculate the cancellation rate for each origin-destination pair
    cancellation_rate = canceled_flights.groupBy("Origin", "Destination").agg(
        avg("IsCanceled").alias("CancellationRate")
    )
    
    # Rename columns in airports_df to differentiate origin and destination airports
    airports_origin_df = airports_df.withColumnRenamed("AirportCode", "Origin").withColumnRenamed("City", "OriginCity")
    airports_destination_df = airports_df.withColumnRenamed("AirportCode", "Destination").withColumnRenamed("City", "DestinationCity")
    
    # Join the cancellation_rate with the renamed airports dataframes
    cancellation_rate = cancellation_rate.join(
        airports_origin_df, cancellation_rate["Origin"] == airports_origin_df["Origin"], "inner"
    ).join(
        airports_destination_df, cancellation_rate["Destination"] == airports_destination_df["Destination"], "inner"
    ).select(
        cancellation_rate["Origin"].alias("OriginAirport"), 
        airports_origin_df["OriginCity"],
        cancellation_rate["Destination"].alias("DestinationAirport"), 
        airports_destination_df["DestinationCity"],
        "CancellationRate"
    ).orderBy(col("CancellationRate").desc())
    
    cancellation_rate.write.csv(task3_output, header=True, mode="overwrite")
    print(f"Task 3 output written to {task3_output}")


# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    flights_with_time_period = flights_df.withColumn(
        "TimeOfDay",
        when((hour("ScheduledDeparture") >= 6) & (hour("ScheduledDeparture") < 12), "Morning")
        .when((hour("ScheduledDeparture") >= 12) & (hour("ScheduledDeparture") < 18), "Afternoon")
        .when((hour("ScheduledDeparture") >= 18) & (hour("ScheduledDeparture") < 24), "Evening")
        .otherwise("Night")
    )
    
    carrier_performance = flights_with_time_period.groupBy("CarrierCode", "TimeOfDay").agg(
        avg(col("ActualDeparture") - col("ScheduledDeparture")).alias("AvgDepartureDelay")
    ).join(
        carriers_df, "CarrierCode", "inner"
    ).select(
        carriers_df["CarrierName"].alias("CarrierName"), "TimeOfDay", "AvgDepartureDelay"
    ).orderBy("TimeOfDay", "AvgDepartureDelay")
    
    carrier_performance.write.csv(task4_output, header=True, mode="overwrite")
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
