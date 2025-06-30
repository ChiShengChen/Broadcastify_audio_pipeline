# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta

def check_missing_audio_files(folder_path):
    # Set the date range
    start_date = datetime(2024, 1, 1)  # Start date: January 1, 2024
    end_date = datetime(2024, 12, 10)  # End date: December 10, 2024

    # Generate a list of all dates in the specified range
    all_dates = [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range((end_date - start_date).days + 1)]

    # Retrieve all filenames from the specified folder
    existing_files = os.listdir(folder_path)

    # Extract dates from filenames (assuming the first 8 characters represent the date)
    existing_dates = set()
    for file in existing_files:
        if file[:8].isdigit():  # Check if the first 8 characters are numeric (YYYYMMDD format)
            existing_dates.add(file[:8])

    # Compare the full date list with the existing dates to find missing ones
    missing_dates = [date for date in all_dates if date not in existing_dates]

    # Output the missing dates
    if missing_dates:
        print("The following dates are missing:")
        for date in missing_dates:
            print(date)
    else:
        print("All dates have corresponding files!")

if __name__ == "__main__":
    # Prompt the user to input the folder path
    # folder_path = input("Please enter the folder path: ").strip()
    folder_path = "/media/meow/Elements/ems_call/data/data_2024all_n3"
    
    # Check if the folder exists and is valid
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        check_missing_audio_files(folder_path)
    else:
        print("The specified folder does not exist. Please check the path!")

