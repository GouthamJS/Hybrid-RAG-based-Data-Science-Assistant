"""
Generates a synthetic sales_data.csv dataset.
Columns: date, product, category, revenue, units_sold, region
200 rows spanning 2 years, multiple products and regions.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv")

PRODUCTS = {
    "Laptop X1": "Electronics",
    "SmartPhone Z": "Electronics",
    "Wireless Earbuds": "Accessories",
    "Office Chair": "Furniture",
    "Standing Desk": "Furniture",
    "Coffee Maker": "Appliances",
    "Blender": "Appliances"
}

REGIONS = ["North America", "Europe", "Asia", "South America"]

def random_date(start_date: datetime, end_date: datetime) -> datetime:
    """Gets a random date between two dates."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

def generate_csv():
    os.makedirs(CSV_DIR, exist_ok=True)
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data = []
    
    for _ in range(200):
        date = random_date(start_date, end_date).strftime("%Y-%m-%d")
        product = random.choice(list(PRODUCTS.keys()))
        category = PRODUCTS[product]
        region = random.choice(REGIONS)
        units_sold = random.randint(1, 100)
        
        # Base price variation by product
        if "Laptop" in product:
            base_price = 1000
        elif "Phone" in product:
            base_price = 600
        elif "Desk" in product:
            base_price = 300
        else:
            base_price = 50
            
        # Add some noise to the price
        price = base_price * random.uniform(0.9, 1.1)
        revenue = round(price * units_sold, 2)
        
        data.append({
            "date": date,
            "product": product,
            "category": category,
            "revenue": revenue,
            "units_sold": units_sold,
            "region": region
        })
        
    df = pd.DataFrame(data)
    # Sort by date for chronological order
    df = df.sort_values(by="date").reset_index(drop=True)
    
    csv_path = os.path.join(CSV_DIR, "sales_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated realistic sales data: {csv_path}")

if __name__ == "__main__":
    generate_csv()
