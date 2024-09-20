import requests
import datetime
from db import db
from models import Historical


def fetch_and_calculate_exchange_rates(app):
    def get_formatted_date():
        return datetime.date.today().strftime('%Y-%m-%d')

    today = get_formatted_date()
    app_id = "a363294bb0b24f7fa5e8bbd91f874c62"  # Your API key
    url = f"https://openexchangerates.org/api/historical/{today}.json?base=USD&app_id={app_id}"

    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors

    try:
        data = response.json()
        rates = data['rates']
        TND = rates['TND']
        EUR = rates['EUR']
        GBP = rates['GBP']
        JPY = rates['JPY']

        # Calculations
        GBPTND = TND / GBP
        USDTND = TND
        EURTND = TND / EUR
        JPYTND = (TND / JPY) * 1000

        new_data = {
            'date': today,
            'usd': USDTND,
            'eur': EURTND,
            'gbp': GBPTND,
            'jpy': JPYTND,
        }

        with app.app_context():
            new_record = Historical(**new_data)
            db.session.add(new_record)
            db.session.commit()

    except Exception as error:
        print("Error processing data:", error)
