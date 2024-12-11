# create_roles.py
from app import app, db  # Import the Flask app and database instance from app.py
from models import Role , Order # Import the Role model (make sure it's defined in models.py)
from datetime import datetime
def create_roles():
    # Check if roles already exist
    if not Role.query.filter_by(name='Admin').first():
        admin = Role(id=1, name='Admin')
        db.session.add(admin)
    if not Role.query.filter_by(name='Client').first():
        client = Role(id=2, name='Client')
        db.session.add(client)

    db.session.commit()
    print("Roles created or already exist!")


# Sample interbank rates (date -> rate mapping)
interbank_rates = {
    "2024-12-01": 3.1052,
    "2024-12-02": 3.1079,
    "2024-12-03": 3.1091,
    "2024-12-04": 3.1115,
    "2024-12-05": 3.1132,
    "2024-12-06": 3.1167,
    "2024-12-07": 3.1190,
    "2024-12-08": 3.1187,
    "2024-12-09": 3.1202,
    "2024-12-10": 3.1230,
    "2024-12-11": 3.1254,
    "2024-12-12": 3.1278,
    "2024-12-13": 3.1305,
    "2024-12-14": 3.1331,
    "2024-12-15": 3.1340,
    "2024-12-16": 3.1355,
    "2024-12-17": 3.1372,
    "2024-12-18": 3.1397,
    "2024-12-19": 3.1415,
    "2024-12-20": 3.1432,
    "2024-12-21": 3.1447,
    "2024-12-22": 3.1465,
    "2024-12-23": 3.1489,
    "2024-12-24": 3.1505,
    "2024-12-25": 3.1510,
    "2024-12-26": 3.1532,
    "2024-12-27": 3.1550,
    "2024-12-28": 3.1563,
    "2024-12-29": 3.1585,
    "2024-12-30": 3.1600,
    "2024-12-31": 3.1622
}


def populate_interbank_rates():
    for date_str, rate in interbank_rates.items():
        transaction_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Fetch all orders with the matching transaction_date
        orders = Order.query.filter_by(transaction_date=transaction_date).all()

        # Update interbank_rate for these orders
        for order in orders:
            if order.interbank_rate != rate:  # Only update if the rate is different
                order.interbank_rate = rate
                print(f"Updated Order ID {order.id} with interbank_rate {rate}")

    db.session.commit()
    print("Interbank rates populated successfully.")





if __name__ == "__main__":
    # Ensure the Flask application context is pushed
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
        create_roles()  # Call the function to create roles
        populate_interbank_rates()






