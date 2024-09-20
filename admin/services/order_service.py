from models import db, Order
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import func

def generate_unique_key(buyer, seller):
    import random
    import string
    random_digits = ''.join(random.choices(string.digits, k=8))
    return buyer[:1] + seller[:1] + random_digits

def scheduled_matching(app):
    with app.app_context():
        today = datetime.now()
        future_date = today + timedelta(days=2)

        orders_query = Order.query.filter(func.date(Order.value_date) > future_date, Order.status == 'Pending').all()
        Order.query.filter(func.date(Order.value_date) > future_date, Order.status == 'Pending').delete()
        db.session.commit()
        
        orders_dicts = [
            {
                'ID': order.id,
                'Type': order.transaction_type,
                'Transaction Amount': order.amount,
                'Currency': order.currency,
                'Value Date': order.value_date,
                'Order Dates': order.order_date,
                'Bank Account': order.bank_account,
                'reference': order.reference,
                'Client': order.user,
                'Status': order.status,
                'Rating': order.rating
            } for order in orders_query
        ]

        orders_df = pd.DataFrame(orders_dicts)
        if len(orders_df) > 0:
            orders_df['Value Date'] = pd.to_datetime(orders_df['Value Date'])
            orders_df['Order Dates'] = pd.to_datetime(orders_df['Order Dates'])
            matches = []
            remaining = []
            update_match = []
            for (value_date, currency), group in orders_df.groupby(['Value Date', 'Currency']):
                buy_orders = group[group['Type'] == 'buy'].sort_values(
                    by=['Rating', 'Order Dates', 'Transaction Amount'], ascending=[False, True, False])
                sell_orders = group[group['Type'] == 'sell'].sort_values(
                    by=['Rating', 'Order Dates', 'Transaction Amount'], ascending=[False, True, False])

                buy_index, sell_index = 0, 0
                while buy_index < len(buy_orders) and sell_index < len(sell_orders):
                    buy_order = buy_orders.iloc[buy_index]
                    sell_order = sell_orders.iloc[sell_index]

                    match_amount = min(buy_order['Transaction Amount'], sell_order['Transaction Amount'])
                    match = {
                        'Buyer': buy_order['Client'],
                        'Seller': sell_order['Client'],
                        'Value Date': value_date,
                        'Transaction Amount': match_amount,
                        'Currency': currency
                    }
                    matches.append(match)
                    buy_order['Transaction Amount'] -= match_amount
                    sell_order['Transaction Amount'] -= match_amount

                    if buy_order['Transaction Amount'] == 0:
                        buy_index += 1
                    if sell_order['Transaction Amount'] == 0:
                        sell_index += 1

            return matches
