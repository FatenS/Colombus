import base64
import calendar
import io
import os
import random
import string
from collections import defaultdict
import json
import numpy as np
from fpdf import FPDF
from openpyxl.workbook import Workbook
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, send_file, make_response, jsonify
from datetime import datetime, timedelta
from sqlalchemy import func
from models import db, Order, MatchedPosition, Meeting

admin_bp = Blueprint('admin_bp', __name__, template_folder='templates', static_folder='static')


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def generate_unique_key(buyer, seller):
    # Create a unique key based on the first 2 letters of buyer and seller names and 8 random digits
    random_digits = ''.join(random.choices(string.digits, k=8))
    return buyer[:1] + seller[:1] + random_digits


@admin_bp.route('/page-signup')
def sign():
    return render_template('page-signup.html')

@admin_bp.route('/')
def main():
    return render_template('page-signin.html')

@admin_bp.route('/signin', methods=['POST'])
def signin():
    username = request.form.get('username')
    password = request.form.get('password')

    if username=="engine-takwa" and password=="engine2511@":
        return redirect(url_for('admin_bp.index', message='Welcome to the admin room'))
    else:
        return redirect(url_for('admin_bp.page-signin', message='wrong credentials'))

@admin_bp.route('/out')
def logout():
    return redirect(url_for('admin_bp.main', message='Logged out successfully'))

@admin_bp.route('/rates')
def rates():
    return render_template('rates.html')


@admin_bp.route('/index', methods=['GET'])
def index():
    # Fetch new (Pending) orders directly from the database
    new_orders = Order.query.filter_by(status='Pending').all()
    new_orders_list = [
        {
            'id': order.id,
            'transaction_type': order.transaction_type,
            'amount': order.amount,
            'currency': order.currency,
            'value_date': order.value_date.strftime('%Y-%m-%d'),
            'order_date': order.order_date.strftime('%Y-%m-%d'),
            'bank_account': order.bank_account,
            'reference': order.reference,
            'signing_key': order.signing_key,
            'Client': order.user,
            'status': order.status,
            'rating': order.rating
        } for order in new_orders
    ]

    currency = request.args.get('currency', 'EUR')
    matches, metrics, market, labels, buy_data, sell_data, transaction_sums_dict = data_page(currency=currency)
    print(transaction_sums_dict)
    Area_data = {
        'labels': labels,
        'datasets': [
            {'data': buy_data, 'label': 'Buy', 'backgroundColor': 'rgba(212, 215, 222,0.5)', 'borderColor': '#d4d7de'},
            {'data': sell_data, 'label': 'Sell', 'backgroundColor': 'rgba(7, 28, 66,0.5)', 'borderColor': '#071C42'}
        ]
    }

    metrics_json = json.dumps(metrics, cls=JSONEncoder)
    # Process the market data
    processed_data = {}
    for transaction in market:
        amount = transaction['Transaction Amount'] if transaction['Type'] == 'sell' else -transaction[
            'Transaction Amount']
        date_str = transaction['Value Date']
        if date_str in processed_data:
            processed_data[date_str] += amount
        else:
            processed_data[date_str] = amount

    # Sort data by date
    sorted_dates = sorted(processed_data.keys())
    sorted_amounts = [processed_data[date] for date in sorted_dates]
    # Convert to JSON
    chart_data = json.dumps({'labels': sorted_dates, 'data': sorted_amounts})

    # Sort the data based on 'Rating' in descending order
    sorted_market_data = sorted(market, key=lambda x: x['Rating'], reverse=True)
    top_5_clients = sorted_market_data[:5]
    formatted_data = [{
        'Name': client['Client'],
        'Rating': client['Rating'],
        'Value Date': client['Value Date'],
        'Amount': client['Transaction Amount'],
        'Type': client['Type']
    } for client in top_5_clients]
    print(transaction_sums_dict)
    return render_template('index.html', matches=matches, selected_currency=currency, metrics=metrics, market=market,
                           metrics_json=metrics_json,
                           chart_data=chart_data, top_clients=formatted_data, new_orders=new_orders_list,
                           Area_data=Area_data, transaction_sums=transaction_sums_dict)


def scheduled_matching(app):
    with app.app_context():
        today = datetime.now()
        future_date = today + timedelta(days=2)

        # Query the database for orders that meet the conditions
        orders_query = Order.query.filter(func.date(Order.value_date) > future_date, Order.status == 'Pending').all()
        # Delete pending orders that match your conditions
        Order.query.filter(func.date(Order.value_date) > future_date, Order.status == 'Pending').delete()
        db.session.commit()
        # Convert query result to a list of dictionaries
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

        # Convert the list of dictionaries to a pandas DataFrame
        orders_df = pd.DataFrame(orders_dicts)

        if len(orders_df)>0:
            # Ensure 'value_date' and 'order_date' are in datetime format
            orders_df['Value Date'] = pd.to_datetime(orders_df['Value Date'])
            orders_df['Order Dates'] = pd.to_datetime(orders_df['Order Dates'])
            matches = []
            remaining = []
            update_match = []
            # Group by 'Value Date' and 'Currency'
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
                    # Generate a unique key for each match
                    unique_key = generate_unique_key(buy_order['Client'], sell_order['Client'])

                    match = {
                        'ID': unique_key,
                        'Value Date': value_date.strftime('%Y-%m-%d'),
                        'Currency': currency,
                        'Buyer': buy_order['Client'],
                        'Buyer Rating': buy_order['Rating'],
                        'Seller': sell_order['Client'],
                        'Seller Rating': sell_order['Rating'],
                        'Matched Amount': match_amount,
                    }
                    matches.append(match)

                    # Update the transaction amount for matched orders
                    buy_orders.at[buy_orders.index[buy_index], 'Transaction Amount'] -= match_amount
                    sell_orders.at[sell_orders.index[sell_index], 'Transaction Amount'] -= match_amount

                    # Mark the orders as matched and adjust their transaction amount in a separate DataFrame
                    matched_buy = buy_order.copy()
                    matched_buy['Status'] = 'Matched'
                    matched_buy['ID'] = unique_key
                    matched_buy['Transaction Amount'] = match_amount
                    update_match.append(matched_buy)

                    matched_sell = sell_order.copy()
                    matched_sell['Status'] = 'Matched'
                    matched_sell['ID'] = unique_key
                    matched_sell['Transaction Amount'] = match_amount
                    update_match.append(matched_sell)

                    # Move to the next order if the current one is fully matched
                    if buy_orders.iloc[buy_index]['Transaction Amount'] == 0:
                        buy_index += 1
                    if sell_orders.iloc[sell_index]['Transaction Amount'] == 0:
                        sell_index += 1

                # Collect remaining orders from this grouping step
                filtered_buy_orders = buy_orders[buy_orders['Transaction Amount'] > 0]
                filtered_sell_orders = sell_orders[sell_orders['Transaction Amount'] > 0]

                # Concatenate filtered DataFrames
                remaining_orders = pd.concat([filtered_buy_orders, filtered_sell_orders])
                remaining_orders['Status'] = 'Market'
                # Append the remaining_orders DataFrame to the list
                remaining.append(remaining_orders)

            # Preparing the DataFrame of updated matches to append back to the orders sheet
            update_match_df = pd.DataFrame(update_match)
            remaining_orders_df = pd.concat(remaining, ignore_index=True)
            # Combine matched and remaining orders updates
            all_updates_df = pd.concat([update_match_df, remaining_orders_df], ignore_index=True)
            # Convert DataFrame rows to Order objects and add them to the database
            for index, row in all_updates_df.iterrows():
                new_order = Order(
                    id_unique=str(row['ID']),
                    transaction_type=row['Type'],
                    amount=row['Transaction Amount'],
                    currency=row['Currency'],
                    value_date=row['Value Date'].to_pydatetime(),  # Ensure conversion to datetime
                    order_date=row['Order Dates'].to_pydatetime(),  # Ensure conversion to datetime
                    bank_account=row['Bank Account'],
                    reference=row['reference'],
                    user=row['Client'],
                    status=row['Status'],
                    rating=row.get('Rating', 0)  # Use .get() to handle optional fields
                )
                db.session.add(new_order)
            db.session.commit()

            for match in matches:
                matched_order = MatchedPosition(
                    id=str(match['ID']),
                    value_date=datetime.strptime(match['Value Date'], '%Y-%m-%d').date(),
                    currency=match['Currency'],
                    buyer=match['Buyer'],
                    buyer_rate=int(match['Buyer Rating']),
                    seller=match['Seller'],
                    seller_rate=int(match['Seller Rating']),
                    matched_amount=float(match['Matched Amount'])
                )
                db.session.add(matched_order)
            print(matches)

            db.session.commit()
        else:
            pass


def register_admin_jobs(scheduler, app):
    scheduler.add_job(scheduled_matching, 'cron', hour=16, minute=14, args=[app])


def data_page(currency):
    today = datetime.now()
    # Filter orders by currency and status
    new_orders = Order.query.filter(Order.currency == currency, Order.status == 'Pending').all()
    market_orders = Order.query.filter(Order.currency == currency, Order.status == 'Market', Order.value_date>today).all()
    matched_orders = MatchedPosition.query.filter(MatchedPosition.currency == currency).all()
    print(matched_orders)
    # Aggregate sums of buy and sell amounts by value date
    buy_sums = defaultdict(int)
    sell_sums = defaultdict(int)

    for order in new_orders:
        value_date_str = order.value_date.strftime('%Y-%m-%d')
        if order.transaction_type.lower() == 'buy':
            buy_sums[value_date_str] += order.amount
        elif order.transaction_type.lower() == 'sell':
            sell_sums[value_date_str] += order.amount

    # Prepare data for the chart
    labels = sorted(set(buy_sums.keys()) | set(sell_sums.keys()))  # Union of all dates
    buy_data = [buy_sums[date] for date in labels]
    sell_data = [sell_sums[date] for date in labels]

    # Prepare transaction sums for pie chart
    transaction_sums = Order.query.with_entities(
        Order.transaction_type, func.sum(Order.amount).label('total')
    ).group_by(Order.transaction_type).filter(Order.currency == currency).all()
    print(transaction_sums)
    transaction_sums_dict = {result.transaction_type.lower(): result.total for result in transaction_sums}
    print(transaction_sums_dict)

    # Market orders processing
    positions_by_date = {}
    for order in market_orders:
        date_key = order.value_date.strftime('%Y-%m-%d')
        position = positions_by_date.setdefault(date_key,
                                                {'Total Unmatched': 0, 'Unmatched Buy': 0, 'Unmatched Sell': 0,
                                                 'Position': 'Balanced'})
        if order.transaction_type == 'Buy':
            position['Unmatched Buy'] += order.amount
        else:
            position['Unmatched Sell'] += order.amount
        position['Total Unmatched'] += order.amount
        position['Position'] = 'Net Buyer' if position['Unmatched Buy'] > position[
            'Unmatched Sell'] else 'Net Seller' if position['Unmatched Sell'] > position[
            'Unmatched Buy'] else 'Balanced'

    # Convert matched and market orders to dicts for JSON serialization
    data = [{
        'id': order.id,
        'Value Date': order.value_date.isoformat() if order.value_date else None,
        'Currency': order.currency,
        'Buyer': order.buyer,
        'Buyer Rating': order.buyer_rate,
        'Seller': order.seller,
        'Seller Rating': order.seller_rate,
        'Matched Amount': order.matched_amount
    } for order in matched_orders]

    market = [{
        'id': order.id,
        'id_unique': order.id_unique,
        'Type': order.transaction_type,
        'Transaction Amount': order.amount,
        'Currency': order.currency,
        'Value Date': order.value_date.isoformat() if order.value_date else None,
        'Order Dates': order.order_date.isoformat() if order.order_date else None,
        'Bank Account': order.bank_account,
        'reference': order.reference,
        'Client': order.user,
        'Status': order.status,
        'Rating': order.rating if order.rating is not None else 0
    } for order in market_orders]

    # Calculate metrics result
    metrics_result = {}
    for date_key, info in positions_by_date.items():
        matched_amount = sum(
            order.matched_amount for order in matched_orders if order.value_date.strftime('%Y-%m-%d') == date_key)
        unmatched = info['Total Unmatched']
        matching_rate = (matched_amount / (matched_amount + unmatched) if (matched_amount + unmatched) > 0 else 0) * 100
        net_balance = abs(info['Unmatched Buy'] - info['Unmatched Sell'])
        position = info['Position']
        metrics_result[date_key] = {'Matching Rate': matching_rate, 'Volume Matched': matched_amount,
                                    'Net Balance': net_balance, 'Net Position': position}

    return data, metrics_result, market, labels, buy_data, sell_data, transaction_sums_dict


@admin_bp.route('/move_order_back', methods=['POST'])
def move_order_back():
    order_id = request.form.get('reference')

    # Query the database for the order with the given reference
    order = Order.query.filter_by(reference=order_id).first()

    if order:
        # Update the status of the order to 'Pending'
        order.status = 'Pending'

        # Commit the changes to the database
        db.session.commit()

    return redirect(url_for('admin_bp.index'))


@admin_bp.route('/download_excel', methods=['POST'])
def download_excel():
    tables_data = request.get_json()
    output = io.BytesIO()

    # Create a workbook and add a worksheet for each table
    wb = Workbook()
    for sheet_name, data in tables_data.items():
        # Create a new sheet with the name of the tab
        ws = wb.create_sheet(title=sheet_name)
        for row in data:
            ws.append(row)
    wb.remove(wb.active)  # Remove the automatically created default sheet

    # Save the workbook to the BytesIO object
    wb.save(output)
    output.seek(0)

    # Prepare the response
    filename = f"analytics_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    response = make_response(output.getvalue())
    response.headers.set('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers.set('Content-Disposition', 'attachment', filename=filename)

    return response


@admin_bp.route('/export-pdf', methods=['POST'])
def export_pdf():
    images = request.json['images']
    pdf = FPDF()

    for i, img_data in enumerate(images):
        img_data = img_data.split(';base64,')[1]  # Extract base64 data
        img_bytes = base64.b64decode(img_data)
        img_filename = f'temp_image_{i}.png'
        with open(img_filename, 'wb') as img_file:
            img_file.write(img_bytes)

        pdf.add_page()
        pdf.image(img_filename, x=10, y=8, w=190)  # Adjust dimensions as needed
        os.remove(img_filename)  # Clean up after adding to PDF

    pdf_output = f"analytics_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    pdf.output(pdf_output)
    return send_file(pdf_output, as_attachment=True,
                     download_name=f"analytics_{datetime.now().strftime('%Y-%m-%d')}.pdf")


@admin_bp.route('/meetings')
@admin_bp.route('/<int:year>/<int:month>')
def meetings(year=None, month=None):
    if not year or not month:
        now = datetime.now()
        current_year = now.year
        current_month = now.month
    else:
        current_year = year
        current_month = month

    month_name = calendar.month_name[current_month]
    meetings = get_meetings_for_month(current_year, current_month)
    month_days = generate_month_days(current_year, current_month)

    for day in month_days:
        if day['day'] in meetings:
            day['meetings'] = meetings[day['day']]

    # Determine next and previous month and year
    next_month = current_month % 12 + 1
    next_year = current_year + (current_month // 12)
    prev_month = current_month - 1 if current_month > 1 else 12
    prev_year = current_year - 1 if current_month == 1 else current_year

    return render_template('meetings.html', year=current_year, month=month_name, month_num=current_month,
                           month_days=month_days, next_year=next_year, next_month=next_month,
                           prev_year=prev_year, prev_month=prev_month)


def get_meetings_for_month(year, month):
    # Query the database for meetings in the specified year and month
    meetings_query = Meeting.query.filter(
        db.extract('year', Meeting.date) == year,
        db.extract('month', Meeting.date) == month
    ).all()

    # Group meetings by day of the month
    meetings_grouped = {}
    for meeting in meetings_query:
        meeting_day = meeting.date.day
        if meeting_day not in meetings_grouped:
            meetings_grouped[meeting_day] = []

        # Convert SQLAlchemy object to dictionary
        meeting_data = {
            'id': meeting.id,
            'Company Name': meeting.company_name,
            'Representative Name': meeting.representative_name,
            'Position': meeting.position,
            'Email': meeting.email,
            'Date': meeting.date.isoformat(),  # Or format it as needed
            'Time': meeting.time.isoformat(),  # Or format it as needed
            'Notes': meeting.notes
        }
        meetings_grouped[meeting_day].append(meeting_data)

    return meetings_grouped



def generate_month_days(year, month):
    # Number of days in the month and the first weekday of the month
    num_days = calendar.monthrange(year, month)[1]
    days = [{'day': day, 'meetings': []} for day in range(1, num_days + 1)]
    return days


