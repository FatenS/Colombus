from datetime import datetime, timedelta
from io import BytesIO
from sqlalchemy import func
from models import db, User, OpenPosition, Historical, BankAccount, Order, Meeting
from flask_login import login_user, LoginManager, logout_user
from matplotlib import pyplot as plt
import numpy as np
import requests
from flask import render_template, request, redirect, url_for, session, jsonify, Blueprint, flash, send_from_directory
import pandas as pd
import calendar
from bs4 import BeautifulSoup
from flask_socketio import SocketIO
import re

user_bp = Blueprint('user_bp', __name__, static_folder='static', static_url_path='/static/user_bp',
                    template_folder='templates')

login_manager = None


def init_login_manager(app):
    login_manager = LoginManager(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    return login_manager


@user_bp.route('/')
def home():
    return render_template('home.html')


@user_bp.route('/glossary')
def glossary():
    return render_template('glossary.html')


def convert_to_date(value):
    if pd.isnull(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    try:
        return pd.to_datetime(value).date()
    except ValueError:
        return None


@user_bp.route('/single1')
def single1():
    return render_template('single1.html')


@user_bp.route('/single2')
def single2():
    return render_template('single2.html')


@user_bp.route('/single3')
def single3():
    return render_template('single3.html')


@user_bp.route('/single4')
def single4():
    return render_template('single4.html')


@user_bp.route('/single5')
def single5():
    return render_template('single5.html')


@user_bp.route('/report')
def report():
    return render_template('report.html')

@user_bp.route('/main')
def main():
    # Default currency set to USD, can be dynamically changed as required
    calculations = calculate_spread_and_var(currency='USD')
    data = process_data(currency="USD")
    bank_data = process_bank_data(currency='USD')
    exposure_by_asset = load_and_process_data()
    return render_template('main.html',metrics=calculations,data=data.to_dict(orient='records'), bank_data=bank_data.to_dict(orient='records'),exposure_by_asset=exposure_by_asset.to_dict(orient='records'))

def calculate_spread_and_var(currency='USD', alpha=0.05):
    # Load the data
    transaction_data = pd.read_excel('template.xlsx')
    rates_data = pd.read_excel('midmarket.xlsx')

    # Convert dates to datetime format
    transaction_data['value date'] = pd.to_datetime(transaction_data['value date'])
    rates_data['Date'] = pd.to_datetime(rates_data['Date'])

    # Filter data for selected currency
    transaction_data = transaction_data[transaction_data['currency'] == currency]
    rates_data = rates_data[['Date', currency]]  # assuming the column name matches the currency code

    # Merge transaction data with the rates data
    merged_data = transaction_data.merge(rates_data, left_on='value date', right_on='Date', how='left')

    # Calculate days between Payment Date and Value Date
    merged_data['days_gap'] = (merged_data['value date'] - merged_data['Payment Date']).dt.days

    # Calculate Value Date Cost
    merged_data['value_date_cost'] = merged_data['FX Amount'] * (0.1 / 365) * merged_data['days_gap']*merged_data[currency]

    # Sum up the Value Date Cost
    total_value_date_cost = merged_data['value_date_cost'].sum()

    # Calculate the average spread
    merged_data['spread'] = (merged_data['Execution rate'] - merged_data[currency])*100
    merged_data['spread_cost'] = (merged_data['Execution rate'] - merged_data[currency])*merged_data['FX Amount']
    average_spread = merged_data['spread'].mean()

    # Calculate historical returns and VaR
    returns = rates_data[currency].pct_change().dropna()
    var = -np.percentile(returns, 100 * alpha)  # Negative because we're considering losses

    # Total FX amount for transactions
    total_traded = transaction_data['FX Amount'].sum()

    # Scale VaR by the total FX amount (to express VaR in monetary terms)
    monetary_var = var * total_traded

    # Hedge Ratio
    hedged_amount = transaction_data[transaction_data['Hedging'] == 'Yes']['FX Amount'].sum()
    hedge_ratio = hedged_amount / total_traded if total_traded > 0 else 0

    # Net VaR (VaR for unhedged transactions)
    unhedged_amount = transaction_data[transaction_data['Hedging'] == 'No']['FX Amount'].sum()
    net_var = monetary_var * (unhedged_amount / total_traded) if total_traded > 0 else 0

    fixed_cost = 100

    # Calculate total cost
    total_cost = merged_data['spread_cost'].sum() + total_value_date_cost + fixed_cost

    # Calculate width percentages for the bars
    spread_width = (merged_data['spread_cost'].sum() / total_cost) * 100
    value_date_cost_width = (total_value_date_cost/ total_cost) * 100
    fixed_costs_width = (fixed_cost / total_cost) * 100

    return {
        "total_traded": float(f"{total_traded:.2f}"),
        "average_size": float(f"{transaction_data['FX Amount'].mean():.2f}"),
        "gross_var": float(monetary_var),
        "average_spread": float(average_spread),
        "hedge_ratio": f"{hedge_ratio:.2%}",
        "net_var": float(net_var),
        "value_date_cost":float(f"{total_value_date_cost:.2f}"),
        "spread_cost":float(f"{merged_data['spread_cost'].sum():.2f}"),
        "fixed_cost":fixed_cost,
        "spread_width": spread_width,
        "value_date_cost_width": value_date_cost_width,
        "fixed_costs_width": fixed_costs_width,
        "total_cost": total_cost
    }

def process_data(currency="USD"):
    # Load the data
    transaction_data = pd.read_excel('template.xlsx')
    rates_data = pd.read_excel('midmarket.xlsx')

    # Convert dates to datetime format
    transaction_data['value date'] = pd.to_datetime(transaction_data['value date'])
    rates_data['Date'] = pd.to_datetime(rates_data['Date'])

    # Filter data for selected currency
    transaction_data = transaction_data[transaction_data['currency'] == currency]
    rates_data = rates_data[['Date', currency]]  # assuming the column name matches the currency code

     # Group by month and calculate averages
    volume_data = transaction_data.groupby(transaction_data['value date'].dt.to_period("M")).agg({
        'FX Amount': 'sum',  # Adjust column name if necessary
        'Execution rate': 'mean'  # Adjust column name if necessary
    }).rename(columns={'FX Amount': 'Volume','Execution rate':'Execution rate' })

    rates_data = rates_data.groupby(rates_data['Date'].dt.to_period("M")).agg({
        currency: 'mean'  # Adjust column name if necessary
    }).rename(columns={currency: 'MidMarket'})

    # Merge datasets based on month
    combined_data = pd.merge(volume_data, rates_data, left_index=True, right_index=True, how='left')
    combined_data['Month'] = combined_data.index.astype(str)  # Convert PeriodIndex to string for JavaScript

    return combined_data.reset_index(drop=True)

def process_bank_data(currency="USD"):
    # Load the data
    transaction_data = pd.read_excel('template.xlsx')
    rates_data = pd.read_excel('midmarket.xlsx')

    # Convert dates to datetime format
    transaction_data['Date'] = pd.to_datetime(transaction_data['value date'])
    rates_data['Date'] = pd.to_datetime(rates_data['Date'])

    # Filter data for selected currency
    transaction_data = transaction_data[transaction_data['currency'] == currency]
    rates_data = rates_data[['Date', currency]]  # assuming the column name matches the currency code

     # Calculate the average execution rate and sum of notional executed per month for template_data
    volume_data = transaction_data.groupby([transaction_data['Date'].dt.to_period("M"), 'Bank']).agg({
        'FX Amount': 'sum',
        'Execution rate': 'mean'
    }).reset_index()

    # Calculate the average midmarket rate per month for midmarket_data
    rates_data = rates_data.groupby(rates_data['Date'].dt.to_period("M")).agg({
        currency: 'mean'
    }).reset_index()

    # Merge datasets based on the month and Bank, using left join to keep all entries from volume_data
    merged_data = pd.merge(volume_data, rates_data, left_on='Date', right_on='Date', how='left')

    # Calculate spread as the absolute difference between the average execution rate and the average midmarket rate
    merged_data['Spread'] = (merged_data['Execution rate'] - merged_data[currency])

    # Group by 'Bank' again if needed to calculate the average spread per bank
    final_data = merged_data.groupby('Bank').agg({
        'FX Amount': 'sum',  # Sum up all notional executed amounts
        'Spread': 'mean'  # Calculate the average of spreads
    }).reset_index()

    # Format the notional executed values and spread as percentages
    final_data['FX Amount'] = final_data['FX Amount'].apply(lambda x: f"{x:,}")  # Comma-separated format
    final_data['Spread'] = final_data['Spread']*100
    final_data['Spread'] = final_data['Spread'].apply(lambda x: f"{x:.2f}%")  # Convert to percentage string with two decimals

    return final_data

def load_and_process_data(currency='USD'):
    # Load the data
    transaction_data = pd.read_excel('template.xlsx')

    # Convert dates to datetime format
    transaction_data['Date'] = pd.to_datetime(transaction_data['value date'])

    # Filter data for selected currency
    transaction_data = transaction_data[transaction_data['currency'] == currency]
    # Group by 'Asset Type' and sum 'FX amount'
    exposure_data = transaction_data.groupby('Asset Type')['FX Amount'].sum().reset_index()

    # Format the FX amount for better readability
    exposure_data['FX Amount'] = exposure_data['FX Amount'].apply(lambda x: f"${x:,.2f}")

    return exposure_data

# =========== Dashboard ==============================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx'}


@user_bp.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_stream = file.stream
            file_stream.seek(0)
            bytes_io = BytesIO(file_stream.read())

            # Depending on your pandas version and the excel file type, you may need to specify an engine, e.g., engine='openpyxl' for .xlsx files
            uploaded_data = pd.read_excel(bytes_io, header=None, skiprows=1)

            for index, row in uploaded_data.iterrows():
                new_position = OpenPosition(
                    value_date=convert_to_date(row[0]),
                    currency=row[1],
                    fx_amount=row[2],
                    type=row[3],
                    user=session["username"]
                )

                db.session.add(new_position)

            db.session.commit()
            return redirect(url_for('user_bp.dashboard', error='File uploaded and processed successfully'))
        else:
            flash('Please upload a file in the correct format as per our template!')
            return redirect(request.url)
    except Exception as e:
        # Log the error here if needed
        return redirect(
            url_for('user_bp.dashboard', error='Please upload a file in the correct format as per our template!'))


@user_bp.route('/download-template')
def download_template():
    return send_from_directory('user/static', 'template.xlsx', as_attachment=True)

@user_bp.route('/download-template-sec')
def download_template_sec():
    return send_from_directory('user/static', 'templatedash.xlsx', as_attachment=True)


@user_bp.route('/metrics/<currency>')
def metrics(currency):
    period = request.args.get('period', default=30, type=int)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period)

    # Retrieve historical rates for the specified period from the database
    historical_data = Historical.query.filter(Historical.date.between(start_date, end_date)).order_by(
        Historical.date.asc()).all()

    historical_rates = []
    sum_rates = 0
    cheapest = float('inf')
    most_expensive = float('-inf')

    for data in historical_data:
        # Assuming the currency rates are stored directly as attributes
        rate = getattr(data, currency.lower(), None)
        if rate:
            rate = round(rate, 5)
            historical_rates.append(rate)
            sum_rates += rate
            cheapest = min(cheapest, rate)
            most_expensive = max(most_expensive, rate)

    rate_count = len(historical_rates)
    average_rate = sum_rates / rate_count if rate_count else None
    current_rate = historical_rates[-1] if historical_rates else None
    percentage_cheaper_than_average = (sum(
        1 for rate in historical_rates if rate < average_rate) / rate_count) * 100 if rate_count else None
    percentage_cheaper_than_average = round(percentage_cheaper_than_average,
                                            0) if percentage_cheaper_than_average else None

    decision = "Cheap" if percentage_cheaper_than_average and percentage_cheaper_than_average > 50 else "Expensive"

    # Assuming 'potential to move' is based on the volatility
    volatility = most_expensive - cheapest
    potential_to_move = 'HIGH' if volatility > 0.1 else 'LOW'

    metrics = {
        'potential_to_move': potential_to_move,
        'cheapest': cheapest if cheapest != float('inf') else None,
        'most_expensive': most_expensive if most_expensive != float('-inf') else None,
        'average_rate': average_rate,
        'current_rate': current_rate,
        'percentage_cheaper_than_average': percentage_cheaper_than_average,
        'historical_rates': historical_rates,
        'decision': decision
    }

    return jsonify(metrics)


@user_bp.route('/currency-rates-historical/<currency>')
def currency_rates_historical(currency):
    # Assuming all_currency_rates is already populated with the data from your database

    # Calculate the date 30 days ago from today
    thirty_days_ago = datetime.today() - timedelta(days=30)

    # Filter the all_currency_rates to only include the last 30 days
    filtered_rates = {date: rates for date, rates in all_currency_rates.items() if
                      datetime.strptime(date, '%Y-%m-%d') >= thirty_days_ago}

    # Now, extract only the rates for the specified currency
    historical_rates = {date: rates.get(currency) for date, rates in filtered_rates.items() if currency in rates}

    # Convert rates to a JSON-compatible format and return
    rates_json = jsonify(historical_rates)
    return rates_json


@user_bp.route('/dashboard')
def dashboard():
    global all_currency_rates
    # Retrieve open positions from the database
    open_positions = OpenPosition.query.filter_by().all()
    # Convert to DataFrame for processing
    df = pd.DataFrame([{
        'value date': position.value_date,
        'currency': position.currency,
        'FX Amount': position.fx_amount,
        'Type': position.type
    } for position in open_positions])

    if len(df) > 0:
        unique_currencies = df['currency'].unique()
        df['value date'] = pd.to_datetime(df['value date'])
        today = pd.to_datetime('today').normalize()
        filtered_dates = df[df['value date'] >= today]
        print(df)
        date_list = filtered_dates['value date'].dt.strftime('%d/%b/%Y').tolist()
        unique_dates = filtered_dates['value date'].drop_duplicates()
        sub_rows = ['export', 'import']

        rows = []
        for exposure_type in unique_currencies:
            row = {'name': exposure_type, 'sub': [], 'date_sums': []}
            exports_sum = 0
            imports_sum = 0

            for sub_row in sub_rows:
                sub_all_dates_sum = \
                    filtered_dates[(filtered_dates['currency'] == exposure_type) & (filtered_dates['Type'] == sub_row)][
                        'FX Amount'].sum()
                if sub_row == 'export':
                    exports_sum = sub_all_dates_sum
                elif sub_row == 'import':
                    imports_sum = sub_all_dates_sum
                sub_date_sums = [filtered_dates[
                                     (filtered_dates['currency'] == exposure_type) & (
                                             filtered_dates['Type'] == sub_row) & (
                                             filtered_dates['value date'] == date)]['FX Amount'].sum() for date in
                                 unique_dates]
                row['sub'].append({'name': sub_row, 'all_dates_sum': sub_all_dates_sum, 'date_sums': sub_date_sums})
            row['all_dates_sum'] = exports_sum - imports_sum

            for date in unique_dates:
                export_amount = \
                    df[(df['currency'] == exposure_type) & (df['Type'] == 'export') & (df['value date'] == date)][
                        'FX Amount'].sum()
                import_amount = \
                    df[(df['currency'] == exposure_type) & (df['Type'] == 'import') & (df['value date'] == date)][
                        'FX Amount'].sum()
                row['date_sums'].append(export_amount - import_amount)

            rows.append(row)

        # Retrieve historical rates from the database
        historical_rates = Historical.query.order_by(Historical.date.desc()).all()

        all_currency_rates = {}
        for rate in historical_rates:
            rate_date = rate.date.strftime('%Y-%m-%d')
            all_currency_rates[rate_date] = {
                'USD': rate.usd,
                'EUR': rate.eur,
                'GBP': rate.gbp,
                'JPY': rate.jpy
            }

        # Adjust rates fetching logic to use the last available rate if specific date's rate is not found
        rates = {}
        for currency in unique_currencies:
            # Start with today and go back until a rate is found
            search_date = today
            while search_date.strftime(
                    '%Y-%m-%d') not in all_currency_rates and search_date > datetime.today() - timedelta(
                days=365):
                search_date -= timedelta(days=1)
            # Use the found rate, if any
            last_available_date = search_date.strftime('%Y-%m-%d')
            if last_available_date in all_currency_rates:
                rates[currency] = round(all_currency_rates[last_available_date].get(currency, 1), 6)
            else:
                rates[currency] = 'N/A'
        return render_template('dashboard.html', currency_rates=rates, ratesHis=all_currency_rates, date_list=date_list,
                               rows=rows)
    else:
        rows = []
        rates = {}
        all_currency_rates = {}
        date_list = []
        return render_template('dashboard.html', currency_rates=rates, ratesHis=all_currency_rates, date_list=date_list,
                               rows=rows)


# ==========================================================


# ================= Bank Accounts ========================

@user_bp.route('/bank')
def bank():
    session_user = session.get('username')
    if not session_user:
        return redirect(url_for('login'))  # Redirect to login if no user in session

    total_dollar_balance = 0
    total_eur_balance = 0
    total_tnd_balance = 0
    number_of_accounts = 0
    chart_for_data = []
    # Fetch bank accounts for the session user
    accounts = BankAccount.query.filter_by(owner=session_user).all()

    # Calculate the balances, filtering by the current user's username
    total_balances = db.session.query(
        BankAccount.currency,
        func.sum(BankAccount.balance).label('total')
    ).filter(BankAccount.owner == session_user).group_by(BankAccount.currency).all()

    total_dollar_balance = sum(b.total for b in total_balances if b.currency == 'USD')
    total_euro_balance = sum(b.total for b in total_balances if b.currency == 'EUR')
    total_tnd_balance = sum(b.total for b in total_balances if b.currency == 'TND')
    number_of_accounts = len(accounts)

    # Prepare data for the chart
    data_for_chart = [{'Currency': b.currency, 'Balance': b.total} for b in total_balances]

    # Convert account objects to dictionary for the template
    data = [{
        'Status': account.status,
        'Account ID': account.id,
        'Bank': account.bank_name,
        'Branch': account.branch,
        'Account Number': account.account_number,
        'Currency': account.currency,
        'Date': account.date.strftime('%Y-%m-%d'),
        'Category': account.category,
        'Balance': account.balance
    } for account in accounts]
    return render_template('Bank.html', data=data, total_dollar_balance=total_dollar_balance,
                           total_euro_balance=total_euro_balance, total_tnd_balance=total_tnd_balance,
                           number_of_accounts=number_of_accounts, data_for_chart=data_for_chart)


@user_bp.route('/add_account', methods=['POST'])
def add_account():
    # Retrieve form data
    bank = request.form.get('bank')
    currency = request.form.get('currency')
    owner = session['username']  # Using session username as owner
    balance = float(request.form.get('balance'))
    account_number = request.form.get('nbrAcc')
    branch = request.form.get('branch')
    category = request.form.get('category')
    date = datetime.strptime(request.form.get('date'), '%Y-%m-%d')
    status = "Open"

    # Generate a primary key
    pk = f"{bank[:2].upper()}{account_number}"

    # Create and add new bank account to the database
    new_account = BankAccount(
        id=pk,
        bank_name=bank,
        currency=currency,
        owner=owner,
        balance=balance,
        account_number=account_number,
        branch=branch,
        category=category,
        date=date,
        status=status
    )
    db.session.add(new_account)
    db.session.commit()
    return redirect(url_for('user_bp.bank'))


@user_bp.route('/details/<int:account_id>', methods=['GET'])
def get_account_details(account_id):
    account = BankAccount.query.filter_by(id=account_id).first()
    if account:
        account_details = {
            "id": account.id,
            "bank_name": account.bank_name,
            "currency": account.currency,
            "owner": account.owner,
            "balance": account.balance,
            "account_number": account.account_number,
            "branch": account.branch,
            "category": account.category,
            "date": account.date.strftime("%Y-%m-%d"),  # Assuming 'date' is a datetime object
            "status": account.status
        }
        return jsonify(account_details)
    else:
        return jsonify({"message": "Account not found"}), 404


@user_bp.route('/bank-accounts/<account_id>', methods=['PUT'])
def update_bank_account(account_id):
    account = BankAccount.query.get_or_404(account_id)
    data = request.get_json()
    account.bank_name = data['bank_name']
    account.currency = data['currency']
    account.balance = data['balance']
    account.category = data.get('category')
    account.status = data['status']
    db.session.commit()
    return jsonify({'message': 'Bank account updated successfully'})


@user_bp.route('/bank-accounts/<account_id>', methods=['DELETE'])
def delete_bank_account(account_id):
    try:
        account = BankAccount.query.get_or_404(account_id)
        db.session.delete(account)
        db.session.commit()
        return jsonify({'message': 'Bank account deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error deleting account: ' + str(e)}), 500


# ===============================================================

@user_bp.route('/book_demo', methods=['GET', 'POST'])
def book_demo():
    if request.method == 'POST':
        # Retrieve form data
        company_name = request.form.get('name')
        representative_name = request.form.get('rep')
        representative_position = request.form.get('position')
        email = request.form.get('email')
        date_str = request.form.get('date')  # Assuming date is in 'YYYY-MM-DD' format
        time_str = request.form.get('time')  # Assuming time is in 'HH:MM' format
        notes = request.form.get('notes', '')

        # Convert string date and time to Python date and time objects
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        time = datetime.strptime(time_str, '%H:%M').time()

        # Create a new Meeting instance
        new_meeting = Meeting(
            company_name=company_name,
            representative_name=representative_name,
            position=representative_position,
            email=email,
            date=date,
            time=time,
            notes=notes
        )

        # Add the new meeting to the session and commit it to the database
        db.session.add(new_meeting)
        db.session.commit()

        # Redirect to a new page or back to the form with a success message
        return redirect(url_for('user_bp.home', message='Meeting booked successfully'))
    else:
        # Render the booking form page if method is GET
        return render_template('book_demo.html')  # Ensure this template exists


# ============== Sign up and Login ============================
@user_bp.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    rating = request.form.get('rating')

    existing_user = User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first()
    if existing_user:
        return redirect(url_for('admin_bp.sign', error='Username already exists'))

    new_user = User(username=username, email=email, rating=rating)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('admin_bp.sign', message='Account created successfully'))


@user_bp.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session['username'] = user.username
        login_user(user)
        return redirect(url_for('user_bp.dashboard', message='Account created successfully'))
    else:
        return redirect(url_for('user_bp.home', message='wrong credentials'))


# ==========================================
@user_bp.route('/order')
def order():
    session_user = session.get('username')
    # Retrieve bank accounts from the database
    bank_accounts = BankAccount.query.filter_by(owner=session_user).all()
    data = [{
        'id': account.id,
        'bank_name': account.bank_name,
        'currency': account.currency,
        'owner': account.owner,
        'balance': account.balance,
        'account_number': account.account_number,
        'branch': account.branch,
        'category': account.category,
        'date': account.date.strftime('%Y-%m-%d') if account.date else None,
        'status': account.status,
    } for account in bank_accounts]

    # Retrieve orders from the database
    orders = Order.query.filter_by(user=session_user).order_by(Order.status.desc()).all()
    orders_list = [{
        'id': order.reference,
        'transaction_type': order.transaction_type,
        'amount': order.amount,
        'currency': order.currency,
        'value_date': order.value_date.strftime('%Y-%m-%d') if order.value_date else None,
        'order_date': order.order_date.strftime('%Y-%m-%d') if order.order_date else None,
        'bank_account': order.bank_account,
        'reference': order.reference,
        'signing_key': order.signing_key,
        'user': order.user,
        'status': order.status,
        'rating': order.rating,
    } for order in orders]

    if orders_list:
        # Aggregate monthly sums of transaction amounts
        monthly_sums = db.session.query(
            db.func.date_trunc('month', Order.value_date).label('month'),
            db.func.sum(Order.amount).label('total_amount')
        ).group_by('month').order_by('month').all()

        chart_data = [{'year_month': month.strftime("%Y-%m"), 'Transaction Amount': total_amount} for
                      month, total_amount in monthly_sums]
    else:
        chart_data = []

    return render_template('order.html', orders=orders_list, chart_data=chart_data, data=data)


@user_bp.route('/submit_order', methods=['POST'])
def submit_order():
    data = request.form.to_dict()
    session_user = session["username"]
    # Retrieve the user from the database
    current_user = User.query.filter_by(username=session_user).first()

    # Check if the user exists and has a rating
    if current_user and current_user.rating:
        user_rating = current_user.rating
    else:
        user_rating = 0

    new_order = Order(
        user=session_user,
        id_unique=data['reference'],
        status="Pending",
        rating=user_rating,
        transaction_type=data['transaction_type'],
        amount=data['amount'],
        currency=data['currency'],
        value_date=data['value_date'],
        order_date=datetime.today().strftime('%Y-%m-%d'),
        bank_account=data['bank_account'],
        reference=data['reference']
    )
    db.session.add(new_order)
    db.session.commit()

    return 'Order executed successfully'


@user_bp.route('/delete_order/<orderReference>', methods=['DELETE'])
def delete_order(orderReference):
    try:
        order = Order.query.filter_by(reference=orderReference).first()
        if order:
            db.session.delete(order)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Order deleted successfully.'})
        else:
            return jsonify({'success': False, 'message': 'Order not found.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@user_bp.route('/data_for_chart')
def data_for_chart():
    session_user = session.get('username')  # Retrieve the current logged-in user's username from the session

    # Retrieve orders for the current user, filtering by the user's username
    orders = Order.query.filter(Order.user == session_user).with_entities(Order.value_date, Order.amount).all()

    df = pd.DataFrame(orders, columns=['value_date', 'amount'])
    df['month'] = pd.to_datetime(df['value_date']).dt.month
    df['month_abbr'] = df['month'].apply(lambda x: calendar.month_abbr[x])
    monthly_totals = df.groupby('month_abbr')['amount'].sum().reset_index()

    data = {
        'months': monthly_totals['month_abbr'].tolist(),
        'amounts': monthly_totals['amount'].tolist()
    }
    return jsonify(data)


# reporting part
def compute_var(data, historical, alpha=0.01):
    # Compute static VaR for each currency
    var_results = {}
    for currency in data['Currency'].unique():
        returns = historical[currency].pct_change().dropna()
        var = np.percentile(returns, 100 * alpha)
        var_results[currency] = var
    return var_results


def compute_expected_shortfall(data, historical, alpha=0.01):
    # Compute Expected Shortfall for each currency
    es_results = {}
    for currency in data['Currency'].unique():
        returns = historical[currency].pct_change().dropna()
        var = np.percentile(returns, 100 * alpha)
        es = returns[returns <= var].mean()
        es_results[currency] = es
    return es_results


def compute_historical_losses(user_data, historical_rates):
    # Ensure 'Date' column in user_data is in datetime format
    user_data['Date'] = pd.to_datetime(user_data['Date'], errors='coerce')

    # Ensure 'Date' column in historical_rates is in datetime format
    historical_rates['Date'] = pd.to_datetime(historical_rates['Date'], errors='coerce')

    print(user_data['Date'].dtype)  # Debug: Check data type
    print(historical_rates['Date'].dtype)  # Debug: Check data type

    # Proceed with merge operation
    merged_data = pd.merge(user_data, historical_rates, on='Date', how='left')

    # Debug: Print merged data to see if 'Date' columns merged correctly
    print(merged_data.head())

    # Continue with your previous logic
    # ...
    return merged_data


def generate_exposure_graph(results, graph_path):
    if results.empty:
        print("Dataframe is empty. No graph will be generated.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Exposure (Amount FX)', color=color)
    ax1.plot(results['Date'], results['Amount FX'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Loss/Gain (TND)', color=color)  # we already handled the x-label with ax1
    ax2.plot(results['Date'], results['loss_gain_tnd'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(graph_path)
    plt.close(fig)  # Close the figure to free memory


"""""
def reporting(email):
    report_output_path = f'user/databases/reports/{email}_report.pdf'
    graph_path = 'user/databases/reports/graph.png'
    historical_path = 'user/databases/historical.xlsx'
    data_path = f'user/databases/reports/{email}.xlsx'
    # Ensure data is loaded as DataFrame
    historical = pd.read_excel(historical_path)
    data = pd.read_excel(data_path)

    report_output_path = f'user/databases/reports/{email}_report.pdf'
    document = SimpleDocTemplate(report_output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = "Analysis of Historical Performance"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 0.2 * inch))

    # Objective of the report
    objective_text = 
    Objective of the report: This report is a simple analysis of your historical 
    performance compared to the midmarket historical rates, to show the loss/gain from your 
    strategy and identify missed opportunities.
    story.append(Paragraph(objective_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    var = compute_var(data, historical)
    es = compute_expected_shortfall(data, historical)
    results = compute_historical_losses(data, historical)
    average_spread = results['spread'].mean()
    loss_gain_tnd = results['loss_gain_tnd'].sum()

    metrics_text = f
    This analysis reveals key financial metrics based on your transaction history. 
    The Value at Risk (VaR) at 1% is {var}, indicating the maximum loss expected over one year with 99% confidence. 
    The Expected Shortfall is {es}, representing the average loss in scenarios beyond the VaR threshold.
    Your average spread is {average_spread}, and the total loss in TND is {loss_gain_tnd}, 
    highlighting the overall effectiveness of your FX strategy. story.append(Paragraph(metrics_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # If you have a graph to include
    generate_exposure_graph(results, 'user/databases/reports/graph.png')
    graph_path = 'user/databases/reports/graph.png'
    story.append(Image(graph_path, 4 * inch, 3 * inch))  # Adjust size as needed

    # Finalize the PDF
    document.build(story)
    return jsonify({'message': 'Report generated successfully', 'report_path': report_output_path})
"""


@user_bp.route('/save-data', methods=['POST'])
def save_data():
    request_data = request.get_json()
    data = request_data['data']
    email = request_data['email']
    headers = request_data['headers']
    return jsonify({'message': 'We will contact you soon.'})


""""
def load_chart_data(currency):
    session_name = session["username"]
    df = pd.read_excel(f'user/databases/{session_name}.xlsx', sheet_name='Open Positions')

    # Filter the DataFrame for the specified currency
    df = df[df['currency'] == currency]

    # Convert 'value date' column to datetime
    df['value date'] = pd.to_datetime(df['value date'])

    # Group by month and sum the amounts
    df['Month'] = df['value date'].dt.strftime('%Y-%m')
    exports = df[df['Type'] == 'export'].groupby('Month')['FX Amount'].sum().reset_index()
    imports = df[df['Type'] == 'import'].groupby('Month')['FX Amount'].sum().reset_index()

    # Combine exports and imports by month
    combined = pd.merge(exports, imports, on='Month', how='outer').fillna(0)
    combined.columns = ['Month', 'Exports', 'Imports']

    # Calculate intersection (min of exports and imports)
    combined['Intersection'] = (combined['Exports'] - combined['Imports']).abs()

    # Prepare the final dataset for the chart
    chart_data = {
        'labels': combined['Month'].tolist(),  # Use months as labels
        'datasets': [
            {
                'label': 'Exports',
                'data': combined['Exports'].tolist(),
                'fill': 'start',
                'backgroundColor': 'rgba(2, 8, 56,1)',
                'borderColor': 'rgba(2, 8, 56,1)',
                'borderRadius': 3,
                'order': 2
            },
            {
                'label': 'Imports',
                'data': combined['Imports'].tolist(),
                'fill': 'start',
                'backgroundColor': 'rgba(192, 192, 192, 1)',
                'borderColor': 'rgba(192, 192, 192, 1)',
                'borderRadius': 3,
                'order': 1
            },
            {
                'label': 'Net Exposure',
                'data': combined['Intersection'].tolist(),
                'fill': '-1',
                'backgroundColor': 'rgba(0, 128, 0, 1)',
                'borderColor': 'rgba(0, 128, 0, 1)',
                'borderRadius': 3,
                'order': 0
            }
        ]
    }

    return chart_data

@user_bp.route('/chart-data/')
def chart_data():
    currency = request.args.get('currency', default='EUR')
    data = load_chart_data(currency)
    return jsonify(data)
"""


# ===============Databases Part ===========================
@user_bp.route('/logout')
def logout():
    session.clear()
    logout_user()
    return redirect(url_for('user_bp.home', message='Logged out successfully'))


# ============ historical rates ============================
@user_bp.route('/historical', methods=['GET'])
def get_historical_rates():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    currency = request.args.get('currency')

    query = Historical.query
    if start_date:
        query = query.filter(Historical.date >= start_date)
    if end_date:
        query = query.filter(Historical.date <= end_date)
    if currency:
        query = query.filter(getattr(Historical, currency) != None)

    results = query.all()

    if currency:
        data = [{
            'date': record.date.strftime('%Y-%m-%d'),
            currency: getattr(record, currency)
        } for record in results]
    else:
        data = [{
            'id': record.id,
            'date': record.date.strftime('%Y-%m-%d'),
            'usd': record.usd,
            'eur': record.eur,
            'gbp': record.gbp,
            'jpy': record.jpy
        } for record in results]

    return jsonify(data)


@user_bp.route('/historical', methods=['POST'])
def create_historical_record():
    data = request.get_json()
    new_record = Historical(
        date=data['date'],
        usd=data['usd'],
        eur=data['eur'],
        gbp=data['gbp'],
        jpy=data['jpy']
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({'message': 'Historical record created'})


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


def register_user_jobs(scheduler, app):
    scheduler.add_job(fetch_and_calculate_exchange_rates, 'cron', hour=11, minute=5, args=[app])


# ================== Live rates ==================
# Initializing rates as dictionaries to hold rates for each currency
rates = {
    'USD': {'XE': 0, 'WISE': 0, 'YahooFinance': 0},
    'EUR': {'XE': 0, 'WISE': 0, 'YahooFinance': 0},
    'JPY': {'XE': 0, 'WISE': 0, 'YahooFinance': 0}
}
metric = {
    'USD': {"High": 0, "Low": 0, "Average": 0, "Volatility": 0},
    'EUR': {"High": 0, "Low": 0, "Average": 0, "Volatility": 0},
    'JPY': {"High": 0, "Low": 0, "Average": 0, "Volatility": 0}
}
rates_all = {
    'EUR/USD': {'rate': 0, 'change': ''},
    'USD/JPY': {'rate': 0, 'change': ''},
    'GBP/USD': {'rate': 0, 'change': ''},
    'EUR/JPY': {'rate': 0, 'change': ''},
    'GBP/EUR': {'rate': 0, 'change': ''},
    'USD/CHF': {'rate': 0, 'change': ''},
}

lastUpdated = None

# Modify URLs to be formatted with currency codes dynamically
urlTemplateXE = "https://www.xe.com/currencyconverter/convert/?Amount=1&From={}&To=TND"
urlTemplateWISE = "https://wise.com/us/currency-converter/{}-to-tnd-rate?amount=1"
urlTemplateYahooFinance = "https://finance.yahoo.com/quote/TND%3DX?p=TND%3DX"  # Note: This might not change dynamically for each currency, as Yahoo Finance's URL structure may not support direct currency conversion paths for all currencies.

socketio = None


def init_socketio(app):
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*")
    return socketio


def update_currency_rates(currency):
    global rates, lastUpdated, metric, rates_all
    # Modify currency parsing to allow dynamic URL formatting
    urlXE = urlTemplateXE.format(currency)
    urlWISE = urlTemplateWISE.format(currency)
    # Assuming the fixed URL for scraping specific currency pairs
    urlCurrencyPairs = "https://www.xe.com/currencycharts/"

    # XE.com
    try:
        response = requests.get(urlXE)
        soup = BeautifulSoup(response.content, "html.parser")
        rates[currency]['XE'] = re.sub(r"[^\d\-.]", "",
                                       soup.select_one(".result__BigRate-sc-1bsijpp-1.dPdXSB").get_text())
        # Scrape the High value
        high_value = soup.select_one('th:contains("High") + td')
        if high_value:
            metric[currency]["High"] = high_value.text.strip()

        # Scrape the Low value
        low_value = soup.select_one('th:contains("Low") + td')
        if low_value:
            metric[currency]["Low"] = low_value.text.strip()

        # Scrape the Volatility
        volatility_value = soup.select_one('th:contains("Volatility") + td')
        if volatility_value:
            metric[currency]["Volatility"] = volatility_value.text.strip()
    except Exception as e:
        print(f"Error scraping XE.com for {currency}: {e}")

    # Wise.com
    try:
        response = requests.get(urlWISE)
        soup = BeautifulSoup(response.content, "html.parser")
        rates[currency]['WISE'] = re.sub(r"[^\d\-.]", "", soup.select_one(".text-success").get_text())
        rates[currency]['YahooFinance'] = re.sub(r"[^\d\-.]", "", soup.select_one(".text-success").get_text())
    except Exception as e:
        print(f"Error scraping Wise.com for {currency}: {e}")

    # Calculate the average rate (assuming rates are now correctly fetched and converted to floats)
    try:
        rate_values = [float(rates[currency]['XE']), float(rates[currency]['WISE']),
                       float(rates[currency]['YahooFinance'])]
        average_rate = sum(rate_values) / 3
        metric[currency]['Average'] = average_rate
    except Exception as e:
        print(f"Error calculating average rate for {currency}: {e}")

    try:
        response = requests.get(urlCurrencyPairs)
        soup = BeautifulSoup(response.content, "html.parser")
        # Find the table by class name
        currency_table = soup.find_all("table", class_="table__TableBase-sc-1j0jd5l-0")[0]  # Getting the first table
        rows = currency_table.find_all("tr")[1:]  # Skip header row

        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 3:  # Ensure there are enough cells for a pair, its rate, and change
                pair_link = cells[0].find('a')
                if pair_link:
                    pair_text = pair_link.get_text(strip=True)
                    rate_text = cells[1].get_text(strip=True)
                    change_symbol = cells[2].text.strip()

                    # Determine change direction
                    change_direction = 'Stable'
                    if '▲' in change_symbol:
                        change_direction = 'Up'
                    elif '▼' in change_symbol:
                        change_direction = 'Down'

                    for pair in rates_all.keys():
                        if pair.replace("/", " / ") == pair_text:  # Matching format with the HTML content
                            rates_all[pair]['rate'] = float(re.sub(r"[^\d.]", "", rate_text))  # Convert rate to float
                            rates_all[pair]['change'] = change_direction
    except Exception as e:
        print(f"Error all currency: {e}")

    print(rates_all)

    lastUpdated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    socketio.emit('rates_update', {
        'currency': currency,
        'rates': rates[currency],
        'lastUpdated': lastUpdated,
        'metrics': metric[currency],
        'rates_all': rates_all
    })


@user_bp.route('/live-rates', methods=['GET'])
def get_live_rates():
    currency = request.args.get('currency', 'USD')  # Default to USD/TND if not specified
    if currency not in rates:
        return jsonify({'error': 'Unsupported currency pair'}), 400
    return jsonify({
        'currency': currency,
        'rates': rates[currency],
        'lastUpdated': lastUpdated,
        'metrics': metric[currency],
        'rates_all': rates_all
    })


def register_live_rates(scheduler, app):
    for currency in rates.keys():
        scheduler.add_job(update_currency_rates, 'interval', minutes=1, args=[currency])
