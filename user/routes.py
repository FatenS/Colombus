from datetime import datetime, timedelta
from models import db, User, OpenPosition, Historical, BankAccount, Order, Meeting
from flask_login import login_user, LoginManager, logout_user
from matplotlib import pyplot as plt
import numpy as np
from flask import render_template, request, redirect, url_for, session, jsonify, Blueprint, flash, send_from_directory
import pandas as pd
import uuid
from flask_socketio import SocketIO
from .utils import convert_to_date, allowed_file
from .services.bank_service import BankService
from .services.user_service import UserService
from .services.order_service import OrderService
from .services.historical_service import HistoricalService
from .services.exchange_rate_service import fetch_and_calculate_exchange_rates
from .services.live_rates_service import update_currency_rates, rates, metric, rates_all, lastUpdated, socketio
from .services.meeting_service import MeetingService
from flask_jwt_extended import jwt_required, get_jwt_identity


user_bp = Blueprint('user_bp', __name__, static_folder='static', static_url_path='/static/user_bp',
                    template_folder='templates')

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

@user_bp.route('/upload', methods=['POST'])
def upload():
    from services.file_handler import process_uploaded_file
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        flash('Please upload a valid file!')
        return redirect(request.url)
    
    file = request.files['file']
    try:
        process_uploaded_file(file.stream)
        return redirect(url_for('user_bp.dashboard', error='File uploaded and processed successfully'))
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('user_bp.dashboard'))


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

@user_bp.route('/save-data', methods=['POST'])
def save_data():
    request_data = request.get_json()
    data = request_data['data']
    email = request_data['email']
    headers = request_data['headers']
    return jsonify({'message': 'We will contact you soon.'})



# ================= Bank Accounts ========================

@user_bp.route('/bank')
def bank():
    session_user = session.get('username')
    if not session_user:
        return redirect(url_for('login'))

    bank_data = BankService.get_bank_data(session_user)
    return render_template('Bank.html', **bank_data)

@user_bp.route('/add_account', methods=['POST'])
def add_account():
    form_data = request.form
    BankService.add_new_account(form_data, session['username'])
    return redirect(url_for('user_bp.bank'))

@user_bp.route('/details/<int:account_id>', methods=['GET'])
def get_account_details(account_id):
    return BankService.get_account_details(account_id)

@user_bp.route('/bank-accounts/<account_id>', methods=['PUT'])
def update_bank_account(account_id):
    data = request.get_json()
    return BankService.update_bank_account(account_id, data)

@user_bp.route('/bank-accounts/<account_id>', methods=['DELETE'])
def delete_bank_account(account_id):
    return BankService.delete_bank_account(account_id)


# ======================================= book meeting ======================== 
@user_bp.route('/book_demo', methods=['GET', 'POST'])
def book_demo():
    if request.method == 'POST':
        form_data = {
            'name': request.form.get('name'),
            'rep': request.form.get('rep'),
            'position': request.form.get('position'),
            'email': request.form.get('email'),
            'date': request.form.get('date'),
            'time': request.form.get('time'),
            'notes': request.form.get('notes', '')
        }

        try:
            message = MeetingService.book_meeting(form_data)
            flash(message)
            return redirect(url_for('user_bp.home'))
        except Exception as e:
            flash(f'Error booking meeting: {str(e)}')
            return redirect(request.url)
    else:
        return render_template('book_demo.html')  # Ensure this template exists

# ============== Sign up and Login ============================



# ============ Order Routes ==================
@user_bp.route('/orders', methods=['POST'])
@jwt_required()  # Ensure the user is authenticated
def submit_order():
    """
    API for a client to submit an order.
    """
    user_id = get_jwt_identity()  # Get the ID of the logged-in user from the JWT token
    user = User.query.get(user_id)  # Fetch the user object
    data = request.get_json()

    # Generate a unique ID for the order
    unique_id = str(uuid.uuid4())

    # Create a new order, using the relationship directly instead of setting user_id manually
    new_order = Order(
        id_unique=unique_id,
        user=user,  # Use the user object, not just the user_id
        transaction_type=data['transaction_type'],
        amount=data['amount'],
        currency=data['currency'],
        value_date=datetime.strptime(data['value_date'], "%Y-%m-%d"),
        order_date=datetime.now(),
        bank_account=data['bank_account'],
        reference=data.get('reference', 'REF' + unique_id),  # Generate reference if not provided
        status="Pending",
        rating=user.rating  

    )

    db.session.add(new_order)
    db.session.commit()

    return jsonify({"message": "Order executed successfully"}), 201



@user_bp.route('/orders', methods=['GET'])
@jwt_required()  # Ensure the user is authenticated
def view_orders():
    """
    API to view all orders submitted by the logged-in client.
    """
    user_id = get_jwt_identity()  # Get the user ID from the JWT token
    orders = Order.query.filter_by(user_id=user_id).all()
    
    if not orders:
        return jsonify([]), 200  # Return an empty list if no orders found
    
    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "transaction_type": order.transaction_type,
            "amount": order.amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
        })
    
    return jsonify(order_list), 200


# ============ Historical Rates Routes =============
@user_bp.route('/historical', methods=['GET'])
def get_historical_rates_route():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    currency = request.args.get('currency')
    data = HistoricalService.get_historical_rates(start_date, end_date, currency)
    return jsonify(data)

@user_bp.route('/historical', methods=['POST'])
def create_historical_record_route():
    data = request.json
    message = HistoricalService.create_historical_record(data)
    return jsonify(message)
#============================live rates and job registration=========
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

def register_user_jobs(scheduler, app):
    scheduler.add_job(fetch_and_calculate_exchange_rates, 'cron', hour=11, minute=5, args=[app])

def register_live_rates(scheduler, app):
    for currency in rates.keys():
        scheduler.add_job(update_currency_rates, 'interval', minutes=1, args=[currency])

def init_socketio(app):
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*")
    return socketio
