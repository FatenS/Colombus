from datetime import datetime, timedelta, date
from models import db, User,  Order, AuditLog, ExchangeData, OpenPosition
from matplotlib import pyplot as plt
import numpy as np
from flask import  request, session, jsonify, Blueprint, flash, send_from_directory
import pandas as pd
import uuid
from flask_socketio import SocketIO
from .utils import convert_to_date, allowed_file
from .services.live_rates_service import update_currency_rates, rates, metric, rates_all, lastUpdated, socketio
from flask_jwt_extended import jwt_required, get_jwt_identity
from io import BytesIO
import json
import requests
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler

user_bp = Blueprint('user_bp', __name__, static_folder='static', static_url_path='/static/user_bp',
                    template_folder='templates')

# ============ Open positions Routes ==================
@user_bp.route('/upload-open-positions', methods=['POST'])
@jwt_required()
def upload_open_positions():
    """
    API for clients to upload open positions in bulk via an Excel file.
    """
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        return jsonify({'error': 'Invalid file format'}), 400

    file = request.files['file']

    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        file_stream = file.stream
        file_stream.seek(0)
        bytes_io = BytesIO(file_stream.read())
        uploaded_data = pd.read_excel(bytes_io)

        for index, row in uploaded_data.iterrows():
            # Example columns: 'Value Date', 'Currency', 'Amount', 'Transaction Type'
            value_date = convert_to_date(row['Value Date'])
            currency = row['Currency']
            amount = row['Amount']
            transaction_type = row['Transaction Type']

            new_open_pos = OpenPosition(
                value_date=value_date,
                currency=currency,
                amount=amount,
                transaction_type=transaction_type,
                user=user
            )
            db.session.add(new_open_pos)

        db.session.commit()

        # Optionally log action
        log_action(
            action_type='bulk_upload',
            table_name='open_position',
            record_id=-1,
            user_id=user_id,
            details={"uploaded_open_positions": len(uploaded_data)}
        )

        return jsonify({'message': f'{len(uploaded_data)} open positions uploaded'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@user_bp.route('/open-positions', methods=['GET'])
@jwt_required()
def get_open_positions():
    """
    Returns the list of open positions for the logged-in user.
    """
    try:
        user_id = get_jwt_identity()
        open_positions = OpenPosition.query.filter_by(user_id=user_id).all()

        # Transform each record into JSON
        data = []
        for pos in open_positions:
            data.append({
                "id": pos.id,
                "value_date": pos.value_date.strftime('%Y-%m-%d'),  # convert date to string
                "currency": pos.currency,
                "amount": pos.amount,
                "transaction_type": pos.transaction_type,
                # add any other fields you want to expose
            })

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@user_bp.route('/convert-open-position/<int:open_position_id>', methods=['POST'])
@jwt_required()
def convert_open_position(open_position_id):
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        open_pos = OpenPosition.query.get_or_404(open_position_id)

        # Create an Order with the data from the OpenPosition
        new_order = Order(
            value_date=open_pos.value_date,
            currency=open_pos.currency,
            amount=open_pos.amount,
            original_amount=open_pos.amount,
            transaction_type=open_pos.transaction_type,
            user_id=open_pos.user_id,  # or user_id if you want to reassign
            transaction_date=datetime.now(),
            order_date=datetime.now(),
            status='Market',  # or 'Pending' or whatever default
        )
        db.session.add(new_order)

        # Optionally remove the open position or set a 'converted' flag
        db.session.delete(open_pos)
        # or: open_pos.status = 'Converted'

        db.session.commit()

        # Log action
        log_action(
            action_type='convert_open_position',
            table_name='open_position',
            record_id=open_position_id,
            user_id=user_id,
            details={"message": "Converted open_position to Order"}
        )

        return jsonify({'message': 'Open position converted to Order'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def delete_expired_positions(app):
    with app.app_context():
        today = date.today()
        expired_positions = OpenPosition.query.filter(OpenPosition.value_date < today).all()
        for pos in expired_positions:
            db.session.delete(pos)
        db.session.commit()
# ============ Order Routes ==================
@user_bp.route('/orders', methods=['POST'])
@jwt_required()
def submit_order():
    """
    API for a client to submit an order.
    Logs the creation action in the AuditLog and returns debug logs in the response.
    """
    debug_logs = []  # List to store log messages

    try:
        # Fetch the user ID from the JWT token
        user_id = get_jwt_identity()
        debug_logs.append(f"Fetched user ID from JWT: {user_id}")

        user = User.query.get(user_id)
        if not user:
            debug_logs.append("User not found in the database")
            return jsonify({"message": "Invalid user", "debug": debug_logs}), 400

        debug_logs.append(f"User found: {user.email}")

        # Parse and validate incoming data
        data = request.get_json()
        if not data:
            debug_logs.append("Request payload is empty")
            return jsonify({"message": "No data provided", "debug": debug_logs}), 400

        debug_logs.append(f"Raw incoming data: {data}")

        # Validate required fields
        required_fields = ["transaction_type", "amount", "currency", "value_date", "bank_account"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            debug_logs.append(f"Missing fields: {missing_fields}")
            return jsonify({"message": "Missing required fields", "debug": debug_logs}), 400

        # Validate data types
        try:
            transaction_type = data['transaction_type']
            amount = float(data['amount'])
            currency = data['currency']
            value_date = datetime.strptime(data['value_date'], "%Y-%m-%d")
            bank_account = data['bank_account']
            debug_logs.append(f"Parsed data: transaction_type={transaction_type}, amount={amount}, "
                              f"currency={currency}, value_date={value_date}, bank_account={bank_account}")
        except Exception as e:
            debug_logs.append(f"Error parsing data: {str(e)}")
            return jsonify({"message": "Invalid data format", "debug": debug_logs}), 400

        # Generate a unique ID for the order
        unique_id = str(uuid.uuid4())
        debug_logs.append(f"Generated unique order ID: {unique_id}")

        # Create the new order
        try:
            new_order = Order(
                id_unique=unique_id,
                user=user,
                transaction_type=transaction_type,
                amount=amount,
                original_amount=amount,
                currency=currency,
                value_date=value_date,
                transaction_date=datetime.now(),
                order_date=datetime.now(),
                bank_account=bank_account,
                reference=data.get('reference', 'REF' + unique_id),
                status="Pending",
                rating=user.rating,
            )
            debug_logs.append(f"Order object created: {new_order}")
        except Exception as e:
            debug_logs.append(f"Error creating Order object: {str(e)}")
            return jsonify({"message": "Error creating Order object", "debug": debug_logs}), 500

        # Save the new order to the database
        try:
            db.session.add(new_order)
            db.session.commit()
            debug_logs.append("Order saved to the database")
        except Exception as e:
            db.session.rollback()
            debug_logs.append(f"Database error: {str(e)}")
            return jsonify({"message": "Database error", "debug": debug_logs}), 500

        # Log the action
        try:
            log = AuditLog(
                action_type='create',
                table_name='order',
                record_id=new_order.id_unique,
                user_id=user_id,
                details=json.dumps({
                    "id": new_order.id_unique,
                    "transaction_type": transaction_type,
                    "amount": amount,
                    "currency": currency,
                    "value_date": value_date.strftime("%Y-%m-%d"),
                    "bank_account": bank_account,
                })
            )
            db.session.add(log)
            db.session.commit()
            debug_logs.append("Audit log saved to the database")
        except Exception as e:
            db.session.rollback()
            debug_logs.append(f"Error saving audit log: {str(e)}")
            return jsonify({"message": "Error saving audit log", "debug": debug_logs}), 500

        return jsonify({"message": "Order executed successfully", "order_id": new_order.id_unique, "debug": debug_logs}), 201

    except Exception as e:
        # Catch any unexpected error and return its details
        debug_logs.append(f"Unexpected error: {str(e)}")
        return jsonify({"message": "An unexpected error occurred", "debug": debug_logs}), 500


@user_bp.route('/orders', methods=['GET'])
@jwt_required()  # Ensure the user is authenticated
def view_orders():
    """
    API to view all orders submitted by the logged-in client.
    """
    user_id = get_jwt_identity() 
    orders = Order.query.filter_by(user_id=user_id, deleted=False).all()
    
    if not orders:
        return jsonify([]), 200  
    
    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "transaction_type": order.transaction_type,
            "amount": order.original_amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "client_name": order.user.client_name
        })
    
    return jsonify(order_list), 200


def log_action(action_type, table_name, record_id, user_id, details):
    """
    Logs an action to the AuditLog table.
    :param action_type: The type of action (create, update, delete)
    :param table_name: The name of the table where the action was performed
    :param record_id: The ID of the record affected
    :param user_id: The ID of the user performing the action
    :param details: JSON object of the changes (old and new values)
    """
    user = User.query.get(user_id)
    details["client_name"] = user.client_name  
    log_entry = AuditLog(
        action_type=action_type,
        table_name=table_name,
        record_id=record_id,
        user_id=user_id,
        timestamp=datetime.now(),
        details=json.dumps(details)  # Store as JSON string
    )
    db.session.add(log_entry)
    db.session.commit()

@user_bp.route('/orders/<int:order_id>', methods=['PUT'])
@jwt_required()
def update_order_user(order_id):
    """
    API for users to update their own orders.
    Logs old and new values of fields that were updated.
    """
    user_id = get_jwt_identity()  # Get the user ID from the JWT token
    data = request.get_json()

    # Fetch the order by ID and check ownership
    order = Order.query.filter_by(id=order_id, user_id=user_id, deleted=False).first()
    if not order:
        return jsonify({"error": "Order not found or you don't have permission to update this order"}), 404

    # Initialize a dictionary to store changes for logging
    changes = {}

    # Update `amount` and `original_amount` while keeping a log of the original values
    if 'amount' in data and data['amount'] != order.amount:
        changes['original_amount'] = {"old": order.original_amount, "new": data['amount']}
        changes['amount'] = {"old": order.amount, "new": data['amount']}
        
        # Update both fields with the new value
        order.amount = data['amount']
        order.original_amount = data['amount']

    # Update other allowed fields
    if 'currency' in data and data['currency'] != order.currency:
        changes['currency'] = {"old": order.currency, "new": data['currency']}
        order.currency = data['currency']
        
    if 'value_date' in data:
        new_value_date = datetime.strptime(data['value_date'], "%Y-%m-%d")
        if new_value_date != order.value_date:
            changes['value_date'] = {"old": order.value_date.strftime("%Y-%m-%d"), "new": data['value_date']}
            order.value_date = new_value_date
    
    if 'bank_account' in data and data['bank_account'] != order.bank_account:
        changes['bank_account'] = {"old": order.bank_account, "new": data['bank_account']}
        order.bank_account = data['bank_account']
        
    if 'reference' in data and data['reference'] != order.reference:
        changes['reference'] = {"old": order.reference, "new": data['reference']}
        order.reference = data['reference']

    # Log the update action with old and new values
    log_action(
        action_type='update',
        table_name='order',
        record_id=order.id_unique,
        user_id=user_id,
        details=changes  # Log the changes dictionary directly
    )

    db.session.commit()

    return jsonify({"message": "Order updated successfully"}), 200


@user_bp.route('/orders/<int:order_id>', methods=['DELETE'])
@jwt_required()
def delete_order_user(order_id):
    """
    API for users to soft-delete their own orders.
    Logs the delete action in the AuditLog.
    """
    user_id = get_jwt_identity()  # Get the user ID from the JWT token

    # Fetch the order by ID and check ownership
    order = Order.query.filter_by(id=order_id, user_id=user_id, deleted=False).first()
    if not order:
        return jsonify({"error": "Order not found or you don't have permission to delete this order"}), 404

    # Perform soft delete by setting a `deleted` flag
    order.deleted = True

    # Log the delete action
    log_action(
        action_type='delete',
        table_name='order',
        record_id=order.id_unique,
        user_id=user_id,
        details={"status": "deleted"}  
    )

    db.session.commit()

    return jsonify({"message": "Order deleted successfully"}), 200


#=======batch order upload 

@user_bp.route('/upload-orders', methods=['POST'])
@jwt_required() 
def upload_orders():
    """
    API for clients to upload orders in bulk via an Excel file.
    """
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    file = request.files['file']
    
    try:
        # Process the uploaded file, using the logged-in user
        user_id = get_jwt_identity()  # Get the user ID from the JWT token
        user = User.query.get(user_id)  # Fetch the user object

        result = process_uploaded_file(file.stream, user)  # Pass the user object to the function
        return jsonify({'message': 'Orders uploaded successfully', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Template download route
@user_bp.route('/download-template')
def download_template():
    return send_from_directory('static', 'template.xlsx', as_attachment=True)


def process_uploaded_file(file_stream, user):
    file_stream.seek(0)
    bytes_io = BytesIO(file_stream.read())
    uploaded_data = pd.read_excel(bytes_io)

    try:
        for index, row in uploaded_data.iterrows():
            try:
                # Assuming the template has these columns: 'Value Date', 'Currency', 'Amount', 'Transaction Type'
                transaction_date = convert_to_date(row['Transaction Date'])
                value_date = convert_to_date(row['Value Date'])
                currency = row['Currency']
                amount = row['Amount']
                transaction_type = row['Transaction Type']
                interbank_rate = row['Interbancaire']
                # Create a new order with both amount and original_amount set to the uploaded amount
                new_order = Order(
                    value_date=value_date,
                    transaction_date=transaction_date, 
                    currency=currency,
                    amount=amount,  # Current amount, used in matching
                    original_amount=amount,  
                    transaction_type=transaction_type,
                    interbank_rate=interbank_rate,
                    status='Market',
                    user=user,
                    order_date=datetime.now()
                )
                db.session.add(new_order)
            except Exception as e:
                raise Exception(f"Error processing row {index}: {e}")

        db.session.commit()

        # Log the bulk upload action
        log_action(
            action_type='bulk_upload',
            table_name='order',
            record_id=-1,   # No single record ID for bulk actions
            user_id=user.id,
            details={"uploaded_orders": len(uploaded_data)}
        )

        return f"{len(uploaded_data)} orders successfully uploaded."

    except Exception as e:
        db.session.rollback()  # Rollback in case of an error
        raise Exception(f"Bulk upload failed: {str(e)}")
    
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


def init_socketio(app):
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*")
    return socketio




# Helper function to calculate forward rate
def calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days):
    return spot_rate * ((1 + yield_domestic  * days / 360) / (1 + yield_foreign * days / 360))


# VaR table based on currency, period, and alpha level
var_table = {
    'USD': {
        '1m': {'1%': -0.038173, '5%': -0.026578, '10%': -0.020902},
        '3m': {'1%': -0.081835, '5%': -0.062929, '10%': -0.048737},
        '6m': {'1%': -0.200238, '5%': -0.194159, '10%': -0.186580}
    },
    'EUR': {
        '1m': {'1%': -0.188726, '5%': -0.176585, '10%': -0.160856},
        '3m': {'1%': -0.187569, '5%': -0.180371, '10%': -0.174856},
        '6m': {'1%': -0.199737, '5%': -0.192892, '10%': -0.185136}
    }
}

# Helper function to determine yield period 
def get_yield_period(days):
    if days <= 60:
        return '1m'
    elif days <= 120:
        return '3m'
    else:
        return '6m'

# calculate VaR based on currency, days, and amount
def calculate_var(currency, days, amount):

    # Determine the correct time period for VaR (1m, 3m, or 6m)
    period = get_yield_period(days)
    
    # Retrieve the correct VaR rates for the currency and period
    currency_var = var_table.get(currency.upper(), {}).get(period, {})
    
    # Calculate the Value at Risk for each confidence level
    var_1 = currency_var.get('1%', 0.0) * abs(amount)
    var_5 = currency_var.get('5%', 0.0) * abs(amount)
    var_10 = currency_var.get('10%', 0.0) * abs(amount)
    
    return {'1%': var_1, '5%': var_5, '10%': var_10}


# API to calculate VaR for each order
@user_bp.route('/api/calculate-var', methods=['GET'])
def calculate_var_openpositions_api():
    try:
        # Load open positions from the DB, instead of orders
        open_positions = pd.read_sql('SELECT * FROM open_position', db.engine)

        today = datetime.today().date()
        var_calculations = []

        for _, pos in open_positions.iterrows():
            currency = pos['currency']
            amount = abs(pos['amount'])
            # Suppose we treat pos['value_date'] the same as we treat orders
            open_pos_date = pd.to_datetime(pos['value_date']).date()

            # Days difference: how you define it is up to your logic
            # e.g., (today - open_pos_date) or (open_pos_date - today)
            days_diff = (today - open_pos_date).days

            var_values = calculate_var(currency, days_diff, amount)

            var_calculations.append({
                "Value Date": open_pos_date.isoformat(),
                "Days": days_diff,
                "VaR 1%": var_values['1%'],
                "VaR 5%": var_values['5%'],
                "VaR 10%": var_values['10%']
            })

        return jsonify(var_calculations), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@user_bp.route('/api/calculate-forward-rate', methods=['GET'])
def calculate_forward_rate_api():
    try:
        # Load exchange data
        df = pd.read_sql('SELECT * FROM exchange_data', db.engine)
        open_positions = pd.read_sql('SELECT * FROM open_position', db.engine)

        today = datetime.today().date()
        today_data = df[df['Date'] == today]
        forward_rates = []

        if today_data.empty:
            return jsonify({"error": "No exchange data found for today's date"}), 404

        for _, pos in open_positions.iterrows():
            currency = pos['currency']
            open_pos_date = pd.to_datetime(pos['value_date']).date()
            days_diff = (open_pos_date - today).days  # or your chosen logic

            try:
                # Retrieve today's spot rate & yields
                spot_rate = today_data[f'Spot {currency.upper()}'].values[0]
                yield_foreign = today_data[f'{get_yield_period(days_diff).upper()} {currency.upper()}'].values[0]
                yield_domestic = today_data[f'{get_yield_period(days_diff).upper()} TND'].values[0]
            except KeyError as e:
                print(f"Missing required field in exchange data: {str(e)}")
                continue

            forward_rate_value = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)

            # Include the open_position ID so frontend can convert it
            forward_rates.append({
                "open_position_id": int(pos["id"]),  # The ID from open_position
                "Value Date": open_pos_date.isoformat(),
                "Days": days_diff,
                "Forward Rate": forward_rate_value
            })

        return jsonify(forward_rates), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
   
@user_bp.route('/api/dashboard/summary', methods=['GET'])
@jwt_required()
def dashboard_summary():
    user_id = get_jwt_identity()
 
    # Get the currency from the request arguments, default to 'USD'
    currency = request.args.get('currency', 'USD').upper()

    # Fetch orders filtered by status and the selected currency
    orders = Order.query.filter(
        Order.user_id == user_id,
        Order.currency == currency,
        Order.status.in_(['Executed', 'Matched'])
    ).all()

    # Calculate total traded and total covered based on the selected currency
    total_traded = sum(order.original_amount for order in orders)
    total_covered = sum(
        order.original_amount for order in orders
        if calculate_hedge_status(order.transaction_date, order.value_date) == "Yes"
    )
    coverage_percent = (total_covered / total_traded * 100) if total_traded > 0 else 0

    # -------------------------------------------------------
    # FIX: Distinguish Import vs. Export in Gains calculations
    # -------------------------------------------------------
    # Gains in foreign currency (USD, etc.)
    economies_totales = sum(
        (
            (
                (calculate_benchmark(order) / order.execution_rate) - 1  # Import formula
                if order.transaction_type.lower() in ["import", "buy"]
                else
                (order.execution_rate / calculate_benchmark(order)) - 1  # Export formula
            )
        ) * order.original_amount
        for order in orders
        if order.execution_rate is not None
    )

    economies_totales_couverture = sum(
        (
            (
                (calculate_benchmark(order) / order.execution_rate) - 1
                if order.transaction_type.lower() in ["import", "buy"]
                else
                (order.execution_rate / calculate_benchmark(order)) - 1
            )
        ) * order.original_amount
        for order in orders
        if order.execution_rate is not None 
           and calculate_hedge_status(order.transaction_date, order.value_date) == "Yes"
    )

    # Gains in TND (multiply Gains in foreign by order.execution_rate)
    economies_totales_tnd = sum(
        (
            (
                (calculate_benchmark(order) / order.execution_rate) - 1
                if order.transaction_type.lower() in ["import", "buy"]
                else
                (order.execution_rate / calculate_benchmark(order)) - 1
            )
        ) * order.original_amount
        * order.execution_rate  # Convert Gains to TND
        for order in orders
        if order.execution_rate is not None
    )

    economies_totales_couverture_tnd = sum(
        (
            (
                (calculate_benchmark(order) / order.execution_rate) - 1
                if order.transaction_type.lower() in ["import", "buy"]
                else
                (order.execution_rate / calculate_benchmark(order)) - 1
            )
        ) * order.original_amount
        * order.execution_rate
        for order in orders
        if order.execution_rate is not None 
           and calculate_hedge_status(order.transaction_date, order.value_date) == "Yes"
    )

    # Calculate superformance rate
    superformance_rate = calculate_superformance_rate(orders)

    # Group data by value_date month for chart-specific data
    from collections import defaultdict
    import calendar

    monthly_data = defaultdict(lambda: {"monthlyTotalTransacted": 0, "monthlyTotalGain": 0})
    for order in orders:
        month_name = calendar.month_name[order.value_date.month]
        monthly_data[month_name]["monthlyTotalTransacted"] += order.original_amount * order.execution_rate  # Convert to TND

        # Here again, fix the Gains formula inline
        if order.execution_rate is not None:
            gain_percent = (
                (calculate_benchmark(order) / order.execution_rate) - 1
                if order.transaction_type.lower() in ["import", "buy"]
                else
                (order.execution_rate / calculate_benchmark(order)) - 1
            )
            monthly_data[month_name]["monthlyTotalGain"] += (gain_percent * order.original_amount * order.execution_rate)

    # Prepare chart data
    months = list(monthly_data.keys())
    monthly_total_transacted = [
        data["monthlyTotalTransacted"] for data in monthly_data.values()
    ]
    monthly_total_gain = [
        data["monthlyTotalGain"] for data in monthly_data.values()
    ]

    summary_data = {
        "currency": currency,
        "total_traded": total_traded,
        "total_covered": total_covered,
        "coverage_percent": coverage_percent,
        "economies_totales": economies_totales,
        "economies_totales_couverture": economies_totales_couverture,
        "economies_totales_tnd": economies_totales_tnd,
        "economies_totales_couverture_tnd": economies_totales_couverture_tnd,
        "superformance_rate": superformance_rate,
        "months": months,
        "monthlyTotalTransacted": monthly_total_transacted,
        "monthlyTotalGain": monthly_total_gain
    }

    return jsonify(summary_data)




@user_bp.route('/api/dashboard/secured-vs-market-forward-rate', methods=['GET'])
@jwt_required()
def forward_rate_table():
    user_id = get_jwt_identity()

    # Get the currency from query parameters (default to USD)
    currency = request.args.get("currency", "USD").upper()

    # Load orders filtered by currency
    #orders = Order.query.filter_by(user_id=user_id, currency=currency).all()
    orders = Order.query.filter(
    Order.user_id == user_id,
    Order.currency == currency,
    Order.status.in_(['Executed', 'Matched'])  # Add status filter
).all()


    # Prepare forward rate data
    forward_rate_data = []
    for order in orders:
        # Filter for only hedged transactions
        if calculate_hedge_status(order.transaction_date, order.value_date) != "Yes":
            continue  # Skip non-hedged transactions

        # Get the secured forward rate (execution rate)
        secured_forward_rate = order.execution_rate

        # Get the benchmark for the hedged transaction
        benchmark_rate = calculate_benchmark(order)

        # Append data to the result
        forward_rate_data.append({
            "transaction_date": order.transaction_date.strftime('%Y-%m-%d'),
            "value_date": order.value_date.strftime('%Y-%m-%d'),
            "secured_forward_rate_export": secured_forward_rate if order.transaction_type in ["export", "sell"] else None,
            "secured_forward_rate_import": secured_forward_rate if order.transaction_type in ["import", "buy"] else None,
            "market_forward_rate_export": benchmark_rate if order.transaction_type in ["export", "sell"] else None,
            "market_forward_rate_import": benchmark_rate if order.transaction_type in ["import", "buy"] else None,
        })

    return jsonify(forward_rate_data)


@user_bp.route('/api/dashboard/superperformance-trend', methods=['GET'])
@jwt_required()  # Ensure the user is authenticated
def superperformance_trend():
    user_id = get_jwt_identity()
    currency = request.args.get("currency", "USD").upper()

    orders = Order.query.filter(
        Order.user_id == user_id,
        Order.currency == currency,
        Order.transaction_type.in_(['Import', 'buy']),
        Order.status.in_(['Executed', 'Matched'])
    ).order_by(Order.transaction_date).all()

    if not orders:
        return jsonify({"message": "No data available for this user"}), 200

    trend_data = []
    for order in orders:
        trend_data.append({
            "date": order.transaction_date.strftime('%Y-%m-%d'),
            "execution_rate": order.execution_rate,
            "interbank_rate": order.interbank_rate  # Already updated in the database
        })

    return jsonify(trend_data), 200


def calculate_superformance_rate(orders):
    today = datetime.now().date()

    # Fetch orders where transaction_date is less than t-2
    #.filter(Order.transaction_date <= today - timedelta(days=2))
    valid_orders = Order.query.all()

    # Filter for 'Import' or 'buy' transactions
    import_orders = [order for order in valid_orders if order.transaction_type in ['Import', 'buy']]

    # Calculate the superperformance rate
      
    superformance_count = sum(
        1 for order in import_orders 
        if order.execution_rate is not None and order.interbank_rate is not None and order.execution_rate < order.interbank_rate
    )
    superformance_rate = (superformance_count / len(import_orders) * 100) if import_orders else 0

    return superformance_rate


@user_bp.route('/api/dashboard/bank-gains', methods=['GET'])
@jwt_required()
def bank_gains():
    user_id = get_jwt_identity()

    # Get the currency from query parameters (default to USD)
    currency = request.args.get("currency", "USD").upper()

    # Filter orders by currency
    orders = Order.query.filter(
        Order.user_id == user_id,
        Order.currency == currency,
        Order.status.in_(['Executed', 'Matched'])  # Add status filter
    ).all()

    # Group data by bank and month
    bank_data = {}
    for order in orders:
        # Determine month and bank
        month = order.transaction_date.strftime('%Y-%m')
        bank = order.bank_name or "Unknown"

        if bank not in bank_data:
            bank_data[bank] = {}
        if month not in bank_data[bank]:
            bank_data[bank][month] = {'traded': 0, 'gain': 0, 'coverage': 0, 'count': 0}

        # Calculate hedge_status and benchmark
        hedge_status = calculate_hedge_status(order.transaction_date, order.value_date)
        order.hedge_status = hedge_status
        benchmark = calculate_benchmark(order)

        # Update metrics
        bank_data[bank][month]['traded'] += order.original_amount
        bank_data[bank][month]['count'] += 1

        if hedge_status == "Yes":
            bank_data[bank][month]['coverage'] += order.original_amount

        # ---------------------------------------------
        # FIX: Distinguish import/buy vs. export/sell
        # ---------------------------------------------
        if benchmark and order.execution_rate:
            tx_type = order.transaction_type.lower()
            if tx_type in ["import", "buy"]:
                # Gains% = (Benchmark / ExecRate) - 1
                gain_percent = (benchmark / order.execution_rate) - 1
            else:
                # Gains% = (ExecRate / Benchmark) - 1
                gain_percent = (order.execution_rate / benchmark) - 1
            
            bank_data[bank][month]['gain'] += gain_percent * order.original_amount

    # Format data for the table
    formatted_data = []
    for bank, months in bank_data.items():
        for month, stats in months.items():
            coverage_percent = (stats['coverage'] / stats['traded'] * 100) if stats['traded'] > 0 else 0
            formatted_data.append({
                "bank": bank,
                "month": month,
                "total_traded": stats['traded'],
                "coverage_percent": coverage_percent,
                "gain": stats['gain']  # Gains in foreign currency if ExecRate is TND/FC
            })

    return jsonify(formatted_data)


def calculate_hedge_status(transaction_date, value_date):
    return "Yes" if (value_date - transaction_date).days > 2 else "No"


def calculate_benchmark(order):

    # Fetch exchange data for the order's transaction date
    try:
        exchange_data_df = pd.read_sql(
            'SELECT * FROM exchange_data WHERE "Date" = %(transaction_date)s',
            db.engine,
            params={"transaction_date": order.transaction_date.strftime("%Y-%m-%d")}
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch exchange data: {e}")

    # Ensure exchange data is available
    if exchange_data_df.empty:
        raise ValueError(f"No exchange data available for date {order.transaction_date}")

    # Extract the spot rate for the order's currency
    try:
        spot_rate = exchange_data_df[f'Spot {order.currency.upper()}'].values[0]
    except KeyError:
        raise ValueError(f"Spot rate for {order.currency} not found in exchange data for {order.transaction_date}")

    # Validate historical loss
    historical_loss = getattr(order, 'historical_loss', None)
    if historical_loss is None:
        raise ValueError("Historical loss is missing for the order, cannot calculate benchmark.")

    # Adjust spot rate based on transaction type
    if order.transaction_type.lower() in ["import", "buy"]:
        loss_factor = 1 + historical_loss
    elif order.transaction_type.lower() in ["export", "sell"]:
        loss_factor = 1 - historical_loss
    else:
        raise ValueError(f"Unsupported transaction type: {order.transaction_type}")

    base_benchmark = spot_rate * loss_factor

    # Calculate hedge status and days difference
    days_diff = (order.value_date - order.transaction_date).days
    hedge_status = "Yes" if days_diff > 2 else "No"

    # If hedged, apply forward rate adjustment
    if hedge_status == "Yes":
        yield_period = get_yield_period(days_diff)  # Determine yield period (1m, 3m, or 6m)
        try:
            yield_foreign = exchange_data_df[f'{yield_period.upper()} {order.currency.upper()}'].values[0]
            yield_domestic = exchange_data_df[f'{yield_period.upper()} TND'].values[0]
        except KeyError as e:
            raise ValueError(f"Missing yield data for {yield_period} and currency {order.currency}: {e}")

        # Calculate forward rate factor
        forward_rate_factor = calculate_forward_rate(base_benchmark, yield_foreign, yield_domestic, days_diff)
        return  forward_rate_factor

    # Return the base benchmark for non-hedged transactions
    return base_benchmark



@user_bp.route('/update-interbank-rates', methods=['POST'])
def update_interbank_rates():
    try:
        update_order_interbank_rates()  
        return jsonify({'message': 'Interbank rates updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_order_interbank_rates(app):
    with app.app_context():
        try:
            orders = Order.query.all()
            for order in orders:
                if order.interbank_rate is None:  # Only fetch rates for orders missing this value
                    rate = fetch_rate_for_date_and_currency(order.transaction_date, order.currency)
                    if rate:
                        print(f"Updating Order ID {order.id} with rate {rate}")
                        order.interbank_rate = rate
                        db.session.add(order)
            db.session.commit()
        except Exception as e:
            print(f"Error updating interbank rates: {e}")



def fetch_rate_for_date_and_currency(date, currency):
    formatted_date = date.strftime('%Y-%m-%d')
    response = requests.post(f"https://www.bct.gov.tn/bct/siteprod/cours_archiv.jsp?input={formatted_date}&langue=en")
    soup = BeautifulSoup(response.content, 'html.parser')
    rate = None
    # Parsing logic specifically tailored for the structure of the BCT site
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if cells and cells[1].get_text(strip=True).lower() == currency.lower():
            rate = float(cells[3].get_text(strip=True).replace(',', '.'))
            break
    return rate



def start_scheduler(scheduler, app):
    """
    Start background jobs for general user-related functionality.
    """
    
    #job for updating interbank rates daily:
    scheduler.add_job(
        func=update_order_interbank_rates,
        trigger='interval',
        hours=24,
        kwargs={'app': app},
        id="update_order_interbank_rates_job"
    )

    #Schedule daily job to delete expired positions
    scheduler.add_job(
        func=delete_expired_positions,
        trigger='interval',
        hours=24,  # or any frequency you need
        kwargs={'app': app},  # pass the Flask app
        id="delete_expired_positions_job"
    )



