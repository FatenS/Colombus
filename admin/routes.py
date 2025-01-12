import base64
from io import BytesIO
import os
import json
import random
import string
import numpy as np
import pandas as pd
from fpdf import FPDF
from sqlalchemy.orm import joinedload
from collections import defaultdict
from flask import Blueprint, render_template, request, redirect, url_for, send_file, make_response, jsonify, flash, session
from datetime import datetime, timedelta
from sqlalchemy import func
from .services.export_service import export_pdf, download_excel
from models import db, Order, ExchangeData
from flask_login import login_user, logout_user, current_user
from functools import wraps
from models import User, Role, AuditLog
from flask_security import roles_accepted
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app


admin_bp = Blueprint('admin_bp', __name__, template_folder='templates', static_folder='static')

# Custom roles_required decorator
def roles_required(required_role):
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            # Get the user identity from the JWT token
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Check if the user has the required role
            user_roles = [role.name for role in user.roles]
            if required_role not in user_roles:
                return jsonify({"error": "Access forbidden, admin role required"}), 403

            # Proceed to the endpoint
            return fn(*args, **kwargs)
        return decorator
    return wrapper

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def generate_unique_key(buyer, seller):
    # Create a unique key based on the first 2 letters of buyer and seller names and 8 random digits
    random_digits = ''.join(random.choices(string.digits, k=8))
    return buyer[:1] + seller[:1] + random_digits

@admin_bp.route('/signup', methods=['GET', 'POST'])
def sign():
    email = request.json.get('email')
    password = request.json.get('password')
    client_name = request.json.get('client_name')  
    role_id = request.json.get('options')
    rating = request.json.get('rating', 0)  

    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    user = User(email=email, active=1, password=hashed_password,client_name=client_name, rating=rating)  # Set the rating when creating the user
              
    role = Role.query.filter_by(id=int(role_id)).first()
    if not role:
        return jsonify({"msg": "Role not found"}), 400

    user.roles.append(role)
    db.session.add(user)
    db.session.commit()

    return jsonify({"msg": "User created successfully!"}), 201

@admin_bp.route('/signin', methods=['GET', 'POST'])
def signin():
    email = request.json.get('email')
    password = request.json.get('password')
    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        # Create JWT token
        access_token = create_access_token(identity=str(user.id))  # Ensure identity is a string

        # Assuming the user has a relationship with roles, fetch the user's role(s)
        user_roles = [role.name for role in user.roles]  # Get list of role names

        # Return the access token and user's roles
        return jsonify({
            "access_token": access_token,
            "roles": user_roles  # Include roles in the response
        }), 200

    return jsonify({"msg": "Invalid email or password"}), 401

@admin_bp.route('api/orders', methods=['GET'])
@jwt_required()
def view_all_orders():
    """
    API for admins to view all orders in the system, regardless of the user who created them.
    """
    orders = Order.query.options(joinedload(Order.user)).filter(Order.deleted == False).all()

    if not orders:
        return jsonify([]), 200
    
    # Prepare the list of orders to return
    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "user": order.user.email if order.user else "Unknown",  # Show user's email
            "transaction_type": order.transaction_type,
            "amount": order.amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "transaction_date": order.transaction_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "client_name": order.user.client_name if order.user else "Unknown",  # Show client's name
            "execution_rate": order.execution_rate,  # Admin can see the execution rate
            "bank_name": order.bank_name,  # Admin can see the bank name
            "interbank_rate": order.interbank_rate,  # Include the interbank rate
            "historical_loss": order.historical_loss,  # Include the historical loss
        })

    return jsonify(order_list), 200


@admin_bp.route('api/orders/<int:order_id>', methods=['PUT'])
@jwt_required()
def update_order(order_id):
    """
    API for admins to update an order's status, execution rate, and bank name.
    These fields are added by the admin and not by the client.
    """
    data = request.get_json()
    
    # Fetch the order by ID
    order = Order.query.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404

    # Admin can update these fields
    order.status = data.get("status", order.status)  # Update status
    order.execution_rate = data.get("execution_rate", order.execution_rate)  # Add/update execution rate
    order.bank_name = data.get("bank_name", order.bank_name)  # Add/update bank name
    order.historical_loss=data.get("historical_loss", order.historical_loss)
    
    db.session.commit()  # Save changes
    
    return jsonify({"message": "Order updated successfully"}), 200


@admin_bp.route('/run_matching', methods=['POST'])
@jwt_required()
@roles_required('Admin')  
def run_matching():
    """
    API to trigger the scheduled matching process manually.
    This allows Admins to run the matching process via a REST API call.
    """
    try:
        debug_messages = []  # List to capture debug messages
        # Call the matching function and pass the debug list
        process_matching_orders(current_app, debug_messages)
        return jsonify({
            'message': 'Matching process executed successfully',
            'debug_messages': debug_messages  # Include debug messages in the response
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

        
def process_matching_orders(app, debug_messages):
    """
    Processes matching of pending orders by grouping them by value_date and currency,
    and updates their statuses based on matching logic. Unmatched orders are marked as 'Market'.
    """
    try:
        # Fetch Pending Orders
        pending_orders = Order.query.filter_by(status='Pending', deleted=False).all()
        if not pending_orders:
            debug_messages.append("No pending orders found.")
            return {"debug_messages": debug_messages, "message": "No pending orders to process."}

        debug_messages.append(f"Fetched Pending Orders: {[order.id for order in pending_orders]}")

        # Convert to DataFrame for easier processing
        try:
            data = [
                {
                    'id': order.id,
                    'type': order.transaction_type,
                    'amount': order.amount,
                    'original_amount': order.original_amount,
                    'currency': order.currency,
                    'value_date': order.value_date,
                    'order_date': order.order_date,
                    'rating': order.rating,
                    'user_id': order.user_id 
                } for order in pending_orders
            ]
            df = pd.DataFrame(data)
            debug_messages.append(f"Converted {len(data)} orders to DataFrame.")
        except Exception as e:
            debug_messages.append(f"Error converting orders to DataFrame: {e}")
            return {"debug_messages": debug_messages, "message": "Failed to process orders."}

        # Group by value_date and currency
        try:
            groups = df.groupby(['value_date', 'currency'])
            debug_messages.append(f"Grouped orders into {len(groups)} groups.")
        except Exception as e:
            debug_messages.append(f"Error grouping orders: {e}")
            return {"debug_messages": debug_messages, "message": "Failed to group orders."}

        for (value_date, currency), group in groups:
            debug_messages.append(f"Processing Group: Value Date={value_date}, Currency={currency}")
            buy_orders = group[group['type'] == 'buy'].sort_values(by=['rating'], ascending=False)
            sell_orders = group[group['type'] == 'sell'].sort_values(by=['rating'], ascending=True)

            for _, buy_order in buy_orders.iterrows():
                for _, sell_order in sell_orders.iterrows():
                    if buy_order['amount'] <= 0:
                        debug_messages.append(f"Buy Order ID={buy_order['id']} has no remaining amount to match.")
                        break
                    if sell_order['amount'] <= 0:
                        debug_messages.append(f"Sell Order ID={sell_order['id']} has no remaining amount to match.")
                        continue
                    if buy_order['user_id'] == sell_order['user_id']:
                        
                        continue
                    # Match orders
                    match_amount = min(buy_order['amount'], sell_order['amount'])
                    buy_order['amount'] -= match_amount
                    sell_order['amount'] -= match_amount

                    # Update database records
                    try:
                        buy = Order.query.get(buy_order['id'])
                        sell = Order.query.get(sell_order['id'])

                        if buy is None or sell is None:
                            debug_messages.append(f"Error: Buy or Sell order not found for IDs: {buy_order['id']}, {sell_order['id']}.")
                            continue

                        buy.amount = buy_order['amount']
                        sell.amount = sell_order['amount']

                        # Update matched amounts for both orders
                        buy.matched_amount = (buy.matched_amount or 0) + match_amount
                        sell.matched_amount = (sell.matched_amount or 0) + match_amount

                        # Set statuses based on remaining amounts
                        if buy.amount == 0:
                            buy.status = 'Matched'
                            debug_messages.append(f"Buy Order ID={buy.id} fully matched and status updated to Matched.")
                        else:
                            buy.status = 'Market'
                            debug_messages.append(f"Buy Order ID={buy.id} partially matched and status updated to Market.")

                        if sell.amount == 0:
                            sell.status = 'Matched'
                            debug_messages.append(f"Sell Order ID={sell.id} fully matched and status updated to Matched.")
                        else:
                            sell.status = 'Market'
                            debug_messages.append(f"Sell Order ID={sell.id} partially matched and status updated to Market.")

                        # Link matched orders
                        buy.matched_order_id = sell.id
                        sell.matched_order_id = buy.id

                        db.session.add(buy)
                        db.session.add(sell)

                        debug_messages.append(
                            f"Matched Buy ID={buy.id} (Remaining Amount={buy.amount}) with "
                            f"Sell ID={sell.id} (Remaining Amount={sell.amount}) for Match Amount={match_amount}"
                        )
                    except Exception as e:
                        debug_messages.append(f"Error updating database records for Buy ID={buy_order['id']} or Sell ID={sell_order['id']}: {e}")

            # Update unmatched orders to 'Market' status
            try:
                for _, unmatched_order in group[group['amount'] > 0].iterrows():
                    unmatched = Order.query.get(unmatched_order['id'])
                    if unmatched and unmatched.status == 'Pending':
                        unmatched.status = 'Market'
                        db.session.add(unmatched)
                        debug_messages.append(f"Unmatched Order ID={unmatched.id} marked as Market.")
            except Exception as e:
                debug_messages.append(f"Error updating unmatched orders in group Value Date={value_date}, Currency={currency}: {e}")

        # Commit all changes
        try:
            db.session.commit()
            debug_messages.append("All changes committed to the database successfully.")
        except Exception as e:
            db.session.rollback()
            debug_messages.append(f"Error committing changes to the database: {e}")
            return {"debug_messages": debug_messages, "message": "Failed to commit changes to the database."}

        return {"debug_messages": debug_messages, "message": "Matching process executed successfully"}

    except Exception as e:
        db.session.rollback()
        debug_messages.append(f"Unexpected error: {str(e)}")
        return {"debug_messages": debug_messages, "message": "Matching process failed due to an unexpected error."}
     

@admin_bp.route('/matched_orders', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def view_matched_orders():
    # Query orders with status 'Matched'
    matched_orders = Order.query.filter(Order.status == 'Matched').all()

    # Prepare the list of matched orders to return
    order_list = []
    for order in matched_orders:
        # Check for matched_order_id to determine the linked order
        matched_order = Order.query.get(order.matched_order_id) if order.matched_order_id else None
        
        # Add order details
        order_list.append({
            "id": order.id,
            "buyer": order.user.email if order.transaction_type == 'buy' else (matched_order.user.email if matched_order else None),
            "seller": order.user.email if order.transaction_type == 'sell' else (matched_order.user.email if matched_order else None),
            "currency": order.currency,
            "matched_amount": order.matched_amount,
            "amount": order.original_amount,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "execution_rate": order.execution_rate or "",  # Empty string if None
            "bank_name": order.bank_name or "",  # Empty string if None
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "order_date": order.order_date.strftime("%Y-%m-%d"),
        })

    return jsonify(order_list), 200


@admin_bp.route('/market_orders', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def view_market_orders():
    """
    API to view orders with 'Market' status and their remaining unmatched amount.
    """
    try:
        # Fetch all orders with status 'Market'
        market_orders = Order.query.filter(Order.status.in_(['Market', 'Executed']),Order.deleted == False).all()

        if not market_orders:
            return jsonify([]), 200

        # Prepare the list of market orders to return
        order_list = []
        for order in market_orders:
            order_list.append({
                 "id": order.id,
                "transaction_type": order.transaction_type,
                "currency": order.currency,
                "amount": order.amount,
                "status": order.status,
                "execution_rate": order.execution_rate or "",  # Empty string if None
                "bank_name": order.bank_name or "",  # Empty string if None
                "value_date": order.value_date.strftime("%Y-%m-%d"),
                "order_date": order.order_date.strftime("%Y-%m-%d"),
                "client": order.user.email,
                "client_name": order.user.client_name,  

            })

        return jsonify(order_list), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# def register_admin_jobs(scheduler, app):
#     """
#     Register background jobs specific to admin functionality.
#     """
#     # Add a job for scheduled matching (run daily at 4:14 PM)
#     scheduler.add_job(
#         func=scheduled_matching,
#         trigger='cron',
#         hour=16,
#         minute=14,
#         args=[app],
#         id="scheduled_matching_job"
#     )

@admin_bp.route('/upload-file', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = pd.read_excel(file)
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Save to the database without renaming columns
    df.to_sql('exchange_data', db.engine, if_exists='replace', index=False)
    return jsonify({"message": "File uploaded successfully"}), 200


@admin_bp.route('/logs', methods=['GET'])
@jwt_required()
@roles_required('Admin')  # Restrict access to admins only
def get_logs():
    """
    API to retrieve audit logs. Allows filtering by action type, table name, user ID, and date range.
    """
    # Get filters from query parameters
    action_type = request.args.get('action_type')
    table_name = request.args.get('table_name')
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Start query with all logs
    query = AuditLog.query

    # Apply filters if provided
    if action_type:
        query = query.filter(AuditLog.action_type == action_type)
    if table_name:
        query = query.filter(AuditLog.table_name == table_name)
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if start_date:
        query = query.filter(AuditLog.timestamp >= datetime.strptime(start_date, "%Y-%m-%d"))
    if end_date:
        query = query.filter(AuditLog.timestamp <= datetime.strptime(end_date, "%Y-%m-%d"))

    # Execute query and retrieve logs
    logs = query.order_by(AuditLog.timestamp.desc()).all()

    # Format logs for JSON response
    log_list = [{
        "id": log.id,
        "action_type": log.action_type,
        "table_name": log.table_name,
        "record_id": log.record_id,
        "user_id": log.user_id,
        "user_email": log.user.email if log.user else "Unknown",
        "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "details": log.details
    } for log in logs]

    return jsonify(log_list), 200

def process_uploaded_file(df):
    try:
        uploaded_count = 0
        updated_count = 0

        for _, row in df.iterrows():
            try:
                # Match client name to a user
                client_name = row["Client"]
                user = User.query.filter_by(client_name=client_name).first()

                if not user:
                    raise Exception(f"No user found for client: {client_name}")

                # Check if an order with the same details already exists
                existing_order = Order.query.filter_by(
                    transaction_date=row["Transaction date"],
                    value_date=row["Value date"],
                    currency=row["Currency"],
                    transaction_type=row["Type"],
                    amount=row["Amount"],
                    user_id=user.id
                ).first()

                if existing_order:
                    # Update the existing order
                    existing_order.execution_rate = row["Execution rate"]
                    existing_order.interbank_rate = row["Interbancaire"]  # Update interbank rate
                    existing_order.historical_loss = row["Historical Loss"]  # Update historical loss
                    existing_order.bank_name = row["Bank"]
                    updated_count += 1
                else:
                    # Create a new order
                    new_order = Order(
                        transaction_date=row["Transaction date"],
                        value_date=row["Value date"],
                        currency=row["Currency"],
                        transaction_type=row["Type"],
                        amount=row["Amount"],
                        original_amount=row["Amount"],  # Save the initial amount
                        execution_rate=row["Execution rate"],
                        interbank_rate=row["Interbancaire"],  # Save interbank rate
                        historical_loss=row["Historical Loss"],  # Save historical loss
                        bank_name=row["Bank"],
                        status="Executed",  # Set the default status to "Executed"
                        user=user,
                        order_date=datetime.now()
                    )
                    db.session.add(new_order)
                    uploaded_count += 1

            except Exception as e:
                raise Exception(f"Error processing row: {e}")

        db.session.commit()

        return {
            "uploaded": uploaded_count,
            "updated": updated_count
        }

    except Exception as e:
        db.session.rollback()
        raise Exception(f"Bulk upload failed: {str(e)}")



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xls', 'xlsx'}

import io

@admin_bp.route('/upload-orders', methods=['POST'])
@jwt_required()
@roles_required('Admin')  # Ensure only Admins can access this route
def upload_orders():
    """
    API for admins to upload orders in bulk for multiple clients via an Excel file.
    """
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        return jsonify({'error': 'Invalid file format. Please upload an Excel file.'}), 400

    file = request.files['file']

    try:
        # Read the uploaded file using BytesIO for compatibility with pandas
        file_stream = io.BytesIO(file.read())

        # Load the Excel file into a DataFrame
        df = pd.read_excel(file_stream)

        # Validate required columns
        required_columns = [
            "Client", "Transaction date", "Value date", 
            "Currency", "Type", "Amount", "Execution rate", 
            "Bank", "Interbancaire", "Historical Loss"
        ]
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': f'Missing required columns. Ensure {required_columns} are present.'}), 400

        # Data Cleaning and Transformation
        df["Transaction date"] = pd.to_datetime(df["Transaction date"])
        df["Value date"] = pd.to_datetime(df["Value date"])
        df["Amount"] = df["Amount"].replace(",", "", regex=True).astype(float)
        df["Execution rate"] = df["Execution rate"].replace(",", ".", regex=True).astype(float)
        df["Interbancaire"] = df["Interbancaire"].replace(",", ".", regex=True).astype(float)
        df["Historical Loss"] = df["Historical Loss"].replace(",", ".", regex=True).astype(float)  # Clean and convert historical loss

        # Process the uploaded data
        result = process_uploaded_file(df)

        return jsonify({
            'message': 'Orders uploaded successfully',
            'uploaded_count': result["uploaded"],
            'updated_count': result["updated"]
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500


import eikon as ek
from models import db, YieldsData
from datetime import date

@admin_bp.route('/api/fetch-yields', methods=['POST'])
def fetch_yields():
    """
    Fetch TND, EUR, USD yields from Eikon from August 1 until today, 
    then store them in the yields_data table.
    Trigger it once via Postman (POST).
    """
    try:
        # 1) Define date range
        start_date = "2023-08-01"
        end_date = date.today().strftime("%Y-%m-%d")

        # 2) List of instruments (TNDOND, EUR1MD, etc.)
        # We'll do multiple calls to get_timeseries 
        # because each code might have separate timeseries.
        instruments = {
            "TNDOND": ["tnd_1m", "tnd_6m", "tnd_9m"],  # All the same
            "EUR1MD": "eur_1m",
            "EUR6MD": "eur_6m",
            "EUR9MD": "eur_9m",
            "USD1MD": "usd_1m",
            "USD6MD": "usd_6m",
            "USD9MD": "usd_9m",
        }

        # We'll store results in a dict keyed by date, each date is a sub-dict of yields
        data_by_date = {}

        for ric, columns in instruments.items():
            # get daily timeseries from Eikon for that RIC
            df = ek.get_timeseries(
                ric, 
                start_date=start_date, 
                end_date=end_date, 
                interval="daily"
            )
            # 'df' has a DateTime index and typically an 'CLOSE' or 'Value' column 
            # depending on the instrument. Let's see how Eikon returns it.
            # For TNDOND, etc., it might be "CLOSE" or "HIGH" or just 0. We'll assume "CLOSE".

            if df is None or df.empty:
                continue  # no data

            for ts_index, row in df.iterrows():
                d = ts_index.date()  # Convert to Python date (ts_index is a Timestamp)
                # Usually the yield is in row["CLOSE"] if there's only one column
                val = None
                if "CLOSE" in row:
                    val = row["CLOSE"]
                else:
                    # fallback if it's named differently
                    val = row.get("Value", None)

                if val is None:
                    continue

                if d not in data_by_date:
                    data_by_date[d] = {
                        "tnd_1m": None,
                        "tnd_6m": None,
                        "tnd_9m": None,
                        "eur_1m": None,
                        "eur_6m": None,
                        "eur_9m": None,
                        "usd_1m": None,
                        "usd_6m": None,
                        "usd_9m": None,
                    }

                if isinstance(columns, list):
                    # This means TNDOND => fill all 3 columns with the same value
                    for col in columns:
                        data_by_date[d][col] = float(val)
                else:
                    data_by_date[d][columns] = float(val)

        # 3) Insert rows into yields_data table
        for d, yields_dict in data_by_date.items():
            # Check if row for this date already exists (in case you re-run)
            existing = YieldsData.query.filter_by(date=d).first()
            if existing:
                # Just update
                existing.tnd_1m = yields_dict["tnd_1m"]
                existing.tnd_6m = yields_dict["tnd_6m"]
                existing.tnd_9m = yields_dict["tnd_9m"]
                existing.eur_1m = yields_dict["eur_1m"]
                existing.eur_6m = yields_dict["eur_6m"]
                existing.eur_9m = yields_dict["eur_9m"]
                existing.usd_1m = yields_dict["usd_1m"]
                existing.usd_6m = yields_dict["usd_6m"]
                existing.usd_9m = yields_dict["usd_9m"]
            else:
                new_row = YieldsData(
                    date=d,
                    tnd_1m=yields_dict["tnd_1m"],
                    tnd_6m=yields_dict["tnd_6m"],
                    tnd_9m=yields_dict["tnd_9m"],
                    eur_1m=yields_dict["eur_1m"],
                    eur_6m=yields_dict["eur_6m"],
                    eur_9m=yields_dict["eur_9m"],
                    usd_1m=yields_dict["usd_1m"],
                    usd_6m=yields_dict["usd_6m"],
                    usd_9m=yields_dict["usd_9m"],
                )
                db.session.add(new_row)

        db.session.commit()
        return jsonify({"message": "Yields fetched and stored successfully"}), 200

    except ek.EikonError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

@admin_bp.route('/api/fetch-yields', methods=['GET'])
def get_yields():
    """
    Retrieve yields from the yields_data table
    for a given date range or the entire set.
    Example usage:
      GET /api/fetch-yields?start=2023-08-01&end=2023-09-30
    """
    start_str = request.args.get("start", "2023-08-01")
    end_str   = request.args.get("end", date.today().strftime("%Y-%m-%d"))

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date   = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format, use YYYY-MM-DD"}), 400

    rows = (
        YieldsData.query
        .filter(YieldsData.date >= start_date, YieldsData.date <= end_date)
        .order_by(YieldsData.date)
        .all()
    )

    output = []
    for r in rows:
        output.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "tnd_1m": r.tnd_1m,
            "tnd_6m": r.tnd_6m,
            "tnd_9m": r.tnd_9m,
            "eur_1m": r.eur_1m,
            "eur_6m": r.eur_6m,
            "eur_9m": r.eur_9m,
            "usd_1m": r.usd_1m,
            "usd_6m": r.usd_6m,
            "usd_9m": r.usd_9m,
        })

    return jsonify(output), 200
