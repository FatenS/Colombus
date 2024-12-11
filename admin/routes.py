import base64
import io
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
from .services.order_service import generate_unique_key, scheduled_matching
from .services.export_service import export_pdf, download_excel
from .services.meeting_service import get_meetings_for_month, generate_month_days
from models import db, Order, ExchangeData
from flask_login import login_user, logout_user, current_user
from user.services.user_service import UserService
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
    role_id = request.json.get('options')
    rating = request.json.get('rating', 0)  # Get rating from the request, default to 0 if not provided

    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    user = User(email=email, active=1, password=hashed_password, rating=rating)  # Set the rating when creating the user
    
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
            "status": order.status,
            "execution_rate": order.execution_rate,  # Admin can see the execution rate
            "bank_name": order.bank_name,  # Admin can see the bank name
            
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
                    'rating': order.rating
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
                "client": order.user.email
            })

        return jsonify(order_list), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def register_admin_jobs(scheduler, app):
    scheduler.add_job(scheduled_matching, 'cron', hour=16, minute=14, args=[app])

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
