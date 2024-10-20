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
from models import db, Order, MatchedPosition, Meeting
from flask_login import login_user, logout_user, current_user
from user.services.user_service import UserService
from functools import wraps
from models import User, Role
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
        access_token = create_access_token(identity=user.id)

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
    orders = Order.query.options(joinedload(Order.user)).all()

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
    
    db.session.commit()  # Save changes
    
    return jsonify({"message": "Order updated successfully"}), 200

"""
@admin_bp.route('/run_matching', methods=['POST'])
@jwt_required()
@roles_required('Admin')  # Enforces Admin role

def run_matching():
   
    try:
        # Call the scheduled matching process
        scheduled_matching(current_app)
        return jsonify({'message': 'Matching process executed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def scheduled_matching(app):
        today = datetime.now()
        future_date = today + timedelta(days=2)

        # Query the database for orders that meet the conditions
        orders_query = Order.query.filter(func.date(Order.value_date) > future_date, Order.status == 'Pending').all()

        # Delete pending orders that match the condition (value_date > future_date)
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

            # Group by 'Value Date' and 'Currency'
            for (value_date, currency), group in orders_df.groupby(['Value Date', 'Currency']):
                buy_orders = group[group['Type'] == 'buy'].sort_values(by=['Rating', 'Order Dates', 'Transaction Amount'], ascending=[False, True, False])
                sell_orders = group[group['Type'] == 'sell'].sort_values(by=['Rating', 'Order Dates', 'Transaction Amount'], ascending=[False, True, False])

                buy_index, sell_index = 0, 0
                while buy_index < len(buy_orders) and sell_index < len(sell_orders):
                    buy_order = buy_orders.iloc[buy_index]
                    sell_order = sell_orders.iloc[sell_index]
                    match_amount = min(buy_order['Transaction Amount'], sell_order['Transaction Amount'])

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

                    # Update the status of the matched orders in the Order table
                    buy_order_instance = Order.query.get(buy_order['ID'])  # Get buy order from the DB
                    sell_order_instance = Order.query.get(sell_order['ID'])  # Get sell order from the DB

                    # Update their statuses
                    buy_order_instance.status = 'Matched'
                    sell_order_instance.status = 'Matched'

                    # Add them to the session for updating
                    db.session.add(buy_order_instance)
                    db.session.add(sell_order_instance)

                    # Commit changes after processing each match
                    db.session.commit()

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

                filtered_buy_orders = buy_orders[buy_orders['Transaction Amount'] > 0]
                filtered_sell_orders = sell_orders[sell_orders['Transaction Amount'] > 0]
                remaining_orders = pd.concat([filtered_buy_orders, filtered_sell_orders])
                remaining_orders['Status'] = 'Market'
                remaining.append(remaining_orders)

            update_match_df = pd.DataFrame(update_match)
            remaining_orders_df = pd.concat(remaining, ignore_index=True)
            all_updates_df = pd.concat([update_match_df, remaining_orders_df], ignore_index=True)

            for index, row in all_updates_df.iterrows():
                new_order = Order(
                    id_unique=str(row['ID']),
                    transaction_type=row['Type'],
                    amount=row['Transaction Amount'],
                    currency=row['Currency'],
                    value_date=row['Value Date'].to_pydatetime(),
                    order_date=row['Order Dates'].to_pydatetime(),
                    bank_account=row['Bank Account'],
                    reference=row['reference'],
                    user=row['Client'],
                    status=row['Status'],
                    rating=row.get('Rating', 0)
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

            db.session.commit()

        else:
            print("No pending orders found for matching.")


@admin_bp.route('/matched_orders', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def view_matched_orders():
    
    matched_orders = Order.query.filter_by(status='Matched').all()
    
    # Prepare the list of matched orders to return
    order_list = []
    for order in matched_orders:
        order_list.append({
            "id": order.id,
            "user": order.user.email if order.user else "Unknown",
            "transaction_type": order.transaction_type,
            "amount": order.amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "execution_rate": order.execution_rate,
            "bank_name": order.bank_name,
        })

    return jsonify(order_list), 200
 """

@admin_bp.route('/run_matching', methods=['POST'])
@jwt_required()
@roles_required('Admin')  # Enforces Admin role
def run_matching():
    """
    API to trigger the scheduled matching process manually.
    This allows Admins to run the matching process via a REST API call.
    """
    try:
        # Call the matching function
        manual_matching(current_app)
        return jsonify({'message': 'Matching process executed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500




def manual_matching(app):
    with app.app_context():
        today = datetime.now()

        # Fetch orders that are pending and valid for matching
        orders_query = Order.query.filter(
            Order.status == 'Pending',
            func.date(Order.value_date) >= today
        ).all()

        # Prepare order dictionaries
        orders_dicts = [
            {
                'ID': order.id,
                'Type': order.transaction_type,
                'Transaction Amount': order.amount,  # This will be the remaining amount
                'Currency': order.currency,
                'Value Date': order.value_date,
                'Order Dates': order.order_date,
                'Bank Account': order.bank_account,
                'reference': order.reference,
                'Client': order.user.email,
                'Status': order.status,
                'Rating': order.rating
            } for order in orders_query
        ]

        orders_df = pd.DataFrame(orders_dicts)

        if len(orders_df) > 0:
            # Proceed with matching logic
            orders_df['Value Date'] = pd.to_datetime(orders_df['Value Date'])
            orders_df['Order Dates'] = pd.to_datetime(orders_df['Order Dates'])

            matches = []
            remaining = []

            # Group by 'Value Date' and 'Currency'
            for (value_date, currency), group in orders_df.groupby(['Value Date', 'Currency']):
                buy_orders = group[group['Type'] == 'buy'].sort_values(
                    by=['Rating', 'Order Dates', 'Transaction Amount'],
                    ascending=[False, True, False]
                )
                sell_orders = group[group['Type'] == 'sell'].sort_values(
                    by=['Rating', 'Order Dates', 'Transaction Amount'],
                    ascending=[False, True, False]
                )

                buy_index, sell_index = 0, 0
                while buy_index < len(buy_orders) and sell_index < len(sell_orders):
                    buy_order = buy_orders.iloc[buy_index]
                    sell_order = sell_orders.iloc[sell_index]
                    
                    # Conversion to standard Python float here for match_amount calculation
                    match_amount = min(float(buy_order['Transaction Amount']), float(sell_order['Transaction Amount']))

                    # Create match record
                    match = {
                        'ID': buy_order['ID'],
                        'Value Date': value_date.strftime('%Y-%m-%d'),
                        'Currency': currency,
                        'Buyer': buy_order['Client'],
                        'Buyer Rating': buy_order['Rating'],
                        'Seller': sell_order['Client'],
                        'Seller Rating': sell_order['Rating'],
                        'Matched Amount': match_amount,
                    }
                    matches.append(match)

                    # Adjust transaction amounts with float conversion
                    buy_orders.at[buy_orders.index[buy_index], 'Transaction Amount'] -= match_amount
                    sell_orders.at[sell_orders.index[sell_index], 'Transaction Amount'] -= match_amount

                    # Fetch order instances from the database
                    buy_order_instance = Order.query.get(buy_order['ID'])
                    sell_order_instance = Order.query.get(sell_order['ID'])

                    if buy_order_instance is None or sell_order_instance is None:
                        return jsonify({'error': 'Matching failed: Buy or Sell order not found'}), 400

                    # Convert the amounts to Python float before updating the database
                    buy_order_instance.amount = float(buy_orders.iloc[buy_index]['Transaction Amount'])
                    sell_order_instance.amount = float(sell_orders.iloc[sell_index]['Transaction Amount'])

                    # Update the status if fully matched
                    if buy_order_instance.amount == 0:
                        buy_order_instance.status = 'Matched'
                    if sell_order_instance.amount == 0:
                        sell_order_instance.status = 'Matched'

                    # Commit the changes to the database
                    db.session.add(buy_order_instance)
                    db.session.add(sell_order_instance)
                    db.session.commit()

                    # Move to the next order if fully matched
                    if buy_orders.iloc[buy_index]['Transaction Amount'] == 0:
                        buy_index += 1
                    if sell_orders.iloc[sell_index]['Transaction Amount'] == 0:
                        sell_index += 1

                # Handle remaining unmatched orders
                filtered_buy_orders = buy_orders[buy_orders['Transaction Amount'] > 0]
                filtered_sell_orders = sell_orders[sell_orders['Transaction Amount'] > 0]
                remaining_orders = pd.concat([filtered_buy_orders, filtered_sell_orders])

                if not remaining_orders.empty:
                    remaining_orders['Status'] = 'Market'

                    # Update the status and remaining amount of unmatched orders in the database
                    for _, remaining_order in remaining_orders.iterrows():
                        order_instance = Order.query.get(remaining_order['ID'])
                        if order_instance:
                            order_instance.status = 'Market'
                            # Ensure we store Python float
                            order_instance.amount = float(remaining_order['Transaction Amount'])
                            db.session.add(order_instance)

                    db.session.commit()
                    remaining.append(remaining_orders)

            # Update MatchedPosition table with matched orders
            for match in matches:
                matched_order = MatchedPosition(
                    id=str(match['ID']),
                    value_date=datetime.strptime(match['Value Date'], '%Y-%m-%d').date(),
                    currency=match['Currency'],
                    buyer=match['Buyer'],
                    buyer_rate=int(match.get('Buyer Rating', 0)),
                    seller=match['Seller'],
                    seller_rate=int(match.get('Seller Rating', 0)),
                    matched_amount=float(match['Matched Amount']),
                    buy_order_id=Order.query.get(match['ID']).id,
                    sell_order_id=Order.query.get(match['ID']).id
                )
                db.session.add(matched_order)

            db.session.commit()




@admin_bp.route('/matched_orders', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def view_matched_orders():
    """
    API to view all matched orders in the system.
    """
    matched_orders = MatchedPosition.query.all()

    # Prepare the list of matched orders to return
    order_list = []
    for order in matched_orders:
        order_list.append({
            "id": order.id,
            "buyer": order.buyer,
            "seller": order.seller,
            "currency": order.currency,
            "matched_amount": order.matched_amount,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
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
        market_orders = Order.query.filter(Order.status.in_(['Market', 'Executed'])).all()

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
