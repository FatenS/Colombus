from datetime import datetime, timedelta, date
from scipy.interpolate import interp1d, CubicSpline
from models import db, User,  Order, AuditLog, ExchangeData, OpenPosition, PremiumRate, InterbankRate
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
# @user_bp.route('/orders', methods=['POST'])
# @jwt_required()
# def submit_order_or_option():
#     """
#     Handles both normal orders AND options in a single endpoint.
#     - If is_option == True => automatically calculates premium from PremiumRate
#     - If is_option == False => normal order (premium=None)
#     Logs debug information and writes an AuditLog.
#     """

#     debug_logs = []
#     user_id = get_jwt_identity()
#     debug_logs.append(f"Fetched user ID from JWT: {user_id}")

#     user = User.query.get(user_id)
#     if not user:
#         debug_logs.append("User not found in the database")
#         return jsonify({"message": "Invalid user", "debug": debug_logs}), 400
#     debug_logs.append(f"User found: {user.email}")

#     # Parse JSON body
#     data = request.get_json()
#     if not data:
#         debug_logs.append("Request payload is empty")
#         return jsonify({"message": "No data provided", "debug": debug_logs}), 400

#     debug_logs.append(f"Raw incoming data: {data}")

#     # Gather relevant fields
#     transaction_type = data.get('transaction_type')    # "buy" / "sell"
#     amount = data.get('amount', 0)                     # notional
#     currency = data.get('currency')                    # "USD", "EUR", ...
#     value_date_str = data.get('value_date')            # e.g., "2025-05-15"
#     bank_account = data.get('bank_account')            # optional
#     is_option = data.get('is_option', False)           # boolean

#     # Basic validations
#     if not transaction_type or not currency or not value_date_str:
#         debug_logs.append("Missing required fields (transaction_type, currency, value_date).")
#         return jsonify({"message": "Missing required fields", "debug": debug_logs}), 400

#     # Convert fields
#     try:
#         amount = float(amount)
#         value_date = datetime.strptime(value_date_str, "%Y-%m-%d")
#         debug_logs.append(
#             f"Parsed data: transaction_type={transaction_type}, amount={amount}, "
#             f"currency={currency}, value_date={value_date}, bank_account={bank_account}, "
#             f"is_option={is_option}"
#         )
#     except Exception as e:
#         debug_logs.append(f"Error parsing data: {str(e)}")
#         return jsonify({"message": "Invalid data format", "debug": debug_logs}), 400

#     # Decide status: if Option => "Market" (skip matching). Else => "Pending" (normal order).
#     if is_option:
#         status = "Market"
#     else:
#         status = "Pending"

#     # ---------------------------------------------------------------------------
#     #  AUTOMATIC PREMIUM CALCULATION IF is_option == True
#     # ---------------------------------------------------------------------------
#     computed_premium = None
#     if is_option:
#         # 1) Figure out days between 'today' and 'value_date'
#         today = datetime.now().date()
#         days_diff = (value_date.date() - today).days  # e.g., 90, 120, etc.

#         # 2) Query your PremiumRate table for the same currency
#         all_rates = PremiumRate.query.filter_by(currency=currency.upper()).all()
#         if not all_rates:
#             debug_logs.append(f"No premium rates found for currency={currency}.")
#             return jsonify({"message": f"No premium rates found for {currency}", "debug": debug_logs}), 400

#         # 3) Find the row with the closest 'maturity_days' to 'days_diff'
#         chosen_rate = None
#         min_delta = float("inf")
#         for r in all_rates:
#             delta = abs(r.maturity_days - days_diff)
#             if delta < min_delta:
#                 min_delta = delta
#                 chosen_rate = r

#         if not chosen_rate:
#             debug_logs.append("Unable to find suitable premium rate.")
#             return jsonify({"message": "Unable to find suitable premium rate", "debug": debug_logs}), 400

#         # 4) Calculate premium = notional * premium_percentage
#         computed_premium = amount * chosen_rate.premium_percentage
#         debug_logs.append(
#             f"Calculated premium: notional={amount}, days_diff={days_diff}, "
#             f"picked maturity={chosen_rate.maturity_days}, "
#             f"premium_percentage={chosen_rate.premium_percentage}, "
#             f"final_premium={computed_premium}"
#         )

#     # Generate a unique ID for the order (if desired)
#     unique_id = str(uuid.uuid4())
#     debug_logs.append(f"Generated unique order ID: {unique_id}")

#     # Create the Order object
#     try:
#         new_order = Order(
#             id_unique=unique_id,
#             user=user,
#             transaction_type=transaction_type,
#             amount=amount,
#             original_amount=amount,
#             currency=currency,
#             value_date=value_date,
#             transaction_date=datetime.now(),
#             order_date=datetime.now(),
#             bank_account=bank_account,
#             reference=data.get('reference', f'REF-{unique_id}'),
#             status=status,
#             rating=user.rating,
#             premium=computed_premium,  # The computed premium or None
#             is_option=is_option
#         )
#         debug_logs.append(f"Order/Option object created: {new_order}")
#     except Exception as e:
#         debug_logs.append(f"Error creating Order object: {str(e)}")
#         return jsonify({"message": "Error creating Order object", "debug": debug_logs}), 500

#     # Save in DB
#     try:
#         db.session.add(new_order)
#         db.session.commit()
#         debug_logs.append("Order/Option saved to the database")
#     except Exception as e:
#         db.session.rollback()
#         debug_logs.append(f"Database error: {str(e)}")
#         return jsonify({"message": "Database error", "debug": debug_logs}), 500

#     # Log the action in AuditLog
#     try:
#         log = AuditLog(
#             action_type='create',
#             table_name='order',
#             record_id=new_order.id_unique,
#             user_id=user_id,
#             details=json.dumps({
#                 "id": new_order.id_unique,
#                 "transaction_type": transaction_type,
#                 "amount": amount,
#                 "currency": currency,
#                 "value_date": value_date_str,
#                 "bank_account": bank_account,
#                 "is_option": is_option,
#                 "premium": computed_premium
#             })
#         )
#         db.session.add(log)
#         db.session.commit()
#         debug_logs.append("Audit log saved to the database")
#     except Exception as e:
#         db.session.rollback()
#         debug_logs.append(f"Error saving audit log: {str(e)}")
#         return jsonify({"message": "Error saving audit log", "debug": debug_logs}), 500

#     # Return success
#     return jsonify({
#         "message": "Order/Option submitted successfully",
#         "order_id": new_order.id_unique,
#         "premium": computed_premium,
#         "debug": debug_logs
#     }), 201

@user_bp.route('/orders', methods=['POST'])
@jwt_required()
def submit_order_or_option():
    """
    Handles both normal orders AND options in a single endpoint.
    - For options, it calculates the forward rate, determines moneyness based on strike,
      interpolates the premium percentage, and computes the premium.
    - For non-options, it creates a normal order.
    Logs debug information and writes an AuditLog.
    """
    debug_logs = []
    user_id = get_jwt_identity()
    debug_logs.append(f"Fetched user ID from JWT: {user_id}")

    user = User.query.get(user_id)
    if not user:
        debug_logs.append("User not found in the database")
        return jsonify({"message": "Invalid user", "debug": debug_logs}), 400
    debug_logs.append(f"User found: {user.email}")

    data = request.get_json()
    if not data:
        debug_logs.append("Request payload is empty")
        return jsonify({"message": "No data provided", "debug": debug_logs}), 400

    debug_logs.append(f"Raw incoming data: {data}")

    # Gather basic fields
    transaction_type = data.get('transaction_type')       # "buy" / "sell"
    amount = data.get('amount', 0)
    currency = data.get('currency')
    value_date_str = data.get('value_date')                # expected "YYYY-MM-DD"
    bank_account = data.get('bank_account')
    is_option = data.get('is_option', False)
    
    # For options, also accept call/put and strike
    option_type = data.get('option_type', '').upper()      # "CALL" or "PUT"
    user_strike = data.get('strike', None)                 # may be None

    if not transaction_type or not currency or not value_date_str:
        debug_logs.append("Missing required fields (transaction_type, currency, value_date).")
        return jsonify({"message": "Missing required fields", "debug": debug_logs}), 400

    try:
        amount = float(amount)
        value_date = datetime.strptime(value_date_str, "%Y-%m-%d")
        debug_logs.append(
            f"Parsed data: transaction_type={transaction_type}, amount={amount}, "
            f"currency={currency}, value_date={value_date}, bank_account={bank_account}, "
            f"is_option={is_option}, option_type={option_type}, strike={user_strike}"
        )
    except Exception as e:
        debug_logs.append(f"Error parsing data: {str(e)}")
        return jsonify({"message": "Invalid data format", "debug": debug_logs}), 400

    # Set trade_type based on whether it's an option (store "option" for options, else "spot")
    trade_type = "option" if is_option else "spot"
    status = "Market" if is_option else "Pending"

    computed_premium = None
    moneyness = None
    computed_forward = None
    final_strike = None

    if is_option:
        # Validate option type
        if option_type not in ["CALL", "PUT"]:
            debug_logs.append(f"Invalid option_type: {option_type}")
            return jsonify({"message": "Option type must be CALL or PUT", "debug": debug_logs}), 400

        # 1) Calculate days until maturity
        today = datetime.now().date()
        days_diff = (value_date.date() - today).days
        debug_logs.append(f"Days until maturity: {days_diff}")

        # 2) Retrieve today's exchange data to compute forward rate
        exchange_data = ExchangeData.query.filter_by(date=today).first()
        if not exchange_data:
            debug_logs.append("No exchange data found for today's date")
            return jsonify({"message": "Exchange data not available", "debug": debug_logs}), 400

        if currency.upper() == "USD":
            spot_rate = exchange_data.spot_usd
            yield_foreign = getattr(exchange_data, f"usd_{get_yield_period(days_diff)[0]}m")
        elif currency.upper() == "EUR":
            spot_rate = exchange_data.spot_eur
            yield_foreign = getattr(exchange_data, f"eur_{get_yield_period(days_diff)[0]}m")
        else:
            debug_logs.append(f"Unsupported currency: {currency}")
            return jsonify({"message": "Unsupported currency", "debug": debug_logs}), 400

        yield_domestic = getattr(exchange_data, f"tnd_{get_yield_period(days_diff)[0]}m")
        computed_forward = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)
        debug_logs.append(f"Computed forward rate: {computed_forward}")

        # 3) Determine the effective strike
        if user_strike is not None:
            try:
                final_strike = float(user_strike)
            except Exception as e:
                debug_logs.append(f"Error parsing strike: {e}")
                return jsonify({"message": "Strike must be numeric", "debug": debug_logs}), 400
        else:
            final_strike = computed_forward

        # 4) Determine moneyness by comparing computed forward and strike
        tol = 0.01 * computed_forward
        if option_type == "CALL":
            if computed_forward > final_strike + tol:
                moneyness = "in the money"
            elif abs(computed_forward - final_strike) <= tol:
                moneyness = "at the money"
            else:
                moneyness = "out of the money"
        else:  # PUT
            if computed_forward < final_strike - tol:
                moneyness = "in the money"
            elif abs(computed_forward - final_strike) <= tol:
                moneyness = "at the money"
            else:
                moneyness = "out of the money"
        debug_logs.append(f"Final strike: {final_strike}, Moneyness: {moneyness}")

        # 5) Fetch PremiumRate records filtered by currency, option_type, and transaction_type
        premium_rates = PremiumRate.query.filter_by(
            currency=currency.upper(),
            option_type=option_type,
            transaction_type=transaction_type.lower()
        ).all()
        if not premium_rates:
            debug_logs.append(f"No premium rates found for {currency} {option_type} {transaction_type}")
            return jsonify({"message": f"No premium rates found for {currency} {option_type} {transaction_type}", "debug": debug_logs}), 400

        # 6) Interpolation: calculate time-to-maturity (ttm) in years
        ttm = calculate_time_to_maturity(datetime.now().strftime('%d/%m/%Y'), value_date.strftime('%d/%m/%Y'))
        debug_logs.append(f"Calculated time-to-maturity (years): {ttm}")

        # Prepare arrays for interpolation
        known_times = np.array([r.maturity_days / 365.0 for r in premium_rates])
        known_primes = np.array([r.premium_percentage for r in premium_rates])
        debug_logs.append(f"Known times (years): {known_times}")
        debug_logs.append(f"Known premium percentages: {known_primes}")

        interpolated = interpolate_prime(ttm, known_times, known_primes)
        chosen_rate = float(interpolated['Linear'])
        computed_premium = amount * chosen_rate
        debug_logs.append(
            f"Interpolated premium rate: {chosen_rate}, Computed premium: {computed_premium}"
        )
    # End of is_option block

    unique_id = str(uuid.uuid4())
    debug_logs.append(f"Generated unique order ID: {unique_id}")

    try:
        new_order = Order(
            id_unique=unique_id,
            user=user,
            transaction_type=transaction_type,
            trade_type=trade_type,        # "option" for options, "spot" otherwise
            amount=amount,
            original_amount=amount,
            currency=currency,
            value_date=value_date,
            transaction_date=datetime.now(),
            order_date=datetime.now(),
            bank_account=bank_account,
            reference=data.get('reference', f'REF-{unique_id}'),
            status=status,
            rating=user.rating,
            premium=computed_premium,
            is_option=is_option,
            option_type=option_type if is_option else None,
            strike=final_strike if is_option else None,
            forward_rate=computed_forward if is_option else None,
            moneyness=moneyness
        )
        debug_logs.append(f"Order/Option object created: {new_order}")
    except Exception as e:
        debug_logs.append(f"Error creating Order object: {str(e)}")
        return jsonify({"message": "Error creating Order object", "debug": debug_logs}), 500

    try:
        db.session.add(new_order)
        db.session.commit()
        debug_logs.append("Order/Option saved to the database")
    except Exception as e:
        db.session.rollback()
        debug_logs.append(f"Database error: {str(e)}")
        return jsonify({"message": "Database error", "debug": debug_logs}), 500

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
                "value_date": value_date_str,
                "bank_account": bank_account,
                "is_option": is_option,
                "option_type": option_type,
                "strike": final_strike,
                "premium": computed_premium,
                "moneyness": moneyness,
            })
        )
        db.session.add(log)
        db.session.commit()
        debug_logs.append("Audit log saved to the database")
    except Exception as e:
        db.session.rollback()
        debug_logs.append(f"Error saving audit log: {str(e)}")
        return jsonify({"message": "Error saving audit log", "debug": debug_logs}), 500

    return jsonify({
        "message": "Order/Option submitted successfully",
        "order_id": new_order.id_unique,
        "premium": computed_premium,
        "moneyness": moneyness,
        "forward_rate": computed_forward,
        "debug": debug_logs
    }), 201


# =========================
# View Orders Endpoint (for clients)
# =========================
@user_bp.route('/orders', methods=['GET'])
@jwt_required()
def view_orders():
    user_id = get_jwt_identity() 
    orders = Order.query.filter_by(user_id=user_id, deleted=False).all()
    
    if not orders:
        return jsonify([]), 200  
    
    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "transaction_type": order.transaction_type,
            "trade_type": order.trade_type,  # NEW: include trade type
            "amount": order.original_amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "client_name": order.user.client_name,
            "premium": order.premium,       
            "is_option": order.is_option, 
            "option_type": order.option_type,   
            "strike": order.strike,             
            "moneyness": order.moneyness        
        })
    
    return jsonify(order_list), 200

# (Other endpoints remain largely unchanged – they can be updated similarly if needed)

# =========================
# Utility: Log Action
# =========================
def log_action(action_type, table_name, record_id, user_id, details):
    user = User.query.get(user_id)
    details["client_name"] = user.client_name  
    log_entry = AuditLog(
        action_type=action_type,
        table_name=table_name,
        record_id=record_id,
        user_id=user_id,
        timestamp=datetime.now(),
        details=json.dumps(details)
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

    # Update `amount` and `original_amount`
    if 'amount' in data and data['amount'] != order.amount:
        changes['original_amount'] = {"old": order.original_amount, "new": data['amount']}
        changes['amount'] = {"old": order.amount, "new": data['amount']}
        order.amount = data['amount']
        order.original_amount = data['amount']

    # Update currency
    if 'currency' in data and data['currency'] != order.currency:
        changes['currency'] = {"old": order.currency, "new": data['currency']}
        order.currency = data['currency']

    # Update value_date
    if 'value_date' in data:
        new_value_date = datetime.strptime(data['value_date'], "%Y-%m-%d")
        if new_value_date != order.value_date:
            changes['value_date'] = {"old": order.value_date.strftime("%Y-%m-%d"), "new": data['value_date']}
            order.value_date = new_value_date

    # Update bank_account
    if 'bank_account' in data and data['bank_account'] != order.bank_account:
        changes['bank_account'] = {"old": order.bank_account, "new": data['bank_account']}
        order.bank_account = data['bank_account']

    # Update reference
    if 'reference' in data and data['reference'] != order.reference:
        changes['reference'] = {"old": order.reference, "new": data['reference']}
        order.reference = data['reference']

    # NEW: Update trade_type if provided
    if 'trade_type' in data and data['trade_type'] != order.trade_type:
        changes['trade_type'] = {"old": order.trade_type, "new": data['trade_type']}
        order.trade_type = data['trade_type']

    # Optionally update option-specific fields if this is an option order
    if order.is_option:
        if 'option_type' in data and data['option_type'] != order.option_type:
            changes['option_type'] = {"old": order.option_type, "new": data['option_type']}
            order.option_type = data['option_type']
        if 'strike' in data:
            try:
                new_strike = float(data['strike']) if data['strike'] is not None else None
            except Exception:
                return jsonify({"error": "Strike must be numeric"}), 400
            if new_strike != order.strike:
                changes['strike'] = {"old": order.strike, "new": new_strike}
                order.strike = new_strike

    # Log the update action with all changes
    log_action(
        action_type='update',
        table_name='order',
        record_id=order.id_unique,
        user_id=user_id,
        details=changes
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
        Order.status.in_(['Executed', 'Matched'])
    ).order_by(Order.transaction_date).all()

    if not orders:
        return jsonify({"message": "No data available for this user"}), 200

    trend_data = []
    for order in orders:
        trend_data.append({
            "date": order.transaction_date.strftime('%Y-%m-%d'),
            "execution_rate_export": order.execution_rate if order.transaction_type.lower() in ["export", "sell"] else None,
            "execution_rate_import": order.execution_rate if order.transaction_type.lower() in ["import", "buy"] else None,
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

    # Filter orders by currency and status
    orders = Order.query.filter(
        Order.user_id == user_id,
        Order.currency == currency,
        Order.status.in_(['Executed', 'Matched'])
    ).all()

    # Process each order: calculate gain and gain_percentage and save in DB.
    for order in orders:
        try:
            benchmark = calculate_benchmark(order)
        except Exception as e:
            # If benchmark calculation fails, set it to None
            benchmark = None

        if benchmark and order.execution_rate:
            tx_type = order.transaction_type.lower()
            if tx_type in ["import", "buy"]:
                gain_percent = (benchmark / order.execution_rate) - 1
            else:
                gain_percent = (order.execution_rate / benchmark) - 1

            # Convert foreign notional to domestic (TND) using the execution_rate
            gain = gain_percent * order.original_amount * order.execution_rate

            order.gain = float(gain)
            order.gain_percentage = float(gain_percent * 100)
        else:
            order.gain = 0
            order.gain_percentage = 0

    # Commit updates to the database.
    db.session.commit()

    # Group orders by bank and month for the API response.
    bank_data = {}
    for order in orders:
        month = order.transaction_date.strftime('%Y-%m')
        bank = order.bank_name or "Unknown"
        if bank not in bank_data:
            bank_data[bank] = {}
        if month not in bank_data[bank]:
            bank_data[bank][month] = {'traded': 0, 'gain': 0, 'coverage': 0, 'count': 0}
        
        # Convert original_amount from foreign currency to TND.
        traded_tnd = order.original_amount * order.execution_rate if order.execution_rate else 0
        bank_data[bank][month]['traded'] += traded_tnd
        bank_data[bank][month]['count'] += 1
        bank_data[bank][month]['gain'] += order.gain

        hedge_status = calculate_hedge_status(order.transaction_date, order.value_date)
        if hedge_status == "Yes":
            bank_data[bank][month]['coverage'] += traded_tnd

    formatted_data = []
    for bank, months in bank_data.items():
        for month, stats in months.items():
            traded = stats['traded']
            coverage_percent = (stats['coverage'] / traded * 100) if traded > 0 else 0
            # Weighted gain percentage calculated in TND
            gain_percentage = (stats['gain'] / traded * 100) if traded > 0 else 0
            formatted_data.append({
                "bank": bank,
                "month": month,
                "total_traded": round(traded, 2),
                "coverage_percent": round(coverage_percent, 2),
                "gain": round(stats['gain'], 2),
                "gain_percentage": round(gain_percentage, 2)
            })

    return jsonify(formatted_data)



def calculate_hedge_status(transaction_date, value_date):
    return "Yes" if (value_date - transaction_date).days > 2 else "No"


# def calculate_benchmark(order):

#     # Fetch exchange data for the order's transaction date
#     try:
#         exchange_data_df = pd.read_sql(
#             'SELECT * FROM exchange_data WHERE "Date" = %(transaction_date)s',
#             db.engine,
#             params={"transaction_date": order.transaction_date.strftime("%Y-%m-%d")}
#         )
#     except Exception as e:
#         raise ValueError(f"Failed to fetch exchange data: {e}")

#     # Ensure exchange data is available
#     if exchange_data_df.empty:
#         raise ValueError(f"No exchange data available for date {order.transaction_date}")

#     # Extract the spot rate for the order's currency
#     try:
#         spot_rate = exchange_data_df[f'Spot {order.currency.upper()}'].values[0]
#     except KeyError:
#         raise ValueError(f"Spot rate for {order.currency} not found in exchange data for {order.transaction_date}")

#     # Validate historical loss
#     historical_loss = getattr(order, 'historical_loss', None)
#     if historical_loss is None:
#         raise ValueError("Historical loss is missing for the order, cannot calculate benchmark.")

#     # Adjust spot rate based on transaction type
#     if order.transaction_type.lower() in ["import", "buy"]:
#         loss_factor = 1 + historical_loss
#     elif order.transaction_type.lower() in ["export", "sell"]:
#         loss_factor = 1 - historical_loss
#     else:
#         raise ValueError(f"Unsupported transaction type: {order.transaction_type}")

#     base_benchmark = spot_rate * loss_factor

#     # Calculate hedge status and days difference
#     days_diff = (order.value_date - order.transaction_date).days
#     hedge_status = "Yes" if days_diff > 2 else "No"

#     # If hedged, apply forward rate adjustment
#     if hedge_status == "Yes":
#         yield_period = get_yield_period(days_diff)  # Determine yield period (1m, 3m, or 6m)
#         try:
#             yield_foreign = exchange_data_df[f'{yield_period.upper()} {order.currency.upper()}'].values[0]
#             yield_domestic = exchange_data_df[f'{yield_period.upper()} TND'].values[0]
#         except KeyError as e:
#             raise ValueError(f"Missing yield data for {yield_period} and currency {order.currency}: {e}")

#         # Calculate forward rate factor
#         forward_rate_factor = calculate_forward_rate(base_benchmark, yield_foreign, yield_domestic, days_diff)
#         return  forward_rate_factor

#     # Return the base benchmark for non-hedged transactions
#     return base_benchmark

def calculate_benchmark(order):
    """
    Calcule le benchmark (taux de référence) pour une commande.
    Si la commande est hedgée (écart > 2 jours entre transaction_date et value_date),
    on applique une correction via le taux forward.
    """
    try:
        exchange_data_df = pd.read_sql(
            'SELECT * FROM exchange_data WHERE "Date" = %(transaction_date)s',
            db.engine,
            params={"transaction_date": order.transaction_date.strftime("%Y-%m-%d")}
        )
    except Exception as e:
        raise ValueError(f"Échec de récupération des données de change : {e}")

    if exchange_data_df.empty:
        raise ValueError(f"Aucune donnée de change disponible pour la date {order.transaction_date}")

    # Utiliser exactement les clés définies dans le modèle
    if order.currency.upper() == "USD":
        spot_rate = exchange_data_df["Spot USD"].values[0]
    elif order.currency.upper() == "EUR":
        spot_rate = exchange_data_df["Spot EUR"].values[0]
    else:
        raise ValueError(f"Devise non supportée : {order.currency}")

    # Vérifier que historical_loss est défini
    historical_loss = getattr(order, 'historical_loss', None)
    if historical_loss is None:
        raise ValueError("Historical loss manquant pour la commande, impossible de calculer le benchmark.")

    # Ajuster le taux au comptant selon le type de transaction
    transaction_type = order.transaction_type.lower()
    if transaction_type in ["import", "buy"]:
        loss_factor = 1 + historical_loss
    elif transaction_type in ["export", "sell"]:
        loss_factor = 1 - historical_loss
    else:
        raise ValueError(f"Type de transaction non supporté : {order.transaction_type}")

    base_benchmark = spot_rate * loss_factor

    # Calculer la différence de jours entre transaction_date et value_date
    days_diff = (order.value_date - order.transaction_date).days
    hedge_status = "Yes" if days_diff > 2 else "No"

    if hedge_status == "Yes":
        # Obtenir la période de yield sous la forme "1m", "3m" ou "6m"
        yield_period = get_yield_period(days_diff)  # Doit renvoyer "1m", "3m" ou "6m"
        # Convertir en format majuscule pour correspondre aux clés (ex. "1M")
        period_mapping = {"1m": "1M", "3m": "3M", "6m": "6M"}
        period_key = period_mapping[yield_period]

        # Récupérer le taux étranger selon la devise
        if order.currency.upper() == "USD":
            yield_foreign = exchange_data_df[f"{period_key} USD"].values[0]
        elif order.currency.upper() == "EUR":
            yield_foreign = exchange_data_df[f"{period_key} EUR"].values[0]
        else:
            raise ValueError(f"Devise non supportée pour le yield : {order.currency}")

        # Récupérer le taux domestique (TND)
        yield_domestic = exchange_data_df[f"{period_key} TND"].values[0]

        # Calculer et renvoyer le taux forward ajusté
        forward_rate_factor = calculate_forward_rate(base_benchmark, yield_foreign, yield_domestic, days_diff)
        return forward_rate_factor

    return base_benchmark


from flask import current_app

# Helper: Get interbank rate from DB using the order’s transaction_date.
def get_interbank_rate_from_db(date, currency):
    # First try an exact match
    rate_entry = InterbankRate.query.filter_by(date=date, currency=currency).first()
    if rate_entry:
        return rate_entry.rate
    # Otherwise, return the latest available rate before the given date.
    rate_entry = InterbankRate.query.filter(
        InterbankRate.date < date,
        InterbankRate.currency == currency
    ).order_by(InterbankRate.date.desc()).first()
    if rate_entry:
        return rate_entry.rate
    return None

# Helper: Fetch interbank rate from external source (e.g., BCT website)
def fetch_rate_for_date_and_currency(date, currency):
    formatted_date = date.strftime('%Y-%m-%d')
    response = requests.post(f"https://www.bct.gov.tn/bct/siteprod/cours_archiv.jsp?input={formatted_date}&langue=en")
    soup = BeautifulSoup(response.content, 'html.parser')
    rate = None
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if cells and cells[1].get_text(strip=True).lower() == currency.lower():
            rate = float(cells[3].get_text(strip=True).replace(',', '.'))
            break
    return rate

def update_interbank_rates_db_logic(start_date_str="2024-08-01"):
    from datetime import datetime, timedelta
    # Parse the start date
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid start_date format, expected YYYY-MM-DD")
        return {"error": "Invalid start_date format, expected YYYY-MM-DD"}

    end_date = datetime.today().date()
    currencies = ["USD", "EUR"]  # Extend as needed.
    updated_entries = []

    for n in range((end_date - start_date).days + 1):
        current_date = start_date + timedelta(days=n)
        for currency in currencies:
            if not InterbankRate.query.filter_by(date=current_date, currency=currency).first():
                rate = fetch_rate_for_date_and_currency(current_date, currency)
                if rate:
                    new_rate = InterbankRate(date=current_date, currency=currency, rate=rate)
                    db.session.add(new_rate)
                    updated_entries.append({
                        "date": current_date.isoformat(),
                        "currency": currency,
                        "rate": rate
                    })
    try:
        db.session.commit()
        print("Interbank rates DB updated successfully", updated_entries)
        return {"message": "Interbank rates DB updated successfully", "updated": updated_entries}
    except Exception as e:
        db.session.rollback()
        print("Error updating interbank rates DB:", e)
        return {"error": str(e)}

@user_bp.route('/update-interbank-rates-db', methods=['POST'])
def update_interbank_rates_db_endpoint():
    data = request.get_json() or {}
    start_date_str = data.get("start_date", "2024-08-01")
    result = update_interbank_rates_db_logic(start_date_str)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 200


# Update orders: Use transaction_date to look up the interbank rate and update the order.
@user_bp.route('/update-interbank-rates', methods=['POST'])
def update_interbank_rates():
    try:
        update_order_interbank_and_benchmark_rates(current_app)
        return jsonify({'message': 'Orders updated with interbank & benchmark rates successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_order_interbank_and_benchmark_rates(app):
    with app.app_context():
        try:
            orders = Order.query.all()
            for order in orders:
                if order.interbank_rate is None:
                    rate = get_interbank_rate_from_db(order.transaction_date, order.currency)
                    if rate:
                        order.interbank_rate = float(rate)

                # Ensure conversion from np.float64 to Python float
                try:
                    benchmark = calculate_benchmark(order)
                    order.benchmark_rate = float(benchmark)
                except Exception as ex:
                    print(f"Error calculating benchmark for Order ID {order.id}: {ex}")

                db.session.add(order)

            db.session.commit()
            print("Orders successfully updated")

        except Exception as e:
            db.session.rollback()
            print(f"Error updating orders: {e}")


# import eikon as ek
# @user_bp.route("/test_eikon")
# def test_eikon():
#     try:
#         headlines = ek.get_news_headlines("TOP", count=5)
#         return {"status": "success", "data": headlines.to_json()}, 200
#     except Exception as e:
#         return {"status": "error", "message": str(e)}, 500


def calculate_time_to_maturity(trade_date, echeance):
    """
    Calculates time to maturity (in years) given the trade date and the value (or maturity) date.
    Both dates should be strings in the format '%d/%m/%Y'.
    """
    trade_date = pd.to_datetime(trade_date, format='%d/%m/%Y')
    echeance = pd.to_datetime(echeance, format='%d/%m/%Y')
    return (echeance - trade_date).days / 365

def interpolate_prime(time_to_maturity, known_times, known_primes):
    """
    Interpolates the premium percentage based on the target time_to_maturity.
    
    Parameters:
      - time_to_maturity: The calculated time (in years) for the option.
      - known_times: A NumPy array of times (in years) from your PremiumRate model.
      - known_primes: A NumPy array of premium percentages corresponding to the known times.
    
    Returns:
      A dictionary with values computed using different interpolation methods.
    """
    # Ensure known times and primes are sorted in increasing order
    sorted_indices = np.argsort(known_times)
    known_times = known_times[sorted_indices]
    known_primes = known_primes[sorted_indices]

    linear_interp = interp1d(known_times, known_primes, kind='linear', fill_value='extrapolate')
    quadratic_interp = interp1d(known_times, known_primes, kind='quadratic', fill_value='extrapolate')
    cubic_spline = CubicSpline(known_times, known_primes, extrapolate=True)
    
    return {
        'Linear': linear_interp(time_to_maturity),
        'Quadratic': quadratic_interp(time_to_maturity),
        'CubicSpline': cubic_spline(time_to_maturity),
    }
@user_bp.route('/orders/preview', methods=['POST'])
@jwt_required()
def preview_option():
    """
    Preview an option order by computing the forward rate, premium, moneyness, and default strike.
    This endpoint does NOT create an order.
    """
    data = request.get_json()
    if not data:
        return jsonify({"message": "No data provided"}), 400

    # Extract fields from request payload
    try:
        amount = float(data.get("amount", 0))
        transaction_type = data.get("transaction_type")
        value_date_str = data.get("value_date")
        currency = data.get("currency")
        bank_account = data.get("bank_account")
        is_option = data.get("is_option", False)
        option_type = data.get("option_type", "").upper()
        strike = data.get("strike")
    except Exception as e:
        return jsonify({"message": f"Invalid data format: {e}"}), 400

    if not (transaction_type and value_date_str and currency and bank_account):
        return jsonify({"message": "Missing required fields"}), 400

    # Parse the value date
    try:
        value_date = datetime.strptime(value_date_str, "%Y-%m-%d")
    except Exception as e:
        return jsonify({"message": "Invalid date format"}), 400

    if not is_option:
        return jsonify({"message": "Preview is only available for options"}), 400

    # Retrieve exchange data (using today's date as an example)
    today = datetime.today().date()
    exchange_data = ExchangeData.query.filter_by(date=today).first()
    if not exchange_data:
        return jsonify({"message": "Exchange data for today not available"}), 400

    # Calculate days difference
    days_diff = (value_date.date() - today).days
    period = get_yield_period(days_diff)
    if currency.upper() == 'USD':
        spot_rate = exchange_data.spot_usd
        yield_foreign = getattr(exchange_data, f"usd_{period[0]}m")
    elif currency.upper() == 'EUR':
        spot_rate = exchange_data.spot_eur
        yield_foreign = getattr(exchange_data, f"eur_{period[0]}m")
    else:
        return jsonify({"message": "Unsupported currency"}), 400

    yield_domestic = getattr(exchange_data, f"tnd_{period[0]}m")

    # Compute the forward rate using your helper function
    computed_forward = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)

    # Determine moneyness based on strike if provided; if not, use computed_forward as strike
    tol = 0.01 * computed_forward
    if strike is not None:
        try:
            strike_value = float(strike)
        except Exception:
            return jsonify({"message": "Strike must be numeric"}), 400
    else:
        strike_value = computed_forward

    if option_type == "CALL":
        if computed_forward > strike_value + tol:
            moneyness = "in the money"
        elif abs(computed_forward - strike_value) <= tol:
            moneyness = "at the money"
        else:
            moneyness = "out of the money"
    elif option_type == "PUT":
        if computed_forward < strike_value - tol:
            moneyness = "in the money"
        elif abs(computed_forward - strike_value) <= tol:
            moneyness = "at the money"
        else:
            moneyness = "out of the money"
    else:
        return jsonify({"message": "Option type must be CALL or PUT"}), 400

    # Compute premium using your interpolation logic.
    premium_rates = PremiumRate.query.filter_by(currency=currency.upper(), option_type=option_type, transaction_type=transaction_type  # NEW: filter for "buy" or "sell"
).all()
    if not premium_rates:
        return jsonify({"message": f"No premium rates found for {currency} {option_type}"}), 400

    ttm = calculate_time_to_maturity(datetime.now().strftime('%d/%m/%Y'), value_date.strftime('%d/%m/%Y'))
    known_times = np.array([r.maturity_days / 365.0 for r in premium_rates])
    known_primes = np.array([r.premium_percentage for r in premium_rates])
    interpolated = interpolate_prime(ttm, known_times, known_primes)
    chosen_rate = float(interpolated['Linear'])
    computed_premium = amount * chosen_rate

    # Return the computed preview values as JSON
    return jsonify({
        "forward_rate": computed_forward,
        "premium": computed_premium,
        "moneyness": moneyness,

        "strike": strike_value,  # Will equal computed_forward if no strike was provided
    }), 200
@user_bp.route('/get-interbank-rates', methods=['GET'])
def get_interbank_rates():
    try:
        rates = InterbankRate.query.all()
        result = [{
            "date": rate.date.strftime("%Y-%m-%d"),
            "currency": rate.currency,
            "rate": rate.rate
        } for rate in rates]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
