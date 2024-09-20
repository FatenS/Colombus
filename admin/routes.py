import base64
import io
import os
import json
import random
import string
import numpy as np
import pandas as pd
from fpdf import FPDF
from flask import Blueprint, render_template, request, redirect, url_for, send_file, make_response, jsonify, flash, session
from datetime import datetime, timedelta
from sqlalchemy import func
from admin.services.order_service import generate_unique_key, scheduled_matching
from admin.services.export_service import export_pdf, download_excel
from admin.services.meeting_service import get_meetings_for_month, generate_month_days
from models import db, Order, MatchedPosition, Meeting
from flask_login import login_user, logout_user, current_user
from user.services.user_service import UserService
from functools import wraps

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

# Role-based access decorator for admins
def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash("Access denied. Admins only.", "error")
            return redirect(url_for('user_bp.login'))
        return func(*args, **kwargs)
    return wrapper


# Admin sign-in page
#@admin_required
@admin_bp.route('/page-signin')
def main():
    return render_template('page-signin.html')

# Admin sign-up page
@admin_bp.route('/page-signup')
#@admin_required
def sign():
    return render_template('page-signup.html')

# Admin logout
@admin_bp.route('/out')
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for('admin_bp.main'))
"""
# Admin sign-in route
@admin_bp.route('/signin', methods=['POST'])
def signin():
    username = request.form.get('username')
    password = request.form.get('password')

    admin = UserService.get_user_by_username_or_email(username, None)

    if admin and UserService.check_user_password(admin, password) and admin.is_admin():
        session['username'] = admin.username
        login_user(admin)
        flash("Welcome to the admin room", "success")
        return redirect(url_for('admin_bp.index'))
    else:
        flash("Wrong credentials or not an admin", "error")
        return redirect(url_for('admin_bp.main'))

# Admin sign-up users (only accessible to admins)
@admin_bp.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    rating = request.form.get('rating')
    role = request.form.get('role')  # user or admin

    existing_user = UserService.get_user_by_username_or_email(username, email)
    if existing_user:
        flash("Username already exists", "error")
        return redirect(url_for('admin_bp.sign'))

    UserService.create_user(username, email, password, rating, role)
    flash("Account created successfully", "success")
    return redirect(url_for('admin_bp.sign')) """

@admin_bp.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    admin = UserService.get_user_by_username_or_email(username, None)

    if admin and UserService.check_user_password(admin, password) and admin.is_admin():
        session['username'] = admin.username
        login_user(admin)
        return jsonify({"message": "Welcome to the admin room"}), 200
    else:
        return jsonify({"error": "Wrong credentials or not an admin"}), 401

@admin_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    rating = data.get('rating')
    role = data.get('role')  # user or admin

    existing_user = UserService.get_user_by_username_or_email(username, email)
    if existing_user:
        return jsonify({"error": "Username already exists"}), 409

    UserService.create_user(username, email, password, rating, role)
    return jsonify({"message": "Account created successfully"}), 201


# Admin dashboard route
@admin_bp.route('/index')
def index():
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
    matches, metrics, market, labels, buy_data, sell_data, transaction_sums_dict = scheduled_matching(currency=currency)
    Area_data = {
        'labels': labels,
        'datasets': [
            {'data': buy_data, 'label': 'Buy', 'backgroundColor': 'rgba(212, 215, 222,0.5)', 'borderColor': '#d4d7de'},
            {'data': sell_data, 'label': 'Sell', 'backgroundColor': 'rgba(7, 28, 66,0.5)', 'borderColor': '#071C42'}
        ]
    }

    metrics_json = json.dumps(metrics, cls=JSONEncoder)

    processed_data = {}
    for transaction in market:
        amount = transaction['Transaction Amount'] if transaction['Type'] == 'sell' else -transaction['Transaction Amount']
        date_str = transaction['Value Date']
        if date_str in processed_data:
            processed_data[date_str] += amount
        else:
            processed_data[date_str] = amount

    sorted_dates = sorted(processed_data.keys())
    sorted_amounts = [processed_data[date] for date in sorted_dates]
    chart_data = json.dumps({'labels': sorted_dates, 'data': sorted_amounts})

    sorted_market_data = sorted(market, key=lambda x: x['Rating'], reverse=True)
    top_5_clients = sorted_market_data[:5]
    formatted_data = [{
        'Name': client['Client'],
        'Transaction Amount': client['Transaction Amount'],
        'Currency': client['Currency'],
        'Value Date': client['Value Date'].strftime('%Y-%m-%d'),
        'Rating': client['Rating']
    } for client in top_5_clients]

    return render_template('index.html', new_orders_list=new_orders_list, metrics=metrics_json, chart_data=chart_data, Area_data=Area_data, formatted_data=formatted_data)

@admin_bp.route('/rates')
def rates():
    return render_template('rates.html')

@admin_bp.route('/meetings', methods=['GET'])
def meetings():
    year = int(request.args.get('year', datetime.now().year))
    month = int(request.args.get('month', datetime.now().month))
    meetings = get_meetings_for_month(year, month)
    month_days = generate_month_days(year, month)
    return render_template('meetings.html', meetings=meetings, month_days=month_days)

@admin_bp.route('/export', methods=['POST'])
def export():
    export_type = request.form.get('export_type')
    tables_data = {
        'Table1': [['Header1', 'Header2'], ['Row1', 'Data1'], ['Row2', 'Data2']],
        'Table2': [['Header1', 'Header2'], ['Row1', 'Data1'], ['Row2', 'Data2']]
    }

    if export_type == 'pdf':
        images = request.form.getlist('images')
        return export_pdf(images)
    elif export_type == 'excel':
        return download_excel(tables_data)
    else:
        return jsonify({'error': 'Invalid export type'}), 400

def register_admin_jobs(scheduler, app):
    scheduler.add_job(scheduled_matching, 'cron', hour=16, minute=14, args=[app])





