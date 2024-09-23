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
from .services.order_service import generate_unique_key, scheduled_matching
from .services.export_service import export_pdf, download_excel
from .services.meeting_service import get_meetings_for_month, generate_month_days
from models import db, Order, MatchedPosition, Meeting
from flask_login import login_user, logout_user, current_user
from user.services.user_service import UserService
from functools import wraps
from models import User, Role

admin_bp = Blueprint('admin_bp', __name__, template_folder='templates', static_folder='static')

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

"""def generate_unique_key(buyer, seller):
    # Create a unique key based on the first 2 letters of buyer and seller names and 8 random digits
    random_digits = ''.join(random.choices(string.digits, k=8))
    return buyer[:1] + seller[:1] + random_digits"""


# signup page
@admin_bp.route('/signup', methods=['GET', 'POST'])
def sign():
    msg=""
    # if the form is submitted
    if request.method == 'POST':
    # check if user already exists
        user = User.query.filter_by(email=request.form['email']).first()
        msg=""
        # if user already exists render the msg
        if user:
            msg="User already exist"
            # render signup.html if user exists
            return render_template('signup.html', msg=msg)
        
        # if user doesn't exist
        
        # store the user to database
        user = User(email=request.form['email'], active=1, password=request.form['password'])
        # store the role
        role = Role.query.filter_by(id=int(request.form['options'])).first()
        if not role:
            msg = "Role not found."

        user.roles.append(role)

        # commit the changes to database
        db.session.add(user)
        db.session.commit()
        
        # login the user to the app
        # this user is current user
        login_user(user)
        # redirect to index page
        return redirect(url_for('index'))
        
    # case other than submitting form, like loading the page itself
    else:
        return render_template("page-signup.html", msg=msg)

    # signin page
@admin_bp.route('/signin', methods=['GET', 'POST'])
def signin():
    msg=""
    if request.method == 'POST':
        # search user in database
        user = User.query.filter_by(email=request.form['email']).first()
        # if exist check password
        if user:
            if  user.password == request.form['password']:
                # if password matches, login the user
                login_user(user)
                return redirect(url_for('index'))
            # if password doesn't match
            else:
                msg="Wrong password"
        
        # if user does not exist
        else:
            msg="User doesn't exist"
        return render_template('page-signin.html', msg=msg)
        
    else:
        return render_template("page-signin.html", msg=msg)

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





