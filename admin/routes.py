import base64
import io
import os
import json
import numpy as np
import pandas as pd
from fpdf import FPDF
from flask import Blueprint, render_template, request, redirect, url_for, send_file, make_response, jsonify
from datetime import datetime, timedelta
from sqlalchemy import func
from services.order_service import generate_unique_key, scheduled_matching
from services.export_service import export_pdf, download_excel
from services.meeting_service import get_meetings_for_month, generate_month_days
from models.models import db, Order, MatchedPosition, Meeting

admin_bp = Blueprint('admin_bp', __name__, template_folder='templates', static_folder='static')

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

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

    if username == "engine-takwa" and password == "engine2511@":
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
