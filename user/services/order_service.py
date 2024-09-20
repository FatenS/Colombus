from models import Order, BankAccount, User
from db import db
from datetime import datetime
import pandas as pd
import calendar

class OrderService:

    @staticmethod
    def get_order_details(session_user):
        bank_accounts = BankAccount.query.filter_by(owner=session_user).all()
        orders = Order.query.filter_by(user=session_user).order_by(Order.status.desc()).all()

        # Prepare data for the UI
        bank_accounts_data = [{
            'id': account.id,
            'bank_name': account.bank_name,
            'currency': account.currency,
            'balance': account.balance,
            'account_number': account.account_number,
            'branch': account.branch,
            'category': account.category,
            'date': account.date.strftime('%Y-%m-%d') if account.date else None,
            'status': account.status,
        } for account in bank_accounts]

        orders_list = [{
            'id': order.reference,
            'transaction_type': order.transaction_type,
            'amount': order.amount,
            'currency': order.currency,
            'value_date': order.value_date.strftime('%Y-%m-%d') if order.value_date else None,
            'order_date': order.order_date.strftime('%Y-%m-%d') if order.order_date else None,
            'bank_account': order.bank_account,
            'reference': order.reference,
            'user': order.user,
            'status': order.status,
            'rating': order.rating,
        } for order in orders]

        # Aggregate monthly sums for charting
        monthly_sums = db.session.query(
            db.func.date_trunc('month', Order.value_date).label('month'),
            db.func.sum(Order.amount).label('total_amount')
        ).group_by('month').order_by('month').all()

        chart_data = [{'year_month': month.strftime("%Y-%m"), 'Transaction Amount': total_amount}
                      for month, total_amount in monthly_sums]

        return orders_list, chart_data, bank_accounts_data

    @staticmethod
    def submit_order(data, session_user):
        current_user = User.query.filter_by(username=session_user).first()
        user_rating = current_user.rating if current_user and current_user.rating else 0

        new_order = Order(
            user=session_user,
            status="Pending",
            rating=user_rating,
            transaction_type=data['transaction_type'],
            amount=data['amount'],
            currency=data['currency'],
            value_date=data['value_date'],
            order_date=datetime.today(),
            bank_account=data['bank_account'],
            reference=data['reference']
        )
        db.session.add(new_order)
        db.session.commit()

        return {'message': 'Order executed successfully'}

    @staticmethod
    def delete_order(orderReference):
        order = Order.query.filter_by(reference=orderReference).first()
        if order:
            db.session.delete(order)
            db.session.commit()
            return {'message': 'Order deleted successfully'}
        return {'error': 'Order not found'}

    @staticmethod
    def data_for_chart(session_user):
        orders = Order.query.filter(Order.user == session_user).with_entities(Order.value_date, Order.amount).all()

        df = pd.DataFrame(orders, columns=['value_date', 'amount'])
        df['month'] = pd.to_datetime(df['value_date']).dt.month
        df['month_abbr'] = df['month'].apply(lambda x: calendar.month_abbr[x])
        monthly_totals = df.groupby('month_abbr')['amount'].sum().reset_index()

        data = {
            'months': monthly_totals['month_abbr'].tolist(),
            'amounts': monthly_totals['amount'].tolist()
        }
        return data
