from flask import session, jsonify, redirect, url_for, render_template
from db import db
from models import BankAccount
from datetime import datetime
from flask import jsonify
from datetime import datetime
from sqlalchemy import func

class BankService:

    @staticmethod
    def get_bank_data(username):
        accounts = BankAccount.query.filter_by(owner=username).all()

        total_balances = db.session.query(
            BankAccount.currency,
            func.sum(BankAccount.balance).label('total')
        ).filter(BankAccount.owner == username).group_by(BankAccount.currency).all()

        total_dollar_balance = sum(b.total for b in total_balances if b.currency == 'USD')
        total_euro_balance = sum(b.total for b in total_balances if b.currency == 'EUR')
        total_tnd_balance = sum(b.total for b in total_balances if b.currency == 'TND')
        number_of_accounts = len(accounts)

        data_for_chart = [{'Currency': b.currency, 'Balance': b.total} for b in total_balances]

        data = [{
            'Status': account.status,
            'Account ID': account.id,
            'Bank': account.bank_name,
            'Branch': account.branch,
            'Account Number': account.account_number,
            'Currency': account.currency,
            'Date': account.date.strftime('%Y-%m-%d'),
            'Category': account.category,
            'Balance': account.balance
        } for account in accounts]

        return {
            'data': data,
            'total_dollar_balance': total_dollar_balance,
            'total_euro_balance': total_euro_balance,
            'total_tnd_balance': total_tnd_balance,
            'number_of_accounts': number_of_accounts,
            'data_for_chart': data_for_chart
        }

    @staticmethod
    def add_new_account(form_data, username):
        bank = form_data.get('bank')
        currency = form_data.get('currency')
        balance = float(form_data.get('balance'))
        account_number = form_data.get('nbrAcc')
        branch = form_data.get('branch')
        category = form_data.get('category')
        date = datetime.strptime(form_data.get('date'), '%Y-%m-%d')
        status = "Open"

        # Generate a primary key
        pk = f"{bank[:2].upper()}{account_number}"

        new_account = BankAccount(
            id=pk,
            bank_name=bank,
            currency=currency,
            owner=username,
            balance=balance,
            account_number=account_number,
            branch=branch,
            category=category,
            date=date,
            status=status
        )
        db.session.add(new_account)
        db.session.commit()

    @staticmethod
    def get_account_details(account_id):
        account = BankAccount.query.filter_by(id=account_id).first()
        if account:
            account_details = {
                "id": account.id,
                "bank_name": account.bank_name,
                "currency": account.currency,
                "owner": account.owner,
                "balance": account.balance,
                "account_number": account.account_number,
                "branch": account.branch,
                "category": account.category,
                "date": account.date.strftime("%Y-%m-%d"),
                "status": account.status
            }
            return jsonify(account_details)
        else:
            return jsonify({"message": "Account not found"}), 404

    @staticmethod
    def update_bank_account(account_id, data):
        account = BankAccount.query.get_or_404(account_id)
        account.bank_name = data['bank_name']
        account.currency = data['currency']
        account.balance = data['balance']
        account.category = data.get('category')
        account.status = data['status']
        db.session.commit()
        return jsonify({'message': 'Bank account updated successfully'})

    @staticmethod
    def delete_bank_account(account_id):
        try:
            account = BankAccount.query.get_or_404(account_id)
            db.session.delete(account)
            db.session.commit()
            return jsonify({'message': 'Bank account deleted successfully'}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({'message': 'Error deleting account: ' + str(e)}), 500
