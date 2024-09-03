from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.schema import CreateTable
import io

db = SQLAlchemy()


class BankAccount(db.Model):
    id = db.Column(db.String, primary_key=True)
    bank_name = db.Column(db.String(80), nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    owner = db.Column(db.String(120), nullable=False)
    balance = db.Column(db.Float, nullable=False)
    account_number = db.Column(db.String(20), unique=True, nullable=False)
    branch = db.Column(db.String(100))
    category = db.Column(db.String(50))
    date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), nullable=False)


class OpenPosition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value_date = db.Column(db.Date, nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    fx_amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    user = db.Column(db.String(120), nullable=False)


class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    id_unique = db.Column(db.String(80), nullable=False)
    transaction_type = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    value_date = db.Column(db.Date, nullable=False)
    order_date = db.Column(db.Date, nullable=False)
    bank_account = db.Column(db.String(100), nullable=False)
    reference = db.Column(db.String(100))
    signing_key = db.Column(db.String(255))
    user = db.Column(db.String(120), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Integer)


class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100), nullable=False)
    representative_name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    notes = db.Column(db.Text)


class Historical(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    usd = db.Column(db.Float, nullable=False)
    eur = db.Column(db.Float, nullable=False)
    gbp = db.Column(db.Float, nullable=False)
    jpy = db.Column(db.Float, nullable=False)


class MatchedPosition(db.Model):
    id = db.Column(db.String, primary_key=True)
    value_date = db.Column(db.Date, nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    buyer = db.Column(db.String(100), nullable=False)
    buyer_rate = db.Column(db.Integer, nullable=False)
    seller = db.Column(db.String(100), nullable=False)
    seller_rate = db.Column(db.Integer, nullable=False)
    matched_amount = db.Column(db.Float, nullable=False)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    rating = db.Column(db.Float, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

