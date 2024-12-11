from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped
from sqlalchemy import String, Enum, CheckConstraint
import uuid
from flask_security import UserMixin, RoleMixin
from datetime import datetime


db = SQLAlchemy()

# create table in database for assigning roles
roles_users = db.Table('roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

# create table in database for storing users
class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    email = db.Column(db.String, nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean(), nullable=True)
    rating = db.Column(db.Integer, nullable=False, default=0)  
    roles = db.relationship('Role', secondary=roles_users, backref='roled')
    fs_uniquifier: Mapped[str] = db.Column(String(64), unique=True, nullable=True, default=lambda: str(uuid.uuid4()))

# create table in database for storing roles
class Role(db.Model, RoleMixin):
    __tablename__ = 'role'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)

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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    user = db.relationship('User', backref='bank_accounts', lazy=True)

class OpenPosition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value_date = db.Column(db.Date, nullable=False, index=True)
    currency = db.Column(db.String(3), nullable=False, index=True)
    fx_amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    user = db.Column(db.String(120), nullable=False)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    id_unique = db.Column(db.String(80), nullable=False, default=lambda: str(uuid.uuid4()))
    transaction_type = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)  # Remaining amount for unmatched or partially matched orders
    original_amount = db.Column(db.Float, nullable=False)  # Initial order amount
    currency = db.Column(db.String(3), nullable=False, index=True)
    value_date = db.Column(db.Date, nullable=False, index=True)
    transaction_date = db.Column(db.Date, nullable=False)
    order_date = db.Column(db.Date, nullable=False)
    bank_account = db.Column(db.String(100))
    reference = db.Column(db.String(100))
    signing_key = db.Column(db.String(255))
    status = db.Column(Enum('Pending', 'Executed', 'Market','Cancelled', 'Matched', name='order_status'), nullable=False, default='Pending')
    rating = db.Column(db.Integer)
    deleted = db.Column(db.Boolean, default=False, nullable=False)
    historical_loss = db.Column(db.Float, nullable=True, default=0.02)
    interbank_rate = db.Column(db.Float, nullable=True, default=None)
    execution_rate = db.Column(db.Float, nullable=True, default=None)
    bank_name = db.Column(db.String(100), nullable=True, default=None)
    matched_order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=True)  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    matched_amount = db.Column(db.Float, nullable=True, default=0.0)  
    # Relationships
    user = db.relationship('User', backref='orders', lazy=True)
    matched_order = db.relationship('Order', remote_side=[id], backref='related_orders')

    # Adding a check constraint for the 'amount'
    __table_args__ = (
        CheckConstraint('amount >= 0', name='check_amount_positive'),
    )

class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100), nullable=False)
    representative_name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    notes = db.Column(db.Text)

class ExchangeData(db.Model):
    __tablename__ = 'exchange_data'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, name='Date')
    spot_usd = db.Column(db.Float, nullable=False, name='Spot USD')
    spot_eur = db.Column(db.Float, nullable=False, name='Spot EUR')
    tnd_1m = db.Column(db.Float, nullable=False, name='1M TND')
    eur_1m = db.Column(db.Float, nullable=False, name='1M EUR')
    usd_1m = db.Column(db.Float, nullable=False, name='1M USD')
    tnd_3m = db.Column(db.Float, nullable=False, name='3M TND')
    eur_3m = db.Column(db.Float, nullable=False, name='3M EUR')
    usd_3m = db.Column(db.Float, nullable=False, name='3M USD')
    tnd_6m = db.Column(db.Float, nullable=False, name='6M TND')
    eur_6m = db.Column(db.Float, nullable=False, name='6M EUR')
    usd_6m = db.Column(db.Float, nullable=False, name='6M USD')


class AuditLog(db.Model):
    __tablename__ = 'audit_log'
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(50), nullable=False) 
    table_name = db.Column(db.String(50), nullable=False)  
    record_id = db.Column(db.String(80), nullable=False)  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)
    details = db.Column(db.Text)  # JSON string 

    user = db.relationship('User', backref='audit_logs')