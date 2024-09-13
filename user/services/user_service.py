
from models.models import db,  User
from werkzeug.security import generate_password_hash, check_password_hash
from flask import redirect, url_for
from flask_login import login_user , session

class UserService:
    
    @staticmethod
    def signup_user(username, email, password, rating):
        existing_user = User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first()
        if existing_user:
            return redirect(url_for('admin_bp.sign', error='Username already exists'))

        new_user = User(username=username, email=email, rating=rating)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('admin_bp.sign', message='Account created successfully'))
    
    @staticmethod
    def login_user(username, password):
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['username'] = user.username
            login_user(user)
            return redirect(url_for('user_bp.dashboard', message='Logged in successfully'))
        else:
            return redirect(url_for('user_bp.home', message='Wrong credentials'))
