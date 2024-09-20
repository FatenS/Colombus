from werkzeug.security import generate_password_hash, check_password_hash
from models import User
from db import db

class UserService:

    @staticmethod
    def get_user_by_username_or_email(username, email):
        return User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first()

    @staticmethod
    def create_user(username, email, password, rating, role="user"):
        new_user = User(username=username, email=email, rating=rating, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return new_user

    @staticmethod
    def check_user_password(user, password):
        return user and user.check_password(password)

    @staticmethod
    def is_admin(user):
        return user.role == "admin"
