from flask import Blueprint
accounts_bp = Blueprint('accounts_bp', __name__)   # single source of truth

from . import routes     