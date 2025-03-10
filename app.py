from flask import Flask
from flask_cors import CORS
from models import db, User, Role
from admin.routes import admin_bp 
from user.routes import user_bp, init_socketio, delete_expired_positions, update_order_interbank_rates
from scheduler import scheduler, start_scheduler
from scheduler_jobs import check_for_new_files
from flask_security import Security, SQLAlchemySessionUserDatastore
from flask_jwt_extended import JWTManager
import os

app = Flask(__name__)
CORS(app)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@localhost:5432/postgres'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@db:5432/postgres'
app.config['SECRET_KEY'] = 'MY_SECRET'
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']

# Initialize Flask extensions
jwt = JWTManager(app)
db.init_app(app)

# Register blueprints
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(user_bp, url_prefix='/')

# Setup security
user_datastore = SQLAlchemySessionUserDatastore(db.session, User, Role)
security = Security(app, user_datastore)

with app.app_context():
    db.create_all()

# --------------------------------------------------
# 1) Add the check_for_new_files job (5 min interval)
# --------------------------------------------------
scheduler.add_job(
    func=check_for_new_files, 
    trigger='interval', 
    minutes=5,
    id="check_for_new_files_job"
)

# ------------------------------------------------------------
# 2) Add daily job to delete expired positions (24h interval)
# ------------------------------------------------------------
scheduler.add_job(
    func=delete_expired_positions,
    trigger='interval',
    hours=24,
    kwargs={'app': app},
    id="delete_expired_positions_job"
)

# ------------------------------------------------------------
# 3) Add daily job to update interbank rates (24h interval)
# ------------------------------------------------------------
scheduler.add_job(
    func=update_order_interbank_rates,
    trigger='interval',
    hours=24,
    kwargs={'app': app},
    id="update_order_interbank_rates_job"
)

# Start the scheduler
start_scheduler()

socketio = init_socketio(app)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
    socketio.run(app, debug=True)
