from flask import Flask
from flask_cors import CORS
from models import db, User, Role
from admin.routes import admin_bp 
from user.routes import user_bp, init_socketio, start_scheduler
from scheduler import scheduler
from flask_security import Security, SQLAlchemySessionUserDatastore
from flask_jwt_extended import JWTManager
from apscheduler.schedulers.background import BackgroundScheduler
import os
import eikon as ek

# Initialize the Flask app
app = Flask(__name__)

# Initialize Eikon API
def init_eikon():
    try:
        ek.set_app_key(os.getenv("EIKON_APP_KEY"))  # Set App Key
        ek.set_port_number(9000)  # Ensure the port is set manually if needed
        print("Connected to Eikon Data API successfully.")
    except ek.EikonError as e:
        print("Eikon initialization error:", e)
# Call Eikon initialization during app startup
init_eikon()

# Enable CORS for all routes
CORS(app)

# App configuration
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

# Setup security and roles
user_datastore = SQLAlchemySessionUserDatastore(db.session, User, Role)
security = Security(app, user_datastore)

# Register background jobs and socket connections
scheduler = BackgroundScheduler()

# Register jobs
start_scheduler(scheduler, app)

# Start scheduler once
scheduler.start()
socketio = init_socketio(app)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
    socketio.run(app, debug=True)
