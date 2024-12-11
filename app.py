from flask import Flask
from flask_cors import CORS
from models import db, User, Role
from admin.routes import admin_bp, register_admin_jobs
from user.routes import user_bp, init_socketio
from scheduler import scheduler, start_scheduler
from flask_security import Security, SQLAlchemySessionUserDatastore
from flask_jwt_extended import JWTManager

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@db:5432/postgres'
app.config['SECRET_KEY'] = 'MY_SECRET'
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']


jwt = JWTManager(app)
db.init_app(app)

# Register blueprints
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(user_bp, url_prefix='/')

# Setup security and roles
user_datastore = SQLAlchemySessionUserDatastore(db.session, User, Role)
security = Security(app, user_datastore)

# Register background jobs and socket connections
register_admin_jobs(scheduler, app)
start_scheduler()
socketio = init_socketio(app)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
    socketio.run(app, debug=True)
