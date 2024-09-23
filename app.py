from flask import Flask
from models import db , User, Role
from admin.routes import admin_bp, register_admin_jobs
from user.routes import user_bp, register_user_jobs, init_socketio, register_live_rates
from scheduler import scheduler, start_scheduler
from flask_security import Security, SQLAlchemySessionUserDatastore


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@db:5432/postgres'
# needed for session cookies
app.config['SECRET_KEY'] = 'MY_SECRET'
# allows new registrations to application
app.config['SECURITY_REGISTERABLE'] = True
# to send automatic registration email to user
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False
db.init_app(app)
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(user_bp, url_prefix='/')

# load users, roles for a session
user_datastore = SQLAlchemySessionUserDatastore(db.session, User, Role)
security = Security(app, user_datastore)

register_admin_jobs(scheduler, app)
register_user_jobs(scheduler, app)
register_live_rates(scheduler, app)
start_scheduler()
socketio = init_socketio(app)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
    socketio.run(app, debug=True)
