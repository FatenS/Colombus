from flask import Flask
from models import db
from admin.routes import admin_bp, register_admin_jobs
from user.routes import user_bp, register_user_jobs, init_login_manager, init_socketio, register_live_rates
from scheduler import scheduler, start_scheduler

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@db:5432/postgres'
db.init_app(app)
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(user_bp, url_prefix='/')
register_admin_jobs(scheduler, app)
register_user_jobs(scheduler, app)
register_live_rates(scheduler, app)
start_scheduler()
init_login_manager(app)
socketio = init_socketio(app)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
    socketio.run(app, debug=True)
