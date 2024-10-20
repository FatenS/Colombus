# create_roles.py
from app import app, db  # Import the Flask app and database instance from app.py
from models import Role  # Import the Role model (make sure it's defined in models.py)

def create_roles():
    admin = Role(id=1, name='Admin')
    client = Role(id=2, name='Client')
    
    # Add roles to the database
    db.session.add(admin)
    db.session.add(client)
    db.session.commit()
    
    print("Roles created successfully!")

if __name__ == "__main__":
    # Ensure the Flask application context is pushed
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
        create_roles()  # Call the function to create roles
