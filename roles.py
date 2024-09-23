# create_roles.py
from app import Role, User, db

def create_roles():
    admin = Role(id=1, name='Admin')
    client = Role(id=2, name='Client')
    db.session.add(admin)
    db.session.add(client)
    db.session.commit()
    print("Roles created successfully!")

create_roles()