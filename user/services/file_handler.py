from io import BytesIO
import pandas as pd
from db import db
from models import OpenPosition
from utils import convert_to_date
from flask import session 


def process_uploaded_file(file_stream):
    file_stream.seek(0)
    bytes_io = BytesIO(file_stream.read())
    uploaded_data = pd.read_excel(bytes_io, header=None, skiprows=1)

    for index, row in uploaded_data.iterrows():
        new_position = OpenPosition(
            value_date=convert_to_date(row[0]),
            currency=row[1],
            fx_amount=row[2],
            type=row[3],
            user=session["username"]
        )
        db.session.add(new_position)
    
    db.session.commit()
