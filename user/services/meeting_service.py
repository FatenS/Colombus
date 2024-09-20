
from models import Meeting
from datetime import datetime
from db import db


class MeetingService:

    @staticmethod
    def book_meeting(form_data):
        company_name = form_data.get('name')
        representative_name = form_data.get('rep')
        representative_position = form_data.get('position')
        email = form_data.get('email')
        date_str = form_data.get('date')
        time_str = form_data.get('time')
        notes = form_data.get('notes', '')

        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        time = datetime.strptime(time_str, '%H:%M').time()

        new_meeting = Meeting(
            company_name=company_name,
            representative_name=representative_name,
            position=representative_position,
            email=email,
            date=date,
            time=time,
            notes=notes
        )

        db.session.add(new_meeting)
        db.session.commit()
        return 'Meeting booked successfully'
