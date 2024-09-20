import calendar
from datetime import datetime
from models import Meeting

def get_meetings_for_month(year, month):
    meetings = Meeting.query.filter(
        Meeting.start_time >= datetime(year, month, 1),
        Meeting.end_time < datetime(year, month + 1, 1)
    ).all()
    return meetings

def generate_month_days(year, month):
    days_in_month = calendar.monthrange(year, month)[1]
    return [datetime(year, month, day) for day in range(1, days_in_month + 1)]
