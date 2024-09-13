from app import db
from models.models import Historical

def get_historical_rates(start_date=None, end_date=None, currency=None):
    query = Historical.query
    if start_date:
        query = query.filter(Historical.date >= start_date)
    if end_date:
        query = query.filter(Historical.date <= end_date)
    if currency:
        query = query.filter(getattr(Historical, currency) != None)

    results = query.all()

    if currency:
        data = [{
            'date': record.date.strftime('%Y-%m-%d'),
            currency: getattr(record, currency)
        } for record in results]
    else:
        data = [{
            'id': record.id,
            'date': record.date.strftime('%Y-%m-%d'),
            'usd': record.usd,
            'eur': record.eur,
            'gbp': record.gbp,
            'jpy': record.jpy
        } for record in results]

    return data


def create_historical_record(data):
    new_record = Historical(
        date=data['date'],
        usd=data['usd'],
        eur=data['eur'],
        gbp=data['gbp'],
        jpy=data['jpy']
    )
    db.session.add(new_record)
    db.session.commit()
    return {'message': 'Historical record created'}
