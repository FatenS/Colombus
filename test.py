import eikon as ek

ek.set_app_key("20af0572a6364fe8abf9a35cdd16bd367057564a")
ek.set_port_number(9000)  # Default proxy port for Eikon Desktop
headlines= ek.get_news_headlines('R:LHAG.DE', date_from='2024-03-15T09:00:00', date_to='2024-03-15T18:00:00')  
print(headlines)
