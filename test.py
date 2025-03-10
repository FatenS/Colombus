# import eikon as ek
# import pandas as pd
# import datetime
# import refinitiv.data as rd
# import warnings

# # Suppress warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# # Set Eikon app key and port number
# ek.set_app_key("20af0572a6364fe8abf9a35cdd16bd367057564a")
# ek.set_port_number(9000)  # Default proxy port for Eikon Desktop

# # Open a session (ensure Eikon or Workspace is running)
# rd.open_session()

# # Specify RICs and date range
# rics = ['TND=BCTX', 'EURTNDX=BCTX', 'GBPTNDX=BCTX', 'JPYTNDX=BCTX']
# start_date = datetime.datetime(2025, 1, 6)
# end_date = datetime.datetime.today()

# # Fetch per-minute bid/ask data
# try:
#     df = rd.get_history(
#         universe=rics,
#         fields=['BID', 'ASK'],
#         start=start_date,
#         end=end_date,
#         interval='1min'  # Per-minute interval
#     )

#     if df.empty:
#         print("No data available for the specified interval.")
#     else:
#         print(df)

#         # Reset index to flatten the MultiIndex and make it easier to save
#         df = df.reset_index()

#         # Save to a single JSON file
#         df.to_json("bctx.json", orient="records", date_format="iso")
#         print("Data successfully saved to bctx.json")

# except Exception as e:
#     print(f"Error fetching data: {e}")

# # Close the session
# rd.close_session()

import eikon as ek
import pandas as pd
import datetime
import refinitiv.data as rd
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set Eikon app key and port number
ek.set_app_key("20af0572a6364fe8abf9a35cdd16bd367057564a")
ek.set_port_number(9000)  # Default proxy port for Eikon Desktop

# Open a session (ensure Eikon or Workspace is running)
rd.open_session()

# Specify RICs and date range
rics = ['TND=BCTX', 'EURTNDX=BCTX', 'GBPTNDX=BCTX', 'JPYTNDX=BCTX']
start_date = datetime.datetime(2024, 8, 1)
end_date = datetime.datetime.today()

# Fetch per-minute bid/ask data
try:
    df = rd.get_history(
        universe=rics,
        fields=['BID', 'ASK'],
        start=start_date,
        end=end_date,
        interval='1min'  # Per-minute interval
    )

    if df.empty:
        print("No data available for the specified interval.")
    else:
        print(df)

        # Reset index to flatten MultiIndex (Timestamp, RIC)
        df = df.reset_index()

        # Rename columns to remove tuple formatting
        df.columns = ['Timestamp'] + [f"{ric}_{field}" for ric, field in df.columns[1:]]

        # Save to JSON file in a cleaner format
        df.to_json("bctx.json", orient="records", date_format="iso")
        print("Data successfully saved to bctx.json")

except Exception as e:
    print(f"Error fetching data: {e}")

# Close the session
rd.close_session()
