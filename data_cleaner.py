import pandas as pd

#Read the file
aapl = pd.read_csv('data/aapl.csv')
amzn = pd.read_csv('data/amzn.csv')
cvs = pd.read_csv('data/cvs.csv')
nvda = pd.read_csv('data/nvda.csv')
wmt = pd.read_csv('data/wmt.csv')

# Create a new file only with data and prices:


def clean_and_order(df, startdate, enddate):

    # Convert date, filter range and order it:

    df["Date"]=pd.to_datetime(df["Date"])
    df = df[ (df["Date"]>=startdate)&(df["Date"]<=enddate)]
    df = df.sort_values("Date")

    # Remove $ from price and covert to float:

    df["Close/Last"]=df["Close/Last"].astype(str)  # lo pasamos a string
    df["Close/Last"]=df["Close/Last"].str.replace("$", "").str.replace(",", "") #eliminamos $ y ,
    df["Close/Last"]=df["Close/Last"].astype(float)  # lo pasamos a float

    return df[["Date","Close/Last"]]

stocks={"AAPL":aapl,
        "AMZN":amzn,
        "CVS":cvs,
        "NVDA":nvda,
        "WMT":wmt}

cleaned = pd.DataFrame()

for name, df in stocks.items():
    temp = clean_and_order(df,"01/01/2022","12/31/2022")
    temp = temp.set_index('Date')
    cleaned[name]=temp["Close/Last"]

print("Stock closes:")
print(cleaned.head())

#Calculate daily returns:
returns = (cleaned - cleaned.shift(1)) / cleaned.shift(1)

#Drop the first row (is empty):
returns= returns.dropna()

print("Daily returns: ")
print(returns.head())

