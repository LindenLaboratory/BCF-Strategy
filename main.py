#IMPORTS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import os
from scipy.stats import norm
from scipy.optimize import fsolve
import math
import shutil

#SETUP
spreaddata = False
testlst = []
with open('data.txt', 'r') as f:
  data = [line.replace('\n','').strip() for line in f.readlines()]
startdate = str(data[0])
enddate = str(datetime.now().date())
ticker = data[1]
deltalst = [float(item) for item in data[2].split(',')]
deltalst_ = [float(item) for item in data[8].split(',')]
days,momentumtime = int(data[3].split(',')[0]),int(data[3].split(',')[1])
riskfreerate = float(data[4])
cutoff,mincutoff = float(data[5].split(',')[0]),float(data[5].split(',')[1])
bds=data[6]
try: char=str(data[7])
except: char=''
#FUNCTIONS
def getdata(ticker, startdate, enddate):
  dividends_data = pd.DataFrame()
  dividends_data[ticker] = yf.Ticker(ticker).dividends.loc[startdate:enddate]
  NIFTY = yf.download(ticker, start=startdate, end=enddate)
  NIFTY['Log_Ret'] = np.log(NIFTY['Close'] / NIFTY['Close'].shift(1))
  NIFTY['Volatility'] = NIFTY['Log_Ret'].rolling(window=252).std() * np.sqrt(252)
  data_with_prices = NIFTY.reset_index()[['Date', 'Volatility', 'Close']].dropna()
  return dividends_data,data_with_prices
def savedata(data,ticker,path):
  dd,dwp = data
  dd.index = dd.index.tz_localize(None)
  dwp['Date'] = dwp['Date'].dt.tz_localize(None)
  merged_data = pd.merge(dwp, dd, how='left', left_on='Date', right_index=True)
  merged_data[ticker].fillna(method='ffill', inplace=True)
  merged_data.to_csv(path, index=False)
  print(merged_data)
def addyield(input_file, output_file,multiplier):
  data = []
  with open(input_file, 'r') as file:
      reader = csv.reader(file)
      header = next(reader)
      data.append(header + ['Yield'])
      data.extend(row + [(float(row[3])*multiplier) / float(row[2])] if all(row) else row for row in reader)
  with open(output_file, 'w', newline='') as file:
      csv.writer(file).writerows(data)
def getstrike(stock_price, time_to_expiration, interest_rate, volatility, dividend_yield, target_delta):
  global testlst
  testlst.append(f"{stock_price}, {time_to_expiration}, {interest_rate}, {volatility}, {dividend_yield}, {target_delta}")
  def delta_equation(K):
      d1 = (math.log(stock_price / K) + (interest_rate - dividend_yield + (volatility ** 2) / 2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
      return math.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1) - target_delta
  result = fsolve(delta_equation, stock_price)
  return result[0]
def addstrike(input_file, output_file,time,riskfreerate,delta):
  df = pd.read_csv(input_file)
  df['strike'] = df.apply(lambda row: getstrike(row['Close'], time, riskfreerate, row['Volatility'], row['Yield'], delta), axis=1)
  df.to_csv(output_file, index=False)
def delfile(_):
  files = os.listdir()
  for file in files:
    if file.endswith(".csv") and 'v' in file:
      os.remove(file)
  if _:
    shutil.rmtree('gcm')
    os.mkdir('gcm')
def addfutureprice(input_path,output_path, days_ahead):
  df = pd.read_csv(input_path)
  df['Future Price'] = pd.Series([None] * len(df))
  df['Future Price'][:-days_ahead] = df['Close'][days_ahead:].values
  df.to_csv(output_path, index=False)
def cleanup(input_file, output_file, _string_):
  with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        if any(cell == '' for cell in row):
            continue
        writer.writerow(row)
  with open(output_file, 'r', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    rows[0] = _string_.split(',')
  with open(output_file, 'w', newline='') as csvfile:
    csv.writer(csvfile).writerows(rows)
def addpercentagediff(input_file, output_file):
  df = pd.read_csv(input_file)
  df['Percent Diff.'] = (df['Future Price'] / df['Strike Price']) - 1
  df.to_csv(output_file, index=False)
def addmomentum(csv_path, days):
  df = pd.read_csv(csv_path)
  df['Momentum'] = float('nan')
  for i in range(days, len(df)):
      current_price = df.at[i, 'Price']
      past_price = df.at[i - days, 'Price']
      momentum = ((current_price - past_price) / past_price)
      df.at[i, 'Momentum'] = momentum
  df.to_csv(csv_path, index=False)
def addformulas(file_path,bdsleng):
  with open(file_path, 'r') as file:
      reader = csv.reader(file)
      lines = list(reader)
  num_lines = len(lines)
  if bdsleng!="N/A":
    new_line = ["Analysis",f'=AVERAGE(B2:B{num_lines})', f'=(((C${num_lines}/C$2)^(1/(($A${num_lines}-$A$2+1)/365)))-1)', f'=E{num_lines}*(1-H{num_lines+1})', f'=(((E${num_lines}/E$2)^(1/(($A${num_lines}-$A$2+1)/365)))-1)', f'=(AVERAGE(F2:F{num_lines}) - AVERAGE(C2:C{num_lines}))/(AVERAGE(F2:F{num_lines}) + AVERAGE(C2:C{num_lines}))',f'=(AVERAGE(G2:G{num_lines}) - AVERAGE(F2:F{num_lines}))/(AVERAGE(G2:G{num_lines}) + AVERAGE(F2:F{num_lines}))',f'=COUNTIF(H2:H{num_lines}, ">" & 0)/{num_lines}',f'=AVERAGE(I2:I{num_lines})',f'=AVERAGE(J2:J{num_lines})*(365/{days})',f'=(AVERAGE(K2:K{num_lines})*(365/{days}))',f'=AVERAGE(L2:L{num_lines})*(({bdsleng}/{num_lines})*(365/{days}))']
  else:
    new_line = ["Analysis",f'=AVERAGE(B2:B{num_lines})', f'=(((C${num_lines}/C$2)^(1/(($A${num_lines}-$A$2+1)/365)))-1)', f'=E{num_lines}*(1-H{num_lines+1})', f'=(((E${num_lines}/E$2)^(1/(($A${num_lines}-$A$2+1)/365)))-1)', f'=(AVERAGE(F2:F{num_lines}) - AVERAGE(C2:C{num_lines}))/(AVERAGE(F2:F{num_lines}) + AVERAGE(C2:C{num_lines}))',f'=(AVERAGE(G2:G{num_lines}) - AVERAGE(F2:F{num_lines}))/(AVERAGE(G2:G{num_lines}) + AVERAGE(F2:F{num_lines}))',f'=COUNTIF(H2:H{num_lines}, ">" & 0)/{num_lines}',f'=AVERAGE(I2:I{num_lines})',f'=IFERROR((AVERAGE(J2:J{num_lines})*(365/{days})),0)']
  lines.append(new_line)
  with open(file_path, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(lines)
def cleandata(input_file, output_file, cutoff):
  with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    rows_to_process = list(reader)[:-1]
    for row in rows_to_process:
        try:
          if cutoff != "N/A":
            if all(cell.strip() for cell in row.values()) and float(row.get('Momentum', 0)) >= cutoff:
              writer.writerow(row)
          else:
            if all(cell.strip() for cell in row):
              writer.writerow(row)
        except (ValueError, KeyError):
          print("Error processing row:", row)
def blackscholesmerton(spot, strike, time, riskfreerate, volatility, dividend):
  d1 = (math.log(spot / strike) + (riskfreerate - dividend + 0.5 * volatility ** 2) * time) / (volatility * math.sqrt(time))
  d2 = d1 - volatility * math.sqrt(time)
  call_price = spot * math.exp(-dividend * time) * norm.cdf(d1) - strike * math.exp(-riskfreerate * time) * norm.cdf(d2)
  return call_price
def blackscholesmertonput(spot, strike, time, riskfreerate, volatility, dividend):
  d1 = (math.log(spot / strike) + (riskfreerate - dividend + 0.5 * volatility ** 2) * time) / (volatility * math.sqrt(time))
  d2 = d1 - volatility * math.sqrt(time)
  put_price = strike * math.exp(-riskfreerate * time) * norm.cdf(-d2) - spot * math.exp(-dividend * time) * norm.cdf(-d1)
  return put_price
def addoptprice(inpath,outpath, time, riskfreerate):
  df = pd.read_csv(inpath)
  option_prices = df.apply(lambda row: blackscholesmerton(row['Price'], row['Strike Price'], time, riskfreerate, row['Volatility'], row['Dividend Yield'])/row['Price'], axis=1)
  df['Opt. Price'] = option_prices
  df.to_csv(outpath, index=False)
def finalclean(inpath, outpath, lines_to_remove):
  with open(inpath, 'r', newline='') as csvfile:
      reader = csv.reader(csvfile)
      data = list(reader)
  header = data[0]
  remaining_data = data[lines_to_remove + 1:]
  with open(outpath, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(header)
      writer.writerows(remaining_data)
def addadjoptprice(inpath,outpath):
  df=pd.read_csv(inpath)
  df['Adj. Opt. Price'] = df.apply(lambda row: row['Opt. Price'] - row['Percent Diff.'] if row['Percent Diff.'] > 0 else row['Opt. Price'], axis=1)
  df.to_csv(outpath, index=False)
def addbdsgain(datav6_file, gcmdata_file, cutoff,mincutoff,days,riskfreerate,bds,targetdelta___):
  global spreaddata
  datav6 = pd.read_csv(datav6_file)
  filtered_rows = datav6[(datav6['Momentum'] < cutoff) & (datav6['Momentum'] > mincutoff)]
  filtered_rows['Call Gains'] = (filtered_rows['Future Price'] - filtered_rows['Price']) / filtered_rows['Price']
  with open(gcmdata_file, 'r') as file:
      lines = file.readlines()
  bdslst,vollst,divlst = [],[],[]
  for index, row in filtered_rows.iterrows():
      bdslst.append(row['Call Gains'])
      vollst.append(row['Volatility'])
      divlst.append(row['Dividend Yield'])
  bdsleng=len(bdslst)
  if bds=='None':
    bdsleng = 0
  for i in range(1, len(lines)):
      if i - 1 < len(bdslst) and bds != 'None':
        global testlst
        try:
          strike = getstrike(80, days/365, riskfreerate, vollst[i-1], divlst[i-1], targetdelta___)
        except:
          print(testlst[-1],testlst[-2])
        optprice_=blackscholesmerton(80,80,days/365,riskfreerate,vollst[i-1],divlst[i-1])
        scndoptprice_=blackscholesmerton(80,strike,days/365,riskfreerate,vollst[i-1],divlst[i-1])
        thrdoptprice_=blackscholesmertonput(80,80-(strike/2),days/2/365,riskfreerate,vollst[i-1],divlst[i-1])
        thrdoptprice_=thrdoptprice_+(2*divlst[i-1])
        if not spreaddata:
          with open('gcm/spreaddata.csv','a') as f:
            f.write(f'={optprice_}-{scndoptprice_}+{thrdoptprice_}\n')
        if bdslst[i - 1] > 0.05:
          lines[i] = lines[i].strip() + f'{0.05+(scndoptprice_/100)-(optprice_/100)-(thrdoptprice_/100)}\n'
        elif bdslst[i - 1] > 0:
          lines[i] = lines[i].strip() + f'{bdslst[i - 1]-(optprice_/100)+(scndoptprice_/100)-(thrdoptprice_/100)}\n'
        else:
          if bdslst[i - 1] < -0.025:
            profit = (bdslst[i-1]/4.5)*-1
          else:
            profit = (bdslst[i-1]+0.025)*-1
          lines[i] = lines[i].strip() + f'{profit-(optprice_/100)+(scndoptprice_/100)}\n'
      else:
          lines[i] = lines[i].strip() + '-\n'
  with open(gcmdata_file, 'w') as file:
      file.writelines(lines)
  spreaddata=True
  return bdsleng
#MAINLOOP
inp = input('Delete gcmdata (y/n) ')
if inp == 'y': delfile(True)
else: delfile(False)
timemtplr=int(input("Enter time multiplier: "))
for delta in deltalst:
  for delta_ in deltalst_:
    savedata(getdata(ticker, startdate, enddate),ticker,f"datav1:{delta}:{delta_}{char}.csv")
    addyield(f"datav1:{delta}:{delta_}{char}.csv",f"datav2:{delta}:{delta_}{char}.csv",timemtplr)
    addstrike(f"datav2:{delta}:{delta_}{char}.csv",f"datav3:{delta}:{delta_}{char}.csv",days/365,riskfreerate,delta)
    addfutureprice(f"datav3:{delta}:{delta_}{char}.csv",f"datav4:{delta}:{delta_}{char}.csv",days)
    cleanup(f"datav4:{delta}:{delta_}{char}.csv", f"datav5:{delta}:{delta_}{char}.csv","Date,Volatility,Price,Dividend Amount,Dividend Yield,Strike Price,Future Price,Percent Diff.,Momentum,Opt. Price,Adj. Opt. Price,Call Gains")
    addpercentagediff(f"datav5:{delta}:{delta_}{char}.csv",f"datav6:{delta}:{delta_}{char}.csv")
    addmomentum(f"datav6:{delta}:{delta_}{char}.csv",momentumtime)
    addoptprice(f"datav6:{delta}:{delta_}{char}.csv",f"datav7:{delta}:{delta_}{char}.csv",days/365,riskfreerate)
    print(bds,type(bds))
    bdsleng=addbdsgain(f"datav6:{delta}:{delta_}{char}.csv",f"datav7:{delta}:{delta_}{char}.csv",cutoff,mincutoff,days,riskfreerate,bds,delta_)
    addadjoptprice(f"datav7:{delta}:{delta_}{char}.csv",f"datav8:{delta}:{delta_}{char}.csv")
    cleandata(f"datav8:{delta}:{delta_}{char}.csv",f"datav9:{delta}:{delta_}{char}.csv",cutoff)
    finalclean(f"datav9:{delta}:{delta_}{char}.csv",f"gcm/gcmdata:{delta}:{delta_}{char}.csv",momentumtime)
    addformulas(f"gcm/gcmdata:{delta}:{delta_}{char}.csv",bdsleng)
    delfile(False)
  files = os.listdir('gcm')
  for file in files:
    print(file.replace(":","_"),end=",")
