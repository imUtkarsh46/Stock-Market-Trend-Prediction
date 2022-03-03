
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
from datetime import datetime
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")



start = datetime(2011, 1, 1)
end = datetime(2021, 12, 31)




st.title('Stock Prediction Tool')
user_input = st.text_input('Enter Stock Ticker (ex. Reliance Industries Ltd - "RELIANCE")', 'AAPL')
df = web.DataReader(user_input, 'yahoo', start, end)

DataF = pd.DataFrame()
DataF['Actual'] = df['Close']
DataF['Day100'] = df['Close'].rolling(100).mean()
DataF['Day200'] = df['Close'].rolling(200).mean()

st.subheader('Data From 2011-2021')
st.write(df.describe())

st.subheader('Closing Price VS Time')
st.line_chart(data=df['Close'], width=2, height=0, use_container_width=True)

st.subheader('Closing Price VS Time "100 Days Moving Average')
st.line_chart(data=DataF['Day100'], width=2, height=0, use_container_width=True)

st.subheader('Closing Price VS Time "100 VS 200 DaysMoving Average')
st.line_chart(data=DataF, width=2, height=0, use_container_width=True)

train_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test_data = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_data_arr = scaler.fit_transform(train_data)


#Load Model
model = load_model('Kerasfinalmodel.h5')

#Testing
past100_day = train_data.tail(100)
final_df = past100_day.append(test_data, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

#Prediction
Y_pred = model.predict(x_test)
scaler = scaler.scale_  

scale_factor = 1/scaler[0]
Y_pred = Y_pred*scale_factor
y_test = y_test*scale_factor

Pred = pd.DataFrame()
Pred['Original Value'] = y_test
Pred['Predicted Value'] = Y_pred


st.subheader('Final Predicted Chart')
st.line_chart(data=Pred, width=2, height=0, use_container_width=True)



