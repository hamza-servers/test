import pandas as pd
import streamlit as st
import datetime
import pickle
import joblib

import pandas as pd
import streamlit as st
import datetime
import pickle
import joblib

# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("SPEND PREDICTION MODEL")
# st.subheader('Weekly')
st.write("")
st.write("Please enter values with respect to monthly data ...")


orders = st.number_input(
    label="Monthly orders", step=1., format="%.4f")
st.write('You entered orders:', orders)

st.write(orders)

revenue = st.number_input(
    label="Monthly revenue", step=1., format="%.4f")
st.write('You entered revenue:', revenue)

averagePrice = st.number_input(
    label="Monthly average price", step=1., format="%.4f")
st.write('You entered Average Price:', averagePrice)

facebookPurchases = st.number_input(
    label="Monthly facebook purchases", step=1., format="%.4f")
st.write('You entered facebook purchases:', facebookPurchases)

facebookRevenue = st.number_input(
    label="Monthly facebook revenue", step=1., format="%.4f")
st.write('You entered Facebook Revenue:', facebookRevenue)

googleSpend = st.number_input(
    label="Monthly google spend", step=1., format="%.4f")
st.write('You entered google spend:', googleSpend)

googlePurchases = st.number_input(
    label="Monthly google purchases", step=1., format="%.4f")
st.write('You entered google purchases:', googlePurchases)

googleRevenue = st.number_input(
    label="Monthly google revenue", step=1., format="%.4f")
st.write('You entered google revenue:', googleRevenue)

facebookROAS = st.number_input(
    label="Monthly facebook ROAS", step=1., format="%.4f")
st.write('You entered facebook ROAS:', facebookROAS)

googleCPA = st.number_input(
    label="Monthly google CPA", step=1., format="%.4f")
st.write('You entered google CPA:', googleCPA)

googleROAS = st.number_input(
    label="Monthly google ROAS", step=1., format="%.4f")
st.write('You entered google ROAS:', googleROAS)

totalROAS = st.number_input(
    label="Monthly total ROAS", step=1., format="%.4f")
st.write('You entered total ROAS:', totalROAS)

targetRevenue = st.number_input(
    label="Weekly Target Revenue", step=1., format="%.4f")
st.write('You entered Target Revenue:', targetRevenue)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/30)*7


st.write('----------------------------------- FACEBOOK with CPA & Product Enlisting ----------------------------------------')

# ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']

test = {'orders': orders, 'revenue': revenue,
        'average_price': averagePrice, 'fb_purchases': facebookPurchases, 'fb_revenue': facebookRevenue,
        'ga_purchases': googlePurchases, 'ga_spend': googleSpend, 'ga_revenue': googleRevenue,'fb_roas': facebookROAS, 'ga_cpa': googleCPA, 'ga_roas': googleROAS,
        'roas': totalROAS,'target_revenue': targetRevenue}

test_DF = pd.DataFrame([test])

path = './model_scalar/'
inner = pickle.load(path+'inner.sav','rb')
outer = pickle.load(path+'outer.sav','rb')

model = pickle.load(open(path+'xgb_reg_1.pkl', "rb"))

test_DF = pd.DataFrame(inner.transform(test_DF))


test_DF = test_DF.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                  4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_cpa', 10: 'roas', 11: 'target_revenue'})


pred = model.predict(test_DF)
pred = pred.reshape(-1, 1)
pred = outer.inverse_transform(pred)
totalFB_Revenue = float(pred[0])
totalGA_Revenue = float(totalFB_Revenue*0.1)


st.write('Facebook Spend Budget :', totalFB_Revenue)
st.write('Google Spend Budget :', totalGA_Revenue)
