import streamlit as st
import joblib
import pandas as pd
from source import *

# preparing model
one_hot_encoder = joblib.load("/Volumes/Data/deployment/klasifikasi/one_hot_encoder.joblib")
model = joblib.load("/Volumes/Data/deployment/klasifikasi/model.joblib")

# prepare result
result = 0


st.title("Klasifikasi Akseptabilitas Mobil")
st.write("Klasifikasi kelayakan penerimaan mobil berguna untuk mengetahui apakah mobil tersebut layak masuk pemasaran atau tidak berdasarkan beberapa parameter.")


col_1, col_2 = st.columns(2)

with col_1:
   opt_buying_price = st.selectbox("Harga Beli", buying_price)
   opt_nb_doors = st.selectbox("Jumlah Pintu", nb_doors)
   opt_luggage = st.selectbox("Kapasitas Bagasi", size_of_luggage)

with col_2:
   opt_maintenance_price = st.selectbox("Harga Perawatan", maintenance_price)
   opt_person = st.selectbox("Kapasitas Penumpang", person_capacity)
   opt_safety = st.selectbox("Keamanan", safety)

with col_1:
   button_predict = st.button("Prediksi")

with st.container(border=True):
   if(button_predict):

      # inisilisasi dataframe baru 
      df = pd.DataFrame({
         "No_of_Doors": [opt_nb_doors],
         "Person_Capacity": [opt_person],
         "Buying_Price": [opt_buying_price],
         "Maintenance_Price": [opt_maintenance_price],
         "Size_of_Luggage": [opt_luggage],
         "Safety": [opt_safety]
      })

      # menentukan column categorical dan numerical
      categories = df.select_dtypes(include=['object']).columns.to_list()
      numeric = df.select_dtypes(include=['int64']).columns.to_list()

      # transform categorical dengan one hot encoding
      encoded = one_hot_encoder.transform(df[categories])

      # ubah hasil transform ke dataframe
      one_hot_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(categories))

      # gabung dataframe numerik dan categorik
      new_df = pd.concat([df[numeric], one_hot_df], axis=1)

      # prediksi dengan model SVM
      predict = model.predict(new_df)[0]
      predict_proba = model.predict_proba(new_df)[0]
      predict_proba = [round(x*100, 2) for x in predict_proba]

      # membuat dataframe metrik probabilitas hasil perhitungan
      probability_metrics = pd.DataFrame({
         "Diterima": [f"{predict_proba[0]}%"],
         "Bagus": [f"{predict_proba[1]}%"],
         "Tidak Diterima": [f"{predict_proba[2]}%"],
         "Sangat Bagus": [f"{predict_proba[3]}%"],
      })

      # penentuan kelas
      labels = ['Diterima', 'Bagus', 'Tidak Diterima', 'Sangat Bagus']

      # tampilkan di web
      st.write(f"Prediksi : {labels[predict]}")
      st.write("Probabilitas Setiap Kelas :")
      st.table(probability_metrics)

