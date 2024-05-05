import streamlit as st
import joblib
import time

# define model requirement
model =  joblib.load("naive_bayes.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("Sentimen Analisis Twitter Dataset")
st.write("Sentimen Analisis dengan menggunakan dataset twitter Pemilu tahun 2024")


with st.container():
   sentences = st.text_area("Kalimat untuk dianalisis")
   button = st.button("Prediksi")


with st.container(border=True):
   if(button):
      with st.spinner('Wait for it...'):

         # st.write(type(sentences))

         # ubah kalimat ke angka menggunakan vectorizer
         vectorized = vectorizer.transform([sentences])

         # prediksi
         predicted = model.predict(vectorized)[0]

         # hitung probabilitas
         probabilities = model.predict_proba(vectorized)[0]
         probabilities = [f"{round(x*100, 2)}%" for x in probabilities]

         probability = {
            "Negative" : probabilities[0],
            "Positive" : probabilities[1],
         }
         
         time.sleep(2)

         st.write(f"Prediksi : {predicted}")
         st.write("Probabilitas : ")
         st.table(probability)

         
