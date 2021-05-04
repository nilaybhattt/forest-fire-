import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
import pickle
model=pickle.load(open('model.pkl','rb'))

matplotlib.use('Agg')
from PIL import Image

st.title('Forest Fire Prediction/Analysis')
image=Image.open('forest.jpg')
st.image(image,use_column_width=True)
@st.cache
def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    activities=['EDA','Visualisation','Prediction','Effects']
    option=st.sidebar.selectbox('Selection option:',activities)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if option=='EDA':
        st.subheader("Exploratory data analysis")
        data=st.file_uploader("Upload your dataset:",type=['csv','xlsx','txt','json'])
        
        if data is not None:
            st.success("Data successfully uploaded")
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_column=st.multiselect('Select prefered columns:',df.columns)
                df1=df[selected_column]
                st.dataframe(df1)  
            if st.checkbox("Display summary"):
                st.write(df1.describe().T) 
            if st.checkbox("Display datatypes"):
                st.write(df.dtypes)
            if st.checkbox("Display Correlation of data various columns"):
                st.write(df.corr())
            
        
    




    elif option=='Visualisation':
        st.subheader("Visualisation")
        data=st.file_uploader("Upload your dataset:",type=['csv','xlsx','txt','json'])
        
        if data is not None:
            st.success("Data successfully uploaded")
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple Columns to plot'):
                selected_column=st.multiselect('Select your preffered columns',df.columns)
                df1=df[selected_column]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                st.write(sb.heatmap(df1.corr(),vmax=1,vmin=0, xticklabels=True, yticklabels=True,square=True,annot=True,cmap='viridis'))
                st.pyplot()
            if st.checkbox('Display Pairplot'):
                st.write(sb.pairplot(df1,diag_kind='kde'))
                st.pyplot()
            if st.checkbox('Display Countplot'):
                st.write(df.Cover_Type.value_counts())
                st.write(sb.countplot(x='Cover_Type',data=df))
                st.pyplot()
            if st.checkbox('Display Histogram'):
                df1.hist(figsize=(13,11))
                st.pyplot()
           
            if st.checkbox("Visualize Columns wrt Classes"):
                st.write("#### Select column to visualize: ")
                columns = df.columns.tolist()
                class_name = columns[-1]
                column_name = st.selectbox("",columns)
                st.write("#### Select type of plot: ")
                plot_type = st.selectbox("", ["kde","box"])
                if st.button("Generate"):
                    if plot_type == "kde":
                        st.write(sb.FacetGrid(df, hue=class_name, palette="husl", height=6).map(sb.kdeplot, column_name).add_legend())
                        st.pyplot()

                    if plot_type == "box":
                        st.write(sb.boxplot(x=class_name, y=column_name, palette="husl", data=df))
                        st.pyplot()

                   


                        

    if option=='Prediction':
        st.subheader("Predection")
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        oxygen = st.text_input("Oxygen","Type Here")
        humidity = st.text_input("Humidity","Type Here")
        temperature = st.text_input("Temperature","Type Here")
        safe_html="""  
        <div style="background-color:#F4D03F;padding:10px >
        <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
        </div>
        """
        danger_html="""  
        <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
        </div>
        """

        if st.button("Predict"):
            output=predict_forest(oxygen,humidity,temperature)
            st.success('The probability of fire taking place is {}'.format(output))

            if output > 0.5:
                st.markdown(danger_html,unsafe_allow_html=True)
            else:
                st.markdown(safe_html,unsafe_allow_html=True)

    elif option=='Effects':
        st.subheader("Effects due to forest fire")
        dataf=pd.read_csv("Forest_Fire.csv")
        dataf.sort_values("Fire Occurrence", inplace=True)
        filter=dataf["Fire Occurrence"]==1
        dataf.where(filter,inplace=True)
        dataf=dataf.dropna()
        st.dataframe(dataf.head(100))
        st.write("### Regions where Forest Fire Occured")
        st.map(dataf)

        st.write("### Pollution due to Forest Fire")
        df = pd.read_csv('pollution.csv')
      
        df= df.rename(columns = {" pm25": "pm25", 
                         " pm10":"pm10", 
                         " o3": "o3",
                         ' no2' : 'no2',
                         ' so2' : 'so2',
                         ' co' : 'co'})

        
        df['date'] = pd.to_datetime(df.date)
        df21 = df.loc[df['date'] > '2019-7-01']
        df21 = df21.sort_values(by = 'date')
        df21.drop(13, inplace=True)
        df21.replace(' ', '0', inplace=True)

        dates = df21['date']
        pm25 = df21['pm25']
        pm25 = [int(i) for i in pm25]
        o3 = df21['o3']
        o3 = [int(i) for i in o3]
        no2 = df21['no2']
        no2 = [int(i) for i in no2]
        so2 = df21['so2']
        so2 = [int(i) for i in so2]


        plt.figure(figsize=(10,8))
        ploti = st.selectbox("", ["pm25","O3","NO2","SO2"])
        if ploti=="pm25" :
            plt.plot(dates,pm25)
        elif ploti=="O3":
            plt.plot(dates,o3)
        elif ploti=="NO2":
            plt.plot(dates,no2)
        if ploti=="SO2":
           plt.plot(dates,so2)

        st.pyplot()

      



if __name__=='__main__':
    main()