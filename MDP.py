import pandas as pd
import numpy as np
import math
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

#Reading the datasets
hrt_df = pd.read_csv('D:\Imp_Files\Codes\Old\Codes\ML\MDP\heart_attack.csv')
cnr_df = pd.read_csv('D:\Imp_Files\Codes\Old\Codes\ML\MDP\Cancer_data.csv')
dbt_df = pd.read_csv('D:\Imp_Files\Codes\Old\Codes\ML\MDP\diabetes.csv')

#DataPreprocessing for Heart Model
oe = OrdinalEncoder()
hrt_df['Result'] = oe.fit_transform(hrt_df[['Result']])
hrt_df = hrt_df.rename(columns = {'Result': 'Heart Attack'})
hrt_df['BP_ratio'] = hrt_df['Systolic blood pressure'] / hrt_df['Diastolic blood pressure']
hrt_df.drop(columns=['Systolic blood pressure','Diastolic blood pressure'],inplace = True)

#DataPreprocessing for Diabetes Model
dbt_df = dbt_df.rename(columns = {'Outcome':'Diabetic'})

#DataPreprocessing for Cancer Model
cnr_df.BMI = cnr_df.BMI.round(2)
cnr_df.PhysicalActivity = cnr_df.PhysicalActivity.round(2)
cnr_df.AlcoholIntake = cnr_df.AlcoholIntake.round(2)

#Landing Page
st.title("Multi-Disease Prediction\n")
st.write("")
st.info("\nThis is a Machine Learning Ensemble Model which helps the user to predict the outcome of Multiple Diseases "
"by using some famous algorithms like Logistic Regresion,SVM,KNN,Random Forest,AdaBoost and Votingclassifier.This Model is "
"build upon he very concept of Ensemble Learning which increases the overall Accuracy of the model.\n")
st.write("")
option=st.sidebar.radio("Menu",['Home','Heart Attack','Diabetes','Cancer'])
#Starting of Sidebar
if option == 'Home':
    #Selecting the Diseases Model deals with
    menu=st.selectbox("The Model Works for following diseases",["Heart Attack","Diabetes","Cancer"])
    if menu == "Heart Attack":
        #Genral Info about Diseases
        st.markdown("### Heart Attacks Don’t Wait – Neither Should You!\n")
        st.subheader("General Information\n")
        st.write("\n")
        st.info("Heart attacks remain a leading cause of death worldwide, affecting millions each year. Statistically, over 17.9 million "
        "people die annually from cardiovascular diseases, with heart attacks contributing significantly. High blood pressure, diabetes, "
        "smoking, and obesity increase risks. Early detection and lifestyle changes can prevent fatalities, offering hope for healthier lives.\n")

        #Data Visualization Part
        st.subheader("Data Visualization\n")
        label = ['Heart Attacks' ,'No Heart Attacks']
        fig,ax = plt.subplots(figsize = (4.5,4.5))
        plt.pie(hrt_df['Heart Attack'].value_counts(),autopct='%0.1f%%',shadow=True,explode = [0.1,0],labels = label)
        st.pyplot(fig)


        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(hrt_df[hrt_df['Heart Attack']==1.0]['Age'],color = 'r',kde =True) #incase of  heartattack (output = 1)
        plt.title('Age vs Heartattack')
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.countplot(x='Gender', data=hrt_df,width = 0.5) #1 for Male and 0 for Female
        plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
        plt.title('Gender Distribution')
        st.pyplot(fig)

        st.warning("Select the sidebar option to Predict your Disease")

    elif menu=="Diabetes":
        
        #Genral Info about Diseases
        st.markdown("### Know Diabetes, Fight Diabetes!\n")
        st.subheader("General Information\n")
        st.write("\n")
        st.info("Diabetes affects over 537 million adults globally, with cases expected to rise. This chronic condition can lead to heart disease,"
        " kidney failure, and blindness. Sadly, many remain undiagnosed until complications arise. However, with proper management—healthy eating,"
        " exercise, and medication—millions can lead fulfilling lives despite the challenges of this disease./n")

        #Data Visualization Part
        st.subheader("Data Visualization\n")
        label = ['Non-Diabetic','Diabetic']
        fig,ax = plt.subplots(figsize = (4.5,4.5))
        plt.pie(dbt_df['Diabetic'].value_counts(),autopct='%0.1f%%',shadow=True,explode = [0.1,0],labels = label)
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(dbt_df[dbt_df['Diabetic']==1]['Glucose'],kde = True,color = 'r',bins = [10,20,40,60,80,100,120,140,160,180,200,220])
        sns.distplot(dbt_df[dbt_df['Diabetic']==0]['Glucose'],kde = True,color = 'g',bins = [10,20,40,60,80,100,120,140,160,180,200,220])
        plt.title('Red(Diabetic) vs Green(Non Diabetic)')
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(dbt_df[dbt_df['Diabetic']==1]['BloodPressure'],kde = True,color = 'r',bins = [10,20,40,60,80,100,120,140,160,180,200,220])
        sns.distplot(dbt_df[dbt_df['Diabetic']==0]['BloodPressure'],kde = True,color = 'g',bins = [10,20,40,60,80,100,120,140,160,180,200,220])
        plt.title('Red(Diabetic) vs Green(Non Diabetic)')
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(dbt_df[dbt_df['Diabetic']==1]['Insulin'],kde = True,color = 'r',bins = [10,20,40,60,80,100,120,140,160,180,200,250,300])
        sns.distplot(dbt_df[dbt_df['Diabetic']==0]['Insulin'],kde = True,color = 'g',bins = [10,20,40,60,80,100,120,140,160,180,200,250,300])
        plt.title('Red(Diabetic) vs Green(Non Diabetic)')
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(dbt_df[dbt_df['Diabetic']==1]['DiabetesPedigreeFunction'],kde = True,color = 'r')
        sns.distplot(dbt_df[dbt_df['Diabetic']==0]['DiabetesPedigreeFunction'],kde = True,color = 'g')
        plt.title('Red(Diabetic) vs Green(Non Diabetic)')
        st.pyplot(fig)

        st.warning("Select the sidebar option to Predict your Disease")

    elif menu=='Cancer':

        #Genral Info about Diseases
        st.markdown("### Be Aware. Be Strong. Beat Cancer!\n")
        st.subheader("General Information\n")
        st.write("\n")
        st.info("Cancer remains a global health crisis, causing nearly 10 million deaths annually. One in six deaths worldwide is due to cancer, affecting"
        " families emotionally and financially. Early detection and advanced treatments improve survival rates, offering hope. Raising awareness and "
        "supporting research are crucial in the fight against this devastating disease.\n")

        #Data Visualization Part
        st.subheader("Data Visualization/n")

        label = ['No Cancer' ,'Cancer']
        fig,ax = plt.subplots(figsize = (4.5,4.5))
        plt.pie(cnr_df['Diagnosis'].value_counts(),autopct='%0.1f%%',shadow=True,explode = [0.1,0],labels = label)
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(cnr_df[cnr_df['Diagnosis'] == 1]['Age'],kde = True,color = 'b')
        plt.title('Age Distribiution vs Chances of getting Cancer')
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(cnr_df[cnr_df['Diagnosis'] == 1]['AlcoholIntake'],kde = True,color = 'g')
        plt.title('Alcohol Intake Per week vs Chances of getting Cancer')
        st.pyplot(fig)

        fig,ax = plt.subplots(figsize = (10,5))
        sns.distplot(cnr_df[cnr_df['Diagnosis'] == 1]['BMI'],kde = True,color = 'r')
        plt.title('BMI vs Chances of getting Cancer')
        st.pyplot(fig)

        st.warning("Select the sidebar option to Predict your Disease")

#Prediction Model for Heart Attack
elif option == 'Heart Attack':

    heart_ensem_clf = joblib.load('D:\Imp_Files\Codes\Old\Codes\ML\MDP\models\heart_attack_model.joblib')

    # UI for Heart Attack Prediction
    st.header('Heart Attack Prediction')
    st.subheader("Enter the values:")

    hrt_age = st.number_input("Age (in Years)", max_value=100, min_value=1)
    gender = st.selectbox("Gender", options=['Female', 'Male'])
    heart_rate = st.number_input("Heart Rate (in BPM)", max_value=500, min_value=30)
    blood_sugar = st.number_input("Blood Sugar", min_value=0.0)
    ckmb = st.number_input("CK-MB", min_value=0.0)
    troponin = st.number_input("Troponin", format="%.3f", min_value=0.0)
    bp_ratio = st.number_input("BP Ratio (Systolic / Diastolic)", format="%.3f", min_value=0.1)

    #Preprocess input
    gender_int = 1 if gender == "Male" else 0

    hrt_input_data = pd.DataFrame([[
        hrt_age, gender_int, heart_rate, blood_sugar, ckmb, troponin, bp_ratio
    ]], columns=['Age', 'Gender', 'Heart rate', 'Blood sugar', 'CK-MB', 'Troponin', 'BP_ratio'])

    #Prediction
    if st.button('Predict'):
        hrt_input_pred = heart_ensem_clf.predict(hrt_input_data)

        if hrt_input_pred[0] == 1:
            st.error('Heart Attack Risk is VERY HIGH')
        else:
            st.success('Heart Attack Risk is VERY LOW')

        st.info('Prediction made using pre-trained ensemble model.')

#Prediction Model for Diabetes
elif option == 'Diabetes':

    diabetes_model = joblib.load('D:\Imp_Files\Codes\Old\Codes\ML\MDP\models\diabetes_model.joblib')
    # UI
    st.header('Diabetes Prediction')
    st.subheader('Enter the values:')

    # User Inputs
    preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
    gluc = st.number_input("Glucose level in Blood", min_value=0)
    bp = st.number_input("Blood Pressure (in mm Hg)", min_value=0)
    skts = st.number_input("Skin Thickness", min_value=0)
    ins = st.number_input("Insulin level in Blood", format='%.3f', min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dp = st.number_input("Diabetes Pedigree Function", format='%.3f', min_value=0.0)
    dbt_age = st.number_input("Age (in years)", min_value=1, max_value=120)

    # Prepare input for prediction
    dbt_input_data = pd.DataFrame([[
        preg, gluc, bp, skts, ins, bmi, dp, dbt_age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    # Prediction
    if st.button('Predict'):
        dbt_input_pred = diabetes_model.predict(dbt_input_data)
        
        if dbt_input_pred[0] == 1:
            st.error('Diabetes Risk is VERY HIGH')
        else:
            st.success('Diabetes Risk is VERY LOW')

        st.info('Prediction made using pre-trained ensemble model.')

#Prediction Model for Cancer
elif option == 'Cancer':

    cancer_ensem_clf = joblib.load('D:\Imp_Files\Codes\Old\Codes\ML\MDP\models\cancer_model.joblib')

    sc = joblib.load('D:\Imp_Files\Codes\Old\Codes\ML\MDP\models\cancer_scaler.joblib')

    # User Input Section
    st.header('Cancer Prediction')
    st.subheader('Enter the values:')

    cnr_age = st.number_input('Age (in years)', max_value=100)
    cnr_gender = st.selectbox("Gender", options=['Female', 'Male'])
    cnr_bmi = st.number_input('BMI')
    cnr_smoking = st.selectbox("Smoking", options=['Yes', 'No'])
    cnr_gr = st.selectbox("Genetic Risk", options=['Low', 'Medium', 'High'])
    cnr_pact = st.number_input("Physical Activity (hours/week)", max_value=10)
    cnr_alc = st.number_input("Alcohol Intake (units/week)", max_value=5)
    cnr_hist = st.selectbox("Cancer History", options=['Yes', 'No'])

    # Preprocess input
    cnr_gender_int = 1 if cnr_gender == "Female" else 0
    cnr_smoking_int = 1 if cnr_smoking == 'Yes' else 0
    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    cnr_gr_int = mapping[cnr_gr]
    cnr_hist_int = 1 if cnr_hist == 'Yes' else 0

    cnr_input_data = pd.DataFrame([[
        cnr_age, cnr_gender_int, cnr_bmi, cnr_smoking_int,
        cnr_gr_int, cnr_pact, cnr_alc, cnr_hist_int
    ]], columns=[
        'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
        'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
    ])

    # Scale the input
    sc_cnr_input_data = sc.transform(cnr_input_data)

    # Predict and Show Result
    if st.button('Predict'):
        cnr_input_pred = cancer_ensem_clf.predict(sc_cnr_input_data)
        if cnr_input_pred[0] > 0:
            st.error('Cancer Risk is VERY HIGH')
        else:
            st.success('Cancer Risk is VERY LOW')

        st.info('Prediction made using pre-trained ensemble model.')

                        






        