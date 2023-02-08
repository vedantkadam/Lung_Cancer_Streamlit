
import pickle
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image 
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 
from streamlit_option_menu import option_menu
st.set_page_config(page_title="Stock Price", page_icon="🧊", layout="centered", initial_sidebar_state = "auto")
#Loading models
cancer_model = pickle.load(open('models/final_model.sav', 'rb'))

with st.sidebar:
    selection = option_menu('Lung Cancer Detection System',
    ['Introduction',
    'About the Dataset',
    'Lung Cancer Prediction',
    'CNN Based disease Prediction'],
    icons = ['activity','heart', 'person'],
    default_index = 0)

#Introduction page
if (selection == 'Introduction'):

    from PIL import Image

    gg = Image.open("images/lung-cancer.jpg")

    st.image(gg, caption='Introduction to Lung Cancer',width=600)
    #page title
    st.title('How common is lung cancer?')

    st.write("Lung cancer (both small cell and non-small cell) is the second most common cancer in both men and women in the United States (not counting skin cancer). In men, prostate cancer is more common, while in women breast cancer is more common.")
    st.markdown(
    """
    The American Cancer Society’s estimates for lung cancer in the US for 2023 are:
    - About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)
    - About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)
    - Item 3
    """
    )

    st.write("")
    st.title("Is Smoking the only cause ?")
    mawen = Image.open("images/menwa.png")

    st.image(mawen, caption='Smoking is not the major cause',width=650)
    #page title
    
    st.write("The association between air pollution and lung cancer has been well established for decades. The International Agency for Research on Cancer (IARC), the specialised cancer agency of the World Health Organization, classified outdoor air pollution as carcinogenic to humans in 2013, citing an increased risk of lung cancer from greater exposure to particulate matter and air pollution.")




    st.markdown(
    """
    The following list won't indent no matter what I try:
    - A 2012 study by Mumbai’s Tata Memorial Hospital found that 52.1 per cent of lung cancer patients had no history of smoking. 
    - The study contrasted this with a Singapore study that put the number of non-smoking lung cancer patients at 32.5 per cent, and another in the US that found the number to be about 10 per cent.
    - The Tata Memorial study found that 88 per cent of female lung cancer patients were non-smokers, compared with 41.8 per cent of males. It concluded that in the case of non-smokers, environmental and genetic factors were implicated.
    """
    )

    st.title("Not just a Delhi phenomenon ")
    stove = Image.open("images/stove.png")

    st.image(stove, caption='Smoking is not the major cause',width=650)
    #page title
    st.markdown(
    """
    The following list won't indent no matter what I try:
    - In January 2017, researchers at AIIMS, Bhubaneswar, published a demographic profile of lung cancer in eastern India, which found that 48 per cent of patients had not been exposed to active or passive smoking
    - 89 per cent of women patients had never smoked, while the figure for men was 28 per cent.
    - From available research, very little is understood about lung cancer among non-smokers in India. “We need more robust data to identify how strong is the risk and link,” Guleria of AIIMS says.
    """
    )

if (selection == 'About the Dataset'):
    tab1, tab2, tab3 , tab4 ,tab5= st.tabs(["Dataset analysis", "Training Data", "Test Data","Algorithms Used",'CNN Based Indentification'])

    with tab1:

        st.title("Lung Cancer Dataset")
        data=pd.read_csv("data.csv")
        st.write(data.head(10))
        # fig=plt.figure(figsize=(9,7))
        sns.set(rc={'figure.figsize':(8,8)})
        pl=sns.countplot(x ='Level', data = data , palette='rocket')
        st.pyplot(pl.figure)
       

    with tab2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
        
    with tab4:
        st.title("List of Algorithms Used")
        algo = Image.open("images/algo.png")

        st.image(algo, caption='ML Algorithms',width=700)

        st.write("Since this is a Mutlti-Class Classification we have used Algorithms which are maily used for Supervised Learning for the following Problem Statement ")

        st.markdown(
            """
            Supervised Learning Algorithms:
            - Linear Regression
            - Support Vector Machine
            - K-Nearest Neighbours (KNN)
            - Decision Tree Classifier
            """
            )

    with tab5:
        st.title("Convolutional Neural Network Model")
    

# Lung Cancer disease Prediction pages
if (selection == 'Lung Cancer Prediction'):
    #page title

    st.title('Lung Cancer Prediction using ML')

    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age', key="1")
        
    with col2:
        Gender = st.text_input('Gender', key="2")
        
    with col3:
        AirPollution = st.text_input('Air Pollution', key="3")

    with col1:
        Alcoholuse = st.text_input('Alcohol Use', key="4")  

    with col2:
        BalancedDiet = st.text_input('Balanced Diet', key="5")
        
    with col3:
        Obesity = st.text_input('Obesity', key="6")
        
    with col1:
        Smoking = st.text_input('Smoking', key="7")
        
    with col2:
        PassiveSmoker = st.text_input('Passive Smoker', key="8")
        
    with col3:
        Fatigue = st.text_input('Fatigue', key="9")
        
    with col1:
        WeightLoss = st.text_input('Weight Loss', key="10")
        
    with col2:
        ShortnessofBreath = st.text_input('Shortness of Breath', key="11")
        
    with col3:
        Wheezing = st.text_input('Wheezing', key="12")
        
    with col1:
        SwallowingDifficulty = st.text_input('Swallowing Difficulty', key="13")
        
    with col2:
        ClubbingofFingerNails = st.text_input('Clubbing of Finger Nails', key="14")

    with col3:
        FrequentCold = st.text_input('Frequent Cold', key="15")
        
    with col1:
        DryCough = st.text_input('Dry Cough', key="16")    
     
    with col2:
        Snoring = st.text_input('Snoring  ', key="17")
 
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = cancer_model.predict([[Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss,ShortnessofBreath, Wheezing, SwallowingDifficulty,ClubbingofFingerNails, FrequentCold, DryCough, Snoring]])                          
        
        if (heart_prediction[0] == 'High'):
          heart_diagnosis = 'The person is having heart disease'
        elif(heart_prediction[0] == 'Medium'):
          heart_diagnosis = 'The person is chance of having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)



    



if (selection == 'CNN Based disease Prediction'):
  st.set_option('deprecation.showfileUploaderEncoding', False)
  @st.cache(allow_output_mutation=True)

  def loading_model():
    fp = "models/keras_model.h5"
    model_loader = load_model(fp)
    return model_loader

  cnn = loading_model()
  st.write("""
  # Lung Cancer Detection using CNN and CT-Scan Images
  by Vedant :)
  """)



  temp = st.file_uploader("Upload CT-Scan Image",type=['png','jpeg','jpg'])
  if temp is not None:
      file_details = {"FileName":temp.name,"FileType":temp.type,"FileSize":temp.size}
      st.write(file_details)
  #temp = temp.decode()

  buffer = temp
  temp_file = NamedTemporaryFile(delete=False)
  if buffer:
      temp_file.write(buffer.getvalue())
      st.write(image.load_img(temp_file.name))


  if buffer is None:
    st.text("Oops! that doesn't look like an image. Try again.")

  else:

  

    ved_img = image.load_img(temp_file.name, target_size=(224, 224))

    # Preprocessing the image
    pp_ved_img = image.img_to_array(ved_img)
    pp_ved_img = pp_ved_img/255
    pp_ved_img = np.expand_dims(pp_ved_img, axis=0)

    #predict
    hardik_preds= cnn.predict(pp_ved_img)
    print(hardik_preds[0])

    if hardik_preds[0][0]>= 0.5:
      out = ('I am {:.2%} percent confirmed that this is a Normal Case'.format(hardik_preds[0][0]))
      st.balloons()
    
    else: 
      out = ('I am {:.2%} percent confirmed that this is a Lung Cancer Case'.format(1-hardik_preds[0][0]))

    st.success(out)
    
    image = Image.open(temp)
    st.image(image,use_column_width=True)
            
              

    

    
