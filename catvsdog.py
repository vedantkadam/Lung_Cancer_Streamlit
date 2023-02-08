
import pickle
import streamlit as st
import numpy as np
from PIL import Image 
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 
from streamlit_option_menu import option_menu

#Loading models
cancer_model = pickle.load(open('models/final_model.sav', 'rb'))

with st.sidebar:
    selection = option_menu('Lung Cancer detection System',
    ['Introduction',
    'Lung Cancer Prediction',
    'CNN Based disease Prediction'],
    icons = ['activity', 'heart', 'person'],
    default_index = 0)

#Introduction page
if (selection == 'Introduction'):
    #page title
    st.title('Introduction')


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
    
    else: 
      out = ('I am {:.2%} percent confirmed that this is a Lung Cancer Case'.format(1-hardik_preds[0][0]))

    st.success(out)
    
    image = Image.open(temp)
    st.image(image,use_column_width=True)
            
              

    

    
