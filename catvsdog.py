
import streamlit as st
import numpy as np
from PIL import Image 
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

from streamlit_option_menu import option_menu

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
    



if (selection == 'CNN Based disease Prediction'):
  st.set_option('deprecation.showfileUploaderEncoding', False)
  @st.cache(allow_output_mutation=True)

  def loading_model():
    fp = "catvsdog.h5"
    model_loader = load_model(fp)
    return model_loader

  cnn = loading_model()
  st.write("""
  # Cats vs Dog
  by Vedant :)
  """)



  temp = st.file_uploader("Upload X-Ray Image",type=['png','jpeg','jpg'])
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
      out = ('I am {:.2%} percent confirmed that this is a Cat Image'.format(hardik_preds[0][0]))
    
    else: 
      out = ('I am {:.2%} percent confirmed that this is a Dog Image'.format(1-hardik_preds[0][0]))

    st.success(out)
    
    image = Image.open(temp)
    st.image(image,use_column_width=True)
            
              

    

    
