
import streamlit as st
import numpy as np
from PIL import Image 
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "catvsdog.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
# Cats vd Dog
by vedant :)
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
    out = ('I am  percent confirmed that this is a cat case')
  
  else: 
    out = ('I am percent confirmed that this is a dog case')

  st.success(out)
  
  image = Image.open(temp)
  st.image(image,use_column_width=True)
          
            

  

  
