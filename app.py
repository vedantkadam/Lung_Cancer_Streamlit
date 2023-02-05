import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as plt
import seaborn as sns
from streamlit_option_menu import option_menu


st.title("hello world")

data=pd.read_csv("data.csv")

st.write(data)
# fig=plt.figure(figsize=(9,7))
sns.set(rc={'figure.figsize':(19,15)})
pl=sns.countplot(x ='Gender', data = data , palette='rocket')
st.pyplot(pl.figure)


pl1=sns.countplot(x='Level', data=data, palette='rocket')
st.pyplot(pl1.figure)