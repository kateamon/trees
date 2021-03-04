'''  Lovely Trees Application by Team HAKA
Does Image Classification using a variety of algorithms, plus Neural Style Transfer.
Explainable AI (XAI) will be added to the Classification Options.
'''
import numpy as np
import pandas as pd
# Preprocessing data
from sklearn.model_selection import train_test_split     # data-splitter
from sklearn.preprocessing import StandardScaler         # data-normalization
from sklearn.preprocessing import PolynomialFeatures     # for polynomials
from sklearn.pipeline import make_pipeline               # for pipelines
np.random.seed (42)
# Modeling and Metrics
#
# --For Classifier
from sklearn.linear_model import LogisticRegression      # Classifier
from sklearn.metrics import confusion_matrix             # confusion matrix
from sklearn.metrics import classification_report        # goodness of fit report


# --For Neural Network Classifiers
import torch
torch.cuda.is_available()
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Now the Graphical libraries imports and settings
import matplotlib.pyplot as plt                          # for plotting
import seaborn as sns                                    # nicer looking plots
from matplotlib import colors
import warnings
warnings.filterwarnings ('ignore')  # suppress warning
from transformers import pipeline

import streamlit as st
import os
import time
from PIL import Image
import urllib.request

import json

from model_functions import predict, load_checkpoint
from processing_functions import process_image

# For Neural Style Transfer
from style_functions import load_image
from run_vgg19_style_transfer import run_vgg19_style_transfer



def main():
    # Using "radio" option instead of "selectbox"
    page = st.sidebar.radio("Choose Page", ["Homepage", "About", "Classify", "Neural Style Transfer"])
    if page == "Homepage":
        #content_image = homepage()
        homepage()
    elif page == "About":
        about()
    elif page == "Classify":
        # classify()
        classify_not_ready()
    elif page == "Neural Style Transfer":
        style_transfer()

def homepage():
    st.title('Lovely Trees Application')
    st.header('Image Classification & Neural Style Transformation')
    st.header('The user provided image will be classified as a willow tree, pepper tree, or neither.')

    app_logo = Image.open(r'LovelyTrees.jpg')
    # st.image(logo, caption='Project done by SupportVectors Team HAKA Project', width = 300, )
    st.image(app_logo, caption='Project done by SupportVectors Team HAKA', use_column_width = True )
    st.header('Neural Style Transfer applies the style of one image to the content of another image.')
    st.subheader('Note on the left side above the original images of a willow tree and a pepper tree. The right side shows the result of neural style transfer.	For the willow tree (top), the style of a Pablo Picasso painting was applied. For the pepper tree (bottom), the style of a Jackson Pollack painting was applied.')

def about():
    st.title('About Team HAKA')
    team_logo = Image.open(r'team-HAKA-logo.jpg')
    # st.image(team_logo, caption='Team HAKA', use_column_width = True, )
    st.image(team_logo, use_column_width = True, )
    st.header('Team HAKA is a four member project team at Support Vectors')
    st.write('The name of the team is derived from the first name initials of the team members:')
    st.write ('Harini Datla, Abhishek Saxena, Kate Amon, and Aritra Dasgupta.')
    st.write('Our team also likes the fierce spirit of the haka, a pre-battle dance done by Maori warriors of New Zealand.')
    st.write('In 2020 the team members met (virtually) in the ML400 Deep Learning workshop at SupportVectors located in Fremont, California, run by head instructor and guide Asif Qamar.')

    # Add more about SupportVectors too

def classify_not_ready():
    st.title('Classifier models too large for deployment using streamlit sharing.')
    st.header('Please play with the Neural Transfer Styling in the meantime.')


def classify():
    image = None

    #----------------- Uploading User Tree image from URL or File -------------------

    st.title('Please provide an image - will be identified as pepper tree, willow tree, or neither.')

    #----------------- Side Bar Options for User Image ------------------------------
    st.title("Upload Options")
    input_method = st.radio("Options", ('File Upload', 'URL'))

    flag=0
    if input_method == 'File Upload' :
        user_upload_image = st.file_uploader("Upload a picture of Tree",type=['png','jpeg','jpg'])
        if user_upload_image is not None:
            file_details = {"FileName":user_upload_image.name,"FileType":user_upload_image.type,"FileSize":user_upload_image.size}
			# st.write(file_details)
            flag=1
        if flag == 1 :
            image_source = user_upload_image.name
            image = Image.open(user_upload_image)
            st.image(image, caption= user_upload_image.name+'  '+user_upload_image.type+'  '+str(user_upload_image.size)+' bytes', width =300, use_column_width=True, )

	# Image from URL
    if input_method == 'URL' :
        image_url = st.text_area("Enter the complete Url", key="user_url_choice")
        image_url_status = st.button('Upload')

        if image_url_status :
            image_source = image_url
            image = Image.open(urllib.request.urlopen(image_url))
            st.image(image,
            caption= str(image), width = 300, use_column_width=True, )
        else:
            st.warning('click on upload')


    #----------------------  Choosing Classification Method ---------------------------
    st.title('Choose the model for Analysis')
    model_selected = st.radio("Options", ['Pre Trained Model'])

    # model_selected = st.sidebar.radio(
    # 	"Options",
    # 	('Pre Trained Model', 'CNN Model', 'FFN Model', 'Random Forest', 'Logistic Regression', 'Ensemble'),0)

    if model_selected == 'Pre Trained Model':
        model_selected2 = st.selectbox("Choose the Pretrained Model",['VGG16'])

        # model_selected2 = st.sidebar.selectbox("Choose the Pretrained Model",
        # 	['VGG16','PTM2','PTM2','PTM2','PTM2','PTM2','PTM2','PTM2'])

    if model_selected2 == 'VGG16' and image != None :
        # note that the predict function returns top_probabilities, top_classes
        model = load_checkpoint('/home/kate/data_and_models/ai_models/vgg16-tree-3class-model.pth')
        probs, classes = predict(image, model)

        st.title('Tree Classification Results')

        with open('tree_to_name.json', 'r') as f:
            tree_to_name = json.load(f)
        tree_names = [tree_to_name[i] for i in classes]
        chart_data = pd.DataFrame(data = [probs], columns = tree_names)

        # st.write("chart_data type ", type(chart_data))

        if (chart_data["willow tree"][0]) > 0.5:
            tree_detected = "Willow Tree"
        elif (chart_data["pepper tree"][0]) > 0.5:
            tree_detected = "Pepper Tree"
        else:
            tree_detected = "Not a Pepper Tree or a Willow Tree"

        st.write('The image is: ', tree_detected)
        st.write('Percentage confidence in the image identification ', chart_data)

        st.bar_chart(chart_data)

# ------------------ For Neural Style Transfer -----------------------------------------
def style_transfer(image=None):
	#image = None
	got_style_image = False

	#----------------- Uploading User Tree image from URL or File -------------------

	# Image from upload

	st.title('Please provide an image which will be the content image for Neural Style Transfer.')

	st.title("Upload Options")
	input_method = st.radio(
	"Options",
	('File Upload', 'URL'))

	flag=0
	if input_method == 'File Upload' :
		user_upload_image = st.file_uploader("Upload an image",type=['png','jpeg','jpg'])
		if user_upload_image is not None:
			file_details = {"FileName":user_upload_image.name,"FileType":user_upload_image.type,"FileSize":user_upload_image.size}
			# st.write(file_details)
			flag=1
		if flag == 1 :
	#		from PIL import Image
			image_source = user_upload_image.name
			image = Image.open(user_upload_image)
			st.image(image, caption= user_upload_image.name+'  '+user_upload_image.type+'  '+str(user_upload_image.size)+' bytes', width =300, use_column_width=True, )

	# Image from URL
	if input_method == 'URL' :
		image_url = st.text_area("Enter the complete Url", key="user_url_choice")
		image_url_status = st.button('Upload')

		if image_url_status :
			image_source = image_url
			image = Image.open(urllib.request.urlopen(image_url))
			st.image(image,
			caption= str(image), width = 300, use_column_width=True, )
		else:
			st.warning('click on upload')

	st.title('Neural Style Transfer - See the style choices below, select from menu.')
	# Google drawing saved as jpg file of the five style images
	five_artists = Image.open(r'Five_Artists.jpg')
	st.image(five_artists, use_column_width = True )

	st.title("Neural Style Transformation")

	style_selection = st.radio("Style Options",
	('No Transformation', 'Artist 1: Raja Ravi Varma', 'Artist 2: Jackson Pollack', 'Artist 3: Katsushika Hokusai', 'Artist 4: Cheri Samba', 'Artist 5: Frida Kahlo', 'Upload Your Own Style Image'),0)

	got_style_image = False

	if style_selection != 'No Transformation':

		if style_selection == 'Artist 1: Raja Ravi Varma':
			style_image = Image.open('raja-ravi_woman-in-garden.jpeg')
			style_source = "Artist Raja Ravi Varma"
			got_style_image = True
		elif style_selection == 'Artist 2: Jackson Pollack':
			style_image = Image.open('jackson-pollack_style.png')
			style_source = "Artist Jackson Pollack"
			got_style_image = True
		elif style_selection == 'Artist 3: Katsushika Hokusai':
			style_image = Image.open('Katsushika-Hokusai_Great-Wave.jpeg')
			style_source = "Artist Katsushika Hokusai"
			got_style_image = True
		elif style_selection == 'Artist 4: Cheri Samba':
			style_image = Image.open('Cheri-Samba_J-aime-la-couleur.jpeg')
			style_source = "Artist Cheri Samba"
			got_style_image = True
		elif style_selection == 'Artist 5: Frida Kahlo':
			style_image = Image.open('frida-kahlo_style.png')
			style_source = "Artist Frida Kahlo"
			got_style_image = True
		elif style_selection == 'Upload Your Own Style Image':
			st.title("Style Upload Options")
			style_input_method = st.radio(
			"Choose One",
			('Style File Upload', 'Style URL'))

			got_style_image = False
			got_image = False

			# Style image from upload

			if style_input_method == 'Style File Upload' :
				# first clear any previous uploaded style file
				got_style_image = False
				style_upload_image = None
				flag2 = 0
				style_upload_image = st.file_uploader("Upload an image for style",type=['png','jpeg','jpg'])
				if style_upload_image is not None:
					file_details = {"FileName":style_upload_image.name,"FileType":style_upload_image.type,"FileSize":style_upload_image.size}
					flag2 = 1
				if flag2 == 1:
			#		from PIL import Image
					style_image = Image.open(style_upload_image)
					style_source = style_upload_image.name
					got_style_image = True
					# st.write('For debugging: got uploaded image')

			# Image from URL
			elif style_input_method == 'Style URL' :
				# first clear any previous URL style file
				got_style_image = False
				style_image_url = None

				style_image_url = st.text_area("Enter the complete Url", key="style_url_choice")
				style_image_url_status = st.button('Enter Style URL')

				if style_image_url_status :
					style_image = Image.open(urllib.request.urlopen(style_image_url))
					style_source = str(style_image_url)
					got_style_image = True
					#st.image(image, caption= str(image), width = 300, use_column_width=True, )
					# st.write('For debugging: got URL image')
				else:
					st.warning('click on Enter Style URL')

		# if type(style_image) == 'NoneType':
		# 	got_style_image = False

		ready_to_run_style = (style_selection != 'No Transformation') and got_style_image

		if ready_to_run_style:
			st.title('***Neural Style Transfer is beginning***')

			st.write('Please be patient, the transformation takes four or five minutes.')
			#st.write('For debugging: got_style_image is ', got_style_image)
			#st.write('For debugging: style_image type is ', type(style_image))
			content_image = image

			image_w_style = run_vgg19_style_transfer(image, style_image)

			# Captions for the input photo, style photo, and resulting output photo.
			# content_image_caption = 'Your photo: ' + user_upload_image.name # breaks for URL image
			content_image_caption = 'Your photo: ' + image_source
			style_image_caption = 'Art image by ' + style_source
			output_image_caption = 'Image with style applied from ' + style_source

			st.image(content_image, caption= content_image_caption,  width = 200)
			st.image(style_image, caption= style_image_caption,  width = 200)

			st.image(image_w_style, caption=output_image_caption,  width = 400)
			st.write('Right click on or touch output image and choose "Save Image As" to save a copy.')

			# Clear image settings before style transfer runs again
			style_image = None
			got_image = False
			got_style_image = False
			style_upload_image = None

if __name__ == '__main__':
    main()
