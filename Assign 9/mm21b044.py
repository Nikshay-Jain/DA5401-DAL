#!/usr/bin/env python
# coding: utf-8

# ## Nikshay Jain | MM21B044
# ### Assign 9

# In[1]:


import numpy as np
import pandas as pd
from PIL import Image
import random, zipfile, cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import Isomap, TSNE


# ### Extract & read data using pandas

# In[2]:


train_df = pd.read_csv('./visual-taxonomy/train.csv')
test_df = pd.read_csv('./visual-taxonomy/test.csv')
samp_sub = pd.read_csv('./visual-taxonomy/sample_submission.csv')
cat_attr = pd.read_parquet('./visual-taxonomy/category_attributes.parquet')


# In[3]:


train_df


# In[4]:


test_df


# In[5]:


samp_sub


# #### List down all attributes

# In[6]:


cat_attr


# In[7]:


for _ in range(5):
    print(f"{cat_attr['Category'][_]}: {cat_attr['Attribute_list'][_]} \n")


# # Task 1:

# ### Datapreprocessing

# In[8]:


def fragment_dataset(df, column_name, values):
    if df.empty:
        return {}
    try:
        fragments = {}
        for value in values:
            fragments[value] = df[df[column_name] == value]
        return fragments
    except KeyError:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}


# In[9]:


def get_random_rows_by_value(df, column_name, value, num_rows):

    if not isinstance(df, pd.DataFrame):
        print("Error: Input 'df' must be a pandas DataFrame.")
        return None
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return None
    if not isinstance(num_rows, int) or num_rows <= 0 :
        print("Error: 'num_rows' must be a positive integer.")
        return None

    filtered_df = df[df[column_name] == value]
    if filtered_df.empty:
        print(f"No rows found with '{column_name}' equal to '{value}'.")
        return None

    num_rows = min(num_rows, len(filtered_df))  # Ensure we don't try to sample more rows than exist
    random_indices = random.sample(range(len(filtered_df)), num_rows)
    random_rows_df = filtered_df.iloc[random_indices]
    return random_rows_df


# In[10]:


def get_pixel_array_cv2_grayscale(image_path, num_pixels):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale mode
        if img is None:
            print(f"Error: Could not open or read image at '{image_path}'")
            return None

        height, width = img.shape  # Get image dimensions (height and width)
        total_pixels = height * width

        num_pixels = min(num_pixels, total_pixels)  # Ensure num_pixels does not exceed image size

        # Generate random pixel indices
        pixel_indices = np.random.choice(total_pixels, num_pixels, replace=False)

        pixel_array = []
        for index in pixel_indices:
            row = index // width
            col = index % width
            pixel_array.append(img[row, col])

        return np.array(pixel_array)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# In[11]:


def format_integer(integer_list):
    if not isinstance(integer_list, list):
        print("Error: Input must be a list.")
        return []

    formatted_strings = []
    for item in integer_list:
        if not isinstance(item, int):
            print("Error: All elements in the list must be integers.")
            return []
        formatted_strings.append(f"./visual-taxonomy/train_images/{item:06d}.jpg")
    return formatted_strings


# In[12]:


model1 = Isomap(n_components=2)
model2 = TSNE(n_components=2, random_state = 42)

fragments = fragment_dataset(train_df, 'Category', ['Men Tshirts', 'Sarees', 'Kurtis', 'Women Tshirts', 'Women Tops & Tunics'])
fragments['Men Tshirts']


# # Task 2 and 3:

# In[13]:


# Function to create image annotations with larger images
def place_images(ax, x, y, img_path, target_size=(48, 48)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img)
    ab = AnnotationBbox(OffsetImage(img), (x, y), frameon=False, pad=0.0)
    ax.add_artist(ab)


# In[14]:


def execute(df, attr, value, n=100):
    l = []
    lid = get_random_rows_by_value(df, attr, value, n)

    temp_list = lid['id'].astype(int).tolist()
    path_list = format_integer(temp_list)
    for pth in path_list:
        temp = get_pixel_array_cv2_grayscale(pth, 256)
        l.append(temp)

    l_model1 = model1.fit_transform(l)

    l_arr = np.array(l)
    l_model2 = model2.fit_transform(l_arr)
    
    image_paths = path_list

    # Plot for Model 1 (Isomap)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(l_model1[:, 0], l_model1[:, 1])
    ax1.set_title('Model 1 - Isomap')

    # Add images to the first plot
    for i in range(len(image_paths)):
        img_path = image_paths[i]
        place_images(ax1, l_model1[i, 0], l_model1[i, 1], img_path)

    plt.show()

    # Plot for Model 2 (t-SNE)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(l_model2[:, 0], l_model2[:, 1])
    ax2.set_title('Model 2 - t-SNE')

    # Add images to the second plot
    for i in range(len(image_paths)):
        img_path = image_paths[i]
        place_images(ax2, l_model2[i, 0], l_model2[i, 1], img_path)

    plt.show()


# ### Execute the functions for:
# 
# Category: 'Women Tops & Tunics',
# 
# Attribute: occasion
# 
# Value: casual

# In[15]:


execute(fragments['Women Tops & Tunics'], 'attr_5', 'casual', 100)


# ##### Isomap: 
# going from left to right the focus on clothes increase in the image rather than the full model. Also the images consists of darker shades rightwards than left.
# 
# ##### tSNE: 
# going towards top right the shades of clothes become darker. The pics at the top have broad neck and those at the bottom right have narrow neck.

# ### Execute the functions for:
# 
# Category: 'Women Tops & Tunics',
# 
# Attribute: sleeve_styling
# 
# Value: regular sleeves

# In[219]:


execute(fragments['Women Tops & Tunics'], 'attr_9', 'regular sleeves', 100)


# ##### Isomap: 
# going radially outwards no of models increase - the centre has just tops and tunics without model and pictures outwards have >=1 model. 
# 
# ##### tSNE: 
# going towards right the neck size becomes broader, while going down the colour becomes lighter.

# ### Execute the functions for:
# 
# Category: 'Women Tshirts',
# 
# Attribute: occasion
# 
# Value: typography

# In[220]:


execute(fragments['Women Tshirts'], 'attr_5', 'typography', 100)


# ##### Isomap: 
# The images become darker rightwards.
# 
# ##### tSNE: 
# Top desings are rigourous than bottom ones. The images become darker rightwards.

# ### Execute the functions for:
# 
# Category: 'Women Tshirts',
# 
# Attribute: print or pattern type
# 
# Value: regular sleeves

# In[222]:


execute(fragments['Women Tshirts'], 'attr_7', 'regular sleeves', 100)


# ##### Isomap: 
# The left pics have black/white colours predominantly while right pics have more brighter colours.
# 
# ##### tSNE: 
# The top right pics have light tshirts with simple typographic designs while bottom ones had dark with complicated designs.

# ### Execute the functions for:
# 
# Category: 'Kurtis',
# 
# Attribute: length
# 
# Value: calf length

# In[223]:


execute(fragments['Kurtis'], 'attr_3', 'calf length', 100)


# ##### Isomap: 
# Top pics have multiple models in a pic while bottom ones have single models.
# 
# ##### tSNE: 
# No of kurtis increase per image rightwards while left ones have narrow neck. Central ones have bright colours while those at boundaries have darker ones.

# ### Execute the functions for:
# 
# Category: 'Kurtis',
# 
# Attribute: sleeve length
# 
# Value: regular

# In[224]:


execute(fragments['Kurtis'], 'attr_9', 'regular', 100)


# ##### Isomap: 
# Dark backgrounds towards right while lighter ones towards left.
# 
# ##### tSNE: 
# Peripheral pics have multiple models in a pic with dark colour kurtis while central ones have single models with bright colours.

# ### Execute the functions for:
# 
# Category: 'Sarees',
# 
# Attribute: border_width
# 
# Value: small border

# In[225]:


execute(fragments['Sarees'], 'attr_3', 'small border', 100)


# ##### Isomap: 
# Plain white saaris at centre with no model, while designer saaris at the peripheries.
# 
# ##### tSNE: 
# A few centra pics have coloured saaris and rest are simplee white ones.

# ### Execute the functions for:
# 
# Category: 'Sarees',
# 
# Attribute: occasion
# 
# Value: party

# In[226]:


execute(fragments['Sarees'], 'attr_5', 'party', 100)


# ##### Isomap: 
# Leftwards bright colours increases while right ones are simple designs with white saaris.
# 
# ##### tSNE: 
# Left models with bright colour saaris while right cluster of plain simple saaris.

# ### Execute the functions for:
# 
# Category: 'Men Tshirts',
# 
# Attribute: neck
# 
# Value: polo

# In[227]:


execute(fragments['Men Tshirts'], 'attr_2', 'polo', 100)


# ##### Isomap: 
# Left ones have darker shades while right ones have lighter shades of colours. The left cluster has 4 men in a pic which is higher compared to others.
# 
# ##### tSNE: 
# Central models face sidewards while outer ones face straight to the camera view.

# ### Execute the functions for:
# 
# Category: 'Men Tshirts',
# 
# Attribute: sleeve_length
# 
# Value: short sleeves

# In[228]:


execute(fragments['Men Tshirts'], 'attr_5', 'short sleeves', 100)


# ##### Isomap: 
# Top-Right ones have darker shades while bottom-left ones have lighter shades of colours. Topp neck is round while bottom is Polo.
# 
# ##### tSNE: 
# Top tshirts have polo neck with lighter colours while bottom right have dark shades of round neck t shirts.

# In[ ]:




