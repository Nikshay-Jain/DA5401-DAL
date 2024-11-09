import numpy as np
import pandas as pd
from PIL import Image
import random, zipfile, cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import Isomap, TSNE

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

################

model1 = Isomap(n_components=2)
model2 = TSNE(n_components=2, random_state = 42)


df = pd.read_csv('./visual-taxonomy/train.csv')
fragments = fragment_dataset(df, 'Category', ['Men Tshirts', 'Sarees', 'Kurtis', 'Women Tshirts', 'Women Tops & Tunics'])
print(fragments)

# ################

# MenTshirts_shortsleeves = []
# MenTshirts_shortsleeves_id = get_random_rows_by_value(fragments['Men Tshirts'], 'attr_5', 'short sleeves', 100)

# temp_list1 = MenTshirts_shortsleeves_id['id'].astype(int).tolist()
# path_list1 = format_integer(temp_list1)
# for pth in path_list1 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   MenTshirts_shortsleeves.append(temp)

# MenTshirts_shortsleeves_model1 = model1.fit_transform(MenTshirts_shortsleeves)

# MenTshirts_shortsleeves_array = np.array(MenTshirts_shortsleeves)
# MenTshirts_shortsleeves_model2 = model2.fit_transform(MenTshirts_shortsleeves_array)

# ################

# MenTshirts_polo = []
# MenTshirts_polo_id = get_random_rows_by_value(fragments['Men Tshirts'], 'attr_2', 'polo', 100)

# temp_list2 = MenTshirts_polo_id['id'].astype(int).tolist()
# path_list2 = format_integer(temp_list2)
# for pth in path_list2 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   MenTshirts_polo.append(temp)

# MenTshirts_polo_model1 = model1.fit_transform(MenTshirts_polo)

# MenTshirts_polo_array = np.array(MenTshirts_polo)
# MenTshirts_polo_model2 = model2.fit_transform(MenTshirts_polo_array)

# ################

# Sarees_party = []
# Sarees_party_id = get_random_rows_by_value(fragments['Sarees'], 'attr_5', 'party', 100)

# temp_list3 = Sarees_party_id['id'].astype(int).tolist()
# path_list3 = format_integer(temp_list3)
# for pth in path_list3 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   Sarees_party.append(temp)

# Sarees_party_model1 = model1.fit_transform(Sarees_party)

# Sarees_party_array = np.array(Sarees_party)
# Sarees_party_model2 = model2.fit_transform(Sarees_party_array)

# ################

# Sarees_smallborder = []
# Sarees_smallborder_id = get_random_rows_by_value(fragments['Sarees'], 'attr_3', 'small border', 100)

# temp_list4 = Sarees_smallborder_id['id'].astype(int).tolist()
# path_list4 = format_integer(temp_list4)
# for pth in path_list4 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   Sarees_smallborder.append(temp)

# Sarees_smallborder_model1 = model1.fit_transform(Sarees_smallborder)

# Sarees_smallborder_array = np.array(Sarees_smallborder)
# Sarees_smallborder_model2 = model2.fit_transform(Sarees_smallborder_array)

# ################

# Kurtis_regular = []
# Kurtis_regular_id = get_random_rows_by_value(fragments['Kurtis'], 'attr_9', 'regular', 100)

# temp_list5 = Kurtis_regular_id['id'].astype(int).tolist()
# path_list5 = format_integer(temp_list5)
# for pth in path_list5 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   Kurtis_regular.append(temp)

# Kurtis_regular_model1 = model1.fit_transform(Kurtis_regular)

# Kurtis_regular_array = np.array(Kurtis_regular)
# Kurtis_regular_model2 = model2.fit_transform(Kurtis_regular_array)

# ################

# Kurtis_calflength = []
# Kurtis_calflength_id = get_random_rows_by_value(fragments['Kurtis'], 'attr_3', 'calf length', 100)

# temp_list6 = Kurtis_calflength_id['id'].astype(int).tolist()
# path_list6 = format_integer(temp_list6)
# for pth in path_list6 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   Kurtis_calflength.append(temp)

# Kurtis_calflength_model1 = model1.fit_transform(Kurtis_calflength)

# Kurtis_calflength_array = np.array(Kurtis_calflength)
# Kurtis_calflength_model2 = model2.fit_transform(Kurtis_calflength_array)

# ################

# WomenTshirts_regularsleeves = []
# WomenTshirts_regularsleeves_id = get_random_rows_by_value(fragments['Women Tshirts'], 'attr_7', 'regular sleeves', 100)

# temp_list7 = WomenTshirts_regularsleeves_id['id'].astype(int).tolist()
# path_list7 = format_integer(temp_list7)
# for pth in path_list7 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   WomenTshirts_regularsleeves.append(temp)

# WomenTshirts_regularsleeves_model1 = model1.fit_transform(WomenTshirts_regularsleeves)

# WomenTshirts_regularsleeves_array = np.array(WomenTshirts_regularsleeves)
# WomenTshirts_regularsleeves_model2 = model2.fit_transform(WomenTshirts_regularsleeves_array)

# ################

# WomenTshirts_typography = []
# WomenTshirts_typography_id = get_random_rows_by_value(fragments['Women Tshirts'], 'attr_5', 'typography', 100)

# temp_list8 = WomenTshirts_typography_id['id'].astype(int).tolist()
# path_list8 = format_integer(temp_list8)
# for pth in path_list8 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   WomenTshirts_typography.append(temp)

# WomenTshirts_typography_model1 = model1.fit_transform(WomenTshirts_typography)

# WomenTshirts_typography_array = np.array(WomenTshirts_typography)
# WomenTshirts_typography_model2 = model2.fit_transform(WomenTshirts_typography_array)

# ################

# WomenTopsandTunics_regularsleeves = []
# WomenTopsandTunics_regularsleeves_id = get_random_rows_by_value(fragments['Women Tops & Tunics'], 'attr_9', 'regular sleeves', 100)

# temp_list9 = WomenTopsandTunics_regularsleeves_id['id'].astype(int).tolist()
# path_list9 = format_integer(temp_list9)
# for pth in path_list9 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   WomenTopsandTunics_regularsleeves.append(temp)

# WomenTopsandTunics_regularsleeves_model1 = model1.fit_transform(WomenTopsandTunics_regularsleeves)

# WomenTopsandTunics_regularsleeves_array = np.array(WomenTopsandTunics_regularsleeves)
# WomenTopsandTunics_regularsleeves_model2 = model2.fit_transform(WomenTopsandTunics_regularsleeves_array)

# ################

# WomenTopsandTunics_casual = []
# WomenTopsandTunics_casual_id = get_random_rows_by_value(fragments['Women Tops & Tunics'], 'attr_5', 'casual', 100)

# temp_list10 = WomenTopsandTunics_casual_id['id'].astype(int).tolist()
# path_list10 = format_integer(temp_list10)
# for pth in path_list10 :
#   temp = get_pixel_array_cv2_grayscale(pth, 256)
#   WomenTopsandTunics_casual.append(temp)

# WomenTopsandTunics_casual_model1 = model1.fit_transform(WomenTopsandTunics_casual)

# WomenTopsandTunics_casual_array = np.array(WomenTopsandTunics_casual)
# WomenTopsandTunics_casual_model2 = model2.fit_transform(WomenTopsandTunics_casual_array)

# ################


# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from PIL import Image


# image_paths = path_list10

# # Function to create image annotations
# def add_image(ax, x, y, img_path,  target_size=(32, 32)):
#     img = Image.open(img_path).convert('RGB')
#     img = img.resize(target_size)
#     img = np.array(img)
#     ab = AnnotationBbox(OffsetImage(img), (x, y), frameon=False, pad=0.0)
#     ax.add_artist(ab)

# # Create the scatter plot
# fig, ax = plt.subplots()
# ax.scatter(WomenTopsandTunics_casual_model2[:, 0], WomenTopsandTunics_casual_model2[:, 1])

# # Add images to the plot
# for i in range(len(image_paths)):
#     img_path = image_paths[i]
#     add_image(ax, WomenTopsandTunics_casual_model2[i, 0], WomenTopsandTunics_casual_model2[i, 1], img_path)

# plt.show()

def execute(df, attr, value, n=100):
    l = []
    lid = get_random_rows_by_value(df, attr, value, n)

    temp_list = lid['id'].astype(int).tolist()
    path_list = format_integer(temp_list)
    for pth in path_list :
        temp = get_pixel_array_cv2_grayscale(pth, 256)
        l.append(temp)

    l_model1 = model1.fit_transform(l)

    l_arr = np.array(l)
    l_model2 = model2.fit_transform(l_arr)
    
    image_paths = path_list

    # Function to create image annotations
    def add_image(ax, x, y, img_path,  target_size=(32, 32)):
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img = np.array(img)
        ab = AnnotationBbox(OffsetImage(img), (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)

    # Create the scatter plot
    fig, ax = plt.subplots()
    ax.scatter(l_model2[:, 0], l_model2[:, 1])

    # Add images to the plot
    for i in range(len(image_paths)):
        img_path = image_paths[i]
        add_image(ax, l_model2[i, 0], l_model2[i, 1], img_path)

    plt.show()

execute(fragments['Women Tops & Tunics'], 'attr_5', 'casual', 100)