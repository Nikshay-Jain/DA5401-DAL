model1 = Isomap(n_components=2)
model2 = TSNE(n_components=2, random_state = 42)


df = pd.read_csv('./visual-taxonomy/train.csv')
fragments = fragment_dataset(df, 'Category', ['Men Tshirts', 'Sarees', 'Kurtis', 'Women Tshirts', 'Women Tops & Tunics'])

################

MenTshirts_shortsleeves = []
MenTshirts_shortsleeves_id = get_random_rows_by_value(fragments['Men Tshirts'], 'attr_5', 'short sleeves', 100)

temp_list1 = MenTshirts_shortsleeves_id['id'].astype(int).tolist()
path_list1 = format_integer(temp_list1)
for pth in path_list1 :
  temp = get_pixel_array_cv2_grayscale(pth, 256)
  MenTshirts_shortsleeves.append(temp)

MenTshirts_shortsleeves_model1 = model1.fit_transform(MenTshirts_shortsleeves)

MenTshirts_shortsleeves_array = np.array(MenTshirts_shortsleeves)
MenTshirts_shortsleeves_model2 = model2.fit_transform(MenTshirts_shortsleeves_array)