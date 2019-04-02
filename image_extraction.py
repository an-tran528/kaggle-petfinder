# Extract images features
# Idea from https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn 
# with some modifications
# If using more than 1 image, we use PCA to get the most important features

# 4. IMAGE PROCESSING 

# 4.a) PREPROCESSING (RESIZE, LOAD IMAGE)
pca = PCA(n_components=1,random_state=1234)
img_size = 256
train_ids = train['PetID'].values
test_ids = test['PetID'].values


def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path):
    image = cv2.imread(path)
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# 4,b) BUILD MODEL (DENSENET 121)
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, 
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)

########################################################################################################################
# 4,c) IMAGE FEATURE EXTRACTION

max_using_images = 5
def extract_img_features(mode,files, pet_ids, set_tmp):
    features_h = {}
    features_a = {}

    for pet_id in tqdm(pet_ids):
        photoAmt = int(set_tmp.loc[pet_id, 'PhotoAmt'])
        if photoAmt == 0:
            dim = 1
            batch_images_m = np.zeros((1, img_size, img_size, 3))
        else:
            dim = min(photoAmt, max_using_images)
            batch_images_m = np.zeros((dim, img_size, img_size, 3))
            try:
                urls = files[pet_id]
                for i, u in enumerate(urls[:dim]):
                    try:
                        batch_images_m[i] = load_image(u)
                    except:
                        pass
            except:
                pass

        batch_preds_m = m.predict(batch_images_m)
        pred = pca.fit_transform(batch_preds_m.T)
        features_a[pet_id] = pred.reshape(-1)
        features_h[pet_id] = batch_preds_m[0]
        
    feats_h = pd.DataFrame.from_dict(features_h, orient='index')
    feats_a = pd.DataFrame.from_dict(features_a, orient='index')
    
    return feats_h,feats_a
