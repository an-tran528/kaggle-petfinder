# Our team's kernel version for final submission

import glob
import json
import string
import re
import numpy as np
import pandas as pd
import Levenshtein as lv
from joblib import Parallel, delayed
from PIL import Image
import os
import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer
import fastText as ft

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF,PCA

import cv2
from tqdm import tqdm
from keras.applications.densenet import preprocess_input, DenseNet121

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

import scipy as sp
import tensorflow as tf
from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

########################################################################################################################

def seed_everything(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
seed_everything()
########################################################################################################################
#1 LOAD CORE DATA

train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')

labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
labels_color = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
labels_state = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

# extract ids and target
target = train['AdoptionSpeed']
train_id = train['PetID']
test_id = test['PetID']

########################################################################################################################
def sort_path(path,filename):
    rgx = re.compile(path)
    result = {}
    for url in filename:
        match = rgx.search(url)
        if match:
            key = match.group(1)
            if key not in result:
                result[key] = []
            result[key] += [url]
        else:
            print(f'This did not match: {url}')
    return result

train_img_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'),key = lambda x:int(x.split('-')[-1].split('.')[0]))
test_img_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'),key = lambda x:int(x.split('-')[-1].split('.')[0]))
train_img_path = r"\.\./input/petfinder-adoption-prediction/train_images/(.*)-\d+.jpg"
test_img_path = r"\.\./input/petfinder-adoption-prediction/test_images/(.*)-\d+.jpg"

train_images = sort_path(train_img_path,train_img_files)
test_images = sort_path(test_img_path,test_img_files)

########################################################################################################################

#2.a ADD VERTEX FEATURES
def add_vertex_features(train,test,train_images,test_images):
    
    train_id = train['PetID']
    test_id = test['PetID']

    vertex_xs_train = []
    vertex_ys_train = []

    for pet in train_id:
        try:
            im = Image.open(train_images[pet][0])
            width, height = im.size
            vertex_xs_train.append(width)
            vertex_ys_train.append(height)
        except:
            vertex_xs_train.append(-1.0)
            vertex_ys_train.append(-1.0)

    train.loc[:, 'vertex_x'] = vertex_xs_train
    train.loc[:, 'vertex_y'] = vertex_ys_train

    vertex_xs_test = []
    vertex_ys_test = []

    for pet in test_id:
        try:
            im = Image.open(test_images[pet][0])
            width, height = im.size
            vertex_xs_test.append(width)
            vertex_ys_test.append(height)
        except:
            vertex_xs_test.append(-1.0)
            vertex_ys_test.append(-1.0)

    test.loc[:, 'vertex_x'] = vertex_xs_test
    test.loc[:, 'vertex_y'] = vertex_ys_test

    print("Train shape {}, test shape {} after adding vertex features".format(train.shape, test.shape))
    return train, test

train,test = add_vertex_features(train,test,train_images,test_images)

########################################################################################################################
def getSize(filename):
    #filename = images_path + filename
    try:
        st = os.stat(filename)
        size = st.st_size
    except:
        size = -1
    return size

def getDimensions(filename):
    #filename = images_path + filename
    try:
        img_size = Image.open(filename).size
    except:
        image_size = -1
    return img_size 


def add_image_features(train,test):
    train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
    
    train_df_imgs = pd.DataFrame(train_image_files,columns = ['image_filename'])
    
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    
    test_df_imgs = pd.DataFrame(test_image_files,columns = ['image_filename'])
    test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

    train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
    train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
    train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x : x[0])
    train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x : x[1])
    train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)


    test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
    test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
    test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x : x[0])
    test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x : x[1])
    test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)

    aggs = {
    'image_size': ['mean', 'sum', 'var'],
    'width': ['mean', 'sum', 'var'],
    'height': ['mean', 'sum', 'var']}
    
    agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    agg_train_imgs.columns = new_columns
    agg_train_imgs = agg_train_imgs.reset_index()

    
    agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    agg_test_imgs.columns = new_columns
    agg_test_imgs = agg_test_imgs.reset_index()

    agg_imgs = pd.concat([agg_train_imgs,agg_test_imgs],axis=0)
    return agg_imgs

agg_imgs = add_image_features(train,test)
print("Finished creating aggregated image features")
########################################################################################################################
#2.b ADD SENTIMENT and META DATA FEATURES

class PetFinderParser(object):
    
    def __init__(self, debug=False):
        
        self.debug = debug
        self.sentence_sep = ' '
        
        # Does not have to be extracted because main DF already contains description
        self.extract_sentiment_text = True
        
    def open_metadata_file(self, filename):
        """
        Load metadata file.
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file
            
    def open_sentiment_file(self, filename):
        """
        Load sentiment file.
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file
            
    def open_image_file(self, filename):
        """
        Load image file.
        """
        image = np.asarray(Image.open(filename))
        return image
        
    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """
        
        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)

        if self.extract_sentiment_text:
            file_sentences_text = [x['text']['content'] for x in file['sentences']]
            file_sentences_text = self.sentence_sep.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]
        
        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()
        
        file_sentiment.update(file_sentences_sentiment)
        
        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
        if self.extract_sentiment_text:
            df_sentiment['text'] = file_sentences_text
            
        df_sentiment['entities'] = file_entities
        df_sentiment = df_sentiment.add_prefix('sentiment_')
        
        return df_sentiment
    
    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """
        
        file_keys = list(file.keys())
        
        if 'labelAnnotations' in file_keys:
            #file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.5)]
            file_annots = file['labelAnnotations'][:]
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']
        
        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
        
        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc)
        }
        
        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')
        
        return df_metadata

# Helper function for parallel data processing:
def extract_additional_features(pet_id, mode='train'):
    
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(glob.glob('../input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(mode, pet_id)))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_metadata_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]
    
    return dfs

########################################################################################################################
#2.c PARSING DATA

def pet_parsing(train, test):
    
    # Unique IDs from train and test:
    debug = False
    train_pet_ids = train.PetID.unique()
    test_pet_ids = test.PetID.unique()

    if debug:
        train_pet_ids = train_pet_ids[:10]
        test_pet_ids = test_pet_ids[:5]

    # Train set:
    # Parallel processing of data:
    dfs_train = Parallel(n_jobs=6, verbose=1)(
        delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

    # Extract processed data and format them as DFs:
    train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
    train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

    train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
    train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

    print(train_dfs_sentiment.shape, train_dfs_metadata.shape)

    # Test set:
    # Parallel processing of data:
    dfs_test = Parallel(n_jobs=6, verbose=1)(
        delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

    # Extract processed data and format them as DFs:
    test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
    test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

    test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
    test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

    print(test_dfs_sentiment.shape, test_dfs_metadata.shape)
    return train_dfs_sentiment,train_dfs_metadata,test_dfs_sentiment,test_dfs_metadata

pet_parser = PetFinderParser()
train_dfs_sentiment, train_dfs_metadata, test_dfs_sentiment, test_dfs_metadata = pet_parsing(train,test)

########################################################################################################################
train = train.merge(train_dfs_sentiment[['sentiment_text', 'PetID']], how='left', on='PetID')
test = test.merge(test_dfs_sentiment[['sentiment_text', 'PetID']], how='left', on='PetID')

train['sentiment_text'] = train['sentiment_text'].apply(lambda x: str(x).replace('.', ' '))
train['sentiment_text'] = train['sentiment_text'].apply(lambda x: str(x).replace(',', ' '))

test['sentiment_text'] = test['sentiment_text'].apply(lambda x: str(x).replace('.', ' '))
test['sentiment_text'] = test['sentiment_text'].apply(lambda x: str(x).replace(',', ' '))

# at this moment, just drop 
train_dfs_sentiment = train_dfs_sentiment.drop(['sentiment_text'], axis=1)
test_dfs_sentiment = test_dfs_sentiment.drop(['sentiment_text'], axis=1)

########################################################################################################################
# Extend aggregates and improve column naming
aggregates = ['mean', 'sum', 'median', 'min', 'max']

# Train
train_metadata_desc = train_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
train_metadata_desc = train_metadata_desc.reset_index()
train_metadata_desc[
    'metadata_annots_top_desc'] = train_metadata_desc[
    'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

prefix = 'metadata'
train_metadata_gr = train_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
for i in train_metadata_gr.columns:
    if 'PetID' not in i:
        train_metadata_gr[i] = train_metadata_gr[i].astype(float)
train_metadata_gr = train_metadata_gr.groupby(['PetID']).agg(aggregates)
train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in train_metadata_gr.columns.tolist()])
train_metadata_gr = train_metadata_gr.reset_index()


train_sentiment_desc = train_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
train_sentiment_desc = train_sentiment_desc.reset_index()
train_sentiment_desc[
    'sentiment_entities'] = train_sentiment_desc[
    'sentiment_entities'].apply(lambda x: ' '.join(x))

prefix = 'sentiment'
train_sentiment_gr = train_dfs_sentiment.drop(['sentiment_entities'], axis=1)
for i in train_sentiment_gr.columns:
    if 'PetID' not in i:
        train_sentiment_gr[i] = train_sentiment_gr[i].astype(float)
train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(aggregates)
train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in train_sentiment_gr.columns.tolist()])
train_sentiment_gr = train_sentiment_gr.reset_index()


# Test
test_metadata_desc = test_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
test_metadata_desc = test_metadata_desc.reset_index()
test_metadata_desc[
    'metadata_annots_top_desc'] = test_metadata_desc[
    'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

prefix = 'metadata'
test_metadata_gr = test_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
for i in test_metadata_gr.columns:
    if 'PetID' not in i:
        test_metadata_gr[i] = test_metadata_gr[i].astype(float)
test_metadata_gr = test_metadata_gr.groupby(['PetID']).agg(aggregates)
test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in test_metadata_gr.columns.tolist()])
test_metadata_gr = test_metadata_gr.reset_index()


test_sentiment_desc = test_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
test_sentiment_desc = test_sentiment_desc.reset_index()
test_sentiment_desc[
    'sentiment_entities'] = test_sentiment_desc[
    'sentiment_entities'].apply(lambda x: ' '.join(x))

prefix = 'sentiment'
test_sentiment_gr = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)
for i in test_sentiment_gr.columns:
    if 'PetID' not in i:
        test_sentiment_gr[i] = test_sentiment_gr[i].astype(float)
test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(aggregates)
test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in test_sentiment_gr.columns.tolist()])
test_sentiment_gr = test_sentiment_gr.reset_index()

########################################################################################################################

# Train merges:
train_proc = train.copy()
train_proc = train_proc.merge(
    train_sentiment_gr, how='left', on='PetID')
train_proc = train_proc.merge(
    train_metadata_gr, how='left', on='PetID')
train_proc = train_proc.merge(
    train_metadata_desc, how='left', on='PetID')
train_proc = train_proc.merge(
    train_sentiment_desc, how='left', on='PetID')

# Test merges:
test_proc = test.copy()
test_proc = test_proc.merge(
    test_sentiment_gr, how='left', on='PetID')
test_proc = test_proc.merge(
    test_metadata_gr, how='left', on='PetID')
test_proc = test_proc.merge(
    test_metadata_desc, how='left', on='PetID')
test_proc = test_proc.merge(
    test_sentiment_desc, how='left', on='PetID')

print("Train shape {}, test shape {} after adding sentiment and metadata features".format(train_proc.shape, test_proc.shape))
assert train_proc.shape[0] == train.shape[0]
assert test_proc.shape[0] == test.shape[0]


########################################################################################################################
#ADDING BREED FEATURES


train_breed_main = train_proc[['Breed1']].merge(
    labels_breed, how='left',
    left_on='Breed1', right_on='BreedID',
    suffixes=('', '_main_breed'))

train_breed_main = train_breed_main.iloc[:, 2:]
train_breed_main = train_breed_main.add_prefix('main_breed_')

train_breed_second = train_proc[['Breed2']].merge(
    labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',
    suffixes=('', '_second_breed'))

train_breed_second = train_breed_second.iloc[:, 2:]
train_breed_second = train_breed_second.add_prefix('second_breed_')


train_proc = pd.concat(
    [train_proc, train_breed_main, train_breed_second], axis=1)


test_breed_main = test_proc[['Breed1']].merge(
    labels_breed, how='left',
    left_on='Breed1', right_on='BreedID',
    suffixes=('', '_main_breed'))

test_breed_main = test_breed_main.iloc[:, 2:]
test_breed_main = test_breed_main.add_prefix('main_breed_')

test_breed_second = test_proc[['Breed2']].merge(
    labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',
    suffixes=('', '_second_breed'))

test_breed_second = test_breed_second.iloc[:, 2:]
test_breed_second = test_breed_second.add_prefix('second_breed_')

test_proc = pd.concat(
    [test_proc, test_breed_main, test_breed_second], axis=1)

print("Train shape {}, test shape {} after adding breed features".format(train_proc.shape, test_proc.shape))

########################################################################################################################
#ADDING SMOOTH MEAN ENCODING FEATURES for AN's dataset
#https://maxhalford.github.io/blog/target-encoding-done-the-right-way/

def mean_encoding(df, cols, target, alpha):
    global_mean = df[target].mean()
    for col in cols:
        df[col + '_mean'] = df[col].map(df.groupby(col)[target].count()) * df[col].map(df.groupby(col)[target].mean())
        df[col + '_mean'] += global_mean * alpha
        df[col + '_mean'] /= (df[col].map(df.groupby(col)[target].count()) + alpha)
    return df

def mean_encoding_single(train, test, cols, target, alpha):
    train = mean_encoding(train, cols, target, alpha)
    for col in cols:
        test[col+'_mean'] = test[col].map(train.groupby(col)[target].mean())
    return train, test
    
def mean_encoding_kfold(train, test, cols, target, alpha):
    y_tr = train[target].values
    prior = np.median(y_tr)
    for col in cols:
        train[col+'_mean'] = prior

    train_new = train.copy()
    test_new = test.copy()

    # train mapping
    kf = StratifiedKFold(n_splits=3, random_state=2019)
    for i, (tr_ind,val_ind) in enumerate(kf.split(y_tr, y_tr)):
        X_tr, X_val = train.iloc[tr_ind], train.iloc[val_ind]
        X_tr = mean_encoding(X_tr, cols, target, alpha)
        for col in cols:
            means = X_val[col].map(X_tr.groupby(col)[col+'_mean'].mean())
            X_val[col+'_mean'] = means
        train_new.iloc[val_ind] = X_val

    # test mapping
    for col in cols:
        test_new[col+'_mean'] = test_new[col].map(train_new.groupby(col)[target].mean())

    return train_new, test_new    

train_proc, test_proc = mean_encoding_kfold(train_proc, test_proc, ["Breed1", "Color1", "State"], "AdoptionSpeed", alpha=25)    
print("Train shape {}, test shape {} after adding mean encoding features".format(train_proc.shape, test_proc.shape))

########################################################################################################################
# MERGING TRAIN, TEST AND DIVIDE TO HA & AN DATASET

X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)

#count resuer ID
def rescuer_count(X):
    rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']

    # Merge as another feature onto main DF:
    X = X.merge(rescuer_count, how='left', on='RescuerID')
    return X
X = rescuer_count(X)

x_col_a = ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID',
       'VideoAmt', 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed',"RescuerID_COUNT",
       'sentiment_sentiment_magnitude_MEAN',
       'sentiment_sentiment_magnitude_SUM',
       'sentiment_sentiment_magnitude_MAX',
       'sentiment_sentiment_magnitude_MIN', 'sentiment_sentiment_score_MEAN',
       'sentiment_sentiment_score_SUM', 'sentiment_sentiment_score_MAX',
       'sentiment_sentiment_score_MIN',
       'sentiment_sentiment_document_magnitude_MEAN',
       'sentiment_sentiment_document_magnitude_SUM',
       'sentiment_sentiment_document_magnitude_MAX',
       'sentiment_sentiment_document_magnitude_MIN',
       'sentiment_sentiment_document_score_MEAN',
       'sentiment_sentiment_document_score_SUM',
       'sentiment_sentiment_document_score_MAX',
       'sentiment_sentiment_document_score_MIN',
       'metadata_metadata_annots_score_MEAN',
       'metadata_metadata_annots_score_SUM',
       'metadata_metadata_annots_score_MAX',
       'metadata_metadata_annots_score_MIN',
       'metadata_metadata_color_score_MEAN',
       'metadata_metadata_color_score_SUM',
       'metadata_metadata_color_score_MAX',
       'metadata_metadata_color_score_MIN',
       'metadata_metadata_color_pixelfrac_MEAN',
       'metadata_metadata_color_pixelfrac_SUM',
       'metadata_metadata_color_pixelfrac_MAX',
       'metadata_metadata_color_pixelfrac_MIN',
       'metadata_metadata_crop_conf_MEAN', 'metadata_metadata_crop_conf_SUM',
       'metadata_metadata_crop_conf_MAX', 'metadata_metadata_crop_conf_MIN',
       'metadata_metadata_crop_importance_MEAN',
       'metadata_metadata_crop_importance_SUM',
       'metadata_metadata_crop_importance_MAX',
       'metadata_metadata_crop_importance_MIN', 'metadata_annots_top_desc',
       'sentiment_entities','sentiment_text']
X_temp_h = X.copy()
X_temp_a = X[x_col_a].copy()

print("Ha's features shape: {}".format(X_temp_h.shape))
print("An's features shape: {}".format(X_temp_a.shape))

#X_temp_h = X_temp_h.merge(agg_imgs,how='outer', on='PetID')
X_temp_a = X_temp_a.merge(agg_imgs,how='outer', on='PetID')
print("Ha's features shape after adding aggregated image features: {}".format(X_temp_h.shape))
print("An's features shape after adding aggregated image features: {}".format(X_temp_a.shape))

########################################################################################################################
# FEATURE ENGINEERING FOR AN'S DATASET

# CONVERT BREED AND COLOR TO TEXT 
def concat_breed_color(X):
    #breed
    breed_main = X["main_breed_BreedName"].fillna("")
    breed_second = X["second_breed_BreedName"].fillna("")
    
    #color
    color_1 = X[['Color1']].merge(labels_color, how='left', left_on='Color1', right_on='ColorID')
    color_1 = color_1.iloc[:, 2:]["ColorName"]
    color_1 = color_1.fillna("")

    color_2 = X[['Color2']].merge(labels_color, how='left', left_on='Color2', right_on='ColorID')
    color_2 = color_2.iloc[:, 2:]["ColorName"]
    color_2 = color_2.fillna("")

    color_3 = X[['Color3']].merge(
        labels_color, how='left',
        left_on='Color3', right_on='ColorID')

    color_3 = color_3.iloc[:, 2:]["ColorName"]
    color_3 = color_3.fillna("")

    features = []
    for b1,b2,c1,c2,c3 in zip(breed_main,breed_second,color_1,color_2,color_3):
        features.append(" ".join([b1,b2,c1,c2,c3]))
    features = np.asarray(features)

    return features

X_temp_a["Features"] = concat_breed_color(X)

########################################################################################################################
# 3, TEXT PROCESSING 


text_columns_h = ['sentiment_text', 'metadata_annots_top_desc', 'sentiment_entities']
text_columns_a = ['Description', 'metadata_annots_top_desc', 'sentiment_entities','Features','sentiment_text']
tfidf_col_a = ['Processed_desc', 'metadata_annots_top_desc', 'sentiment_entities','Features','Processed_sentiment']
categorical_columns_h = ['main_breed_BreedName', 'second_breed_BreedName']

categorical_columns_a = ['Type','Breed1', 'Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength',
                      'Vaccinated','Dewormed','Sterilized','Health','State']

to_drop_columns = ['PetID', 'Name', 'RescuerID']
#meta_sentiment = ['metadata_annots_top_desc', 'sentiment_entities',"Features"]

# Factorize categorical columns (Ha's data):
for i in categorical_columns_h:
    X_temp_h.loc[:, i] = pd.factorize(X_temp_h.loc[:, i])[0]

X_text_h = X_temp_h[text_columns_h]
X_text_a = X_temp_a[text_columns_a]


#Fill NA
for i,j in zip(X_text_h.columns,X_text_a.columns):
    X_text_h.loc[:, i] = X_text_h.loc[:, i].fillna('<MISSING>')
    X_text_a.loc[:, j] = X_text_a.loc[:, j].fillna('<MISSING>')


########################################################################################################################
# 3.a) PREPROCESSING AND FASTTEXT
X_text_a["Description"] = X_text_a["Description"].apply(lambda x: re.sub('\n', ' ', x))
X_text_a["sentiment_text"] = X_text_a["sentiment_text"].apply(lambda x: re.sub('\n', ' ', x))

# PREPROCESSING (HA)
def alpha(s):
    s = [word.lower() for word in s if word.isalpha() == True]
    return s

def remove_stopwords(s):
    s = [word.lower() for word in s if not word.lower() in stopwords.words('english')]
    return s

def lemmatize(s):
    lemmatizer = WordNetLemmatizer()
    s = [lemmatizer.lemmatize(word) for word in s]
    return s

def tokenize(data):
    
    tokenized_docs = data.apply(lambda x: word_tokenize(str(x)))
    
    filtered_docs = tokenized_docs.apply(lambda x: remove_stopwords(x)) 
    alpha_tokens = filtered_docs.apply(lambda x: alpha(x))
    lem_tokens = alpha_tokens.apply(lambda x: lemmatize(x))
    sentences_back = lem_tokens.apply(lambda x:  " ".join(x))
    
    return sentences_back

data=X_text_h.sentiment_text
data = tokenize(data)
X_text_h.sentiment_text = data
X_text_h['sentiment_text'].to_csv('train_unsupervised.txt', index=False, header = False) #save to file

# PREPROCESSING (AN)

num2words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', \
             6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', \
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', \
            19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty', \
            50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', \
            90: 'ninety', 0: 'zero'}

def n2w(m):
    n = int(m.group(0))
    if len(str(n)) == 1:
        return num2words[n%10].lower()
    elif len(str(n)) == 2:
        return num2words[n-n%10].lower() +" " +num2words[n%10].lower()
    else:
        return "abovehundred"

X_text_a["New_description"] = X_text_a["Description"].apply(lambda x: re.sub("\d+",n2w,x))
X_text_a["New_sentiment"] = X_text_a["sentiment_text"].apply(lambda x: re.sub("\d+",n2w,x))


tokenizer = RegexpTokenizer(r'[A-Za-z]\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['+','-','.','@','/', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

raw_docs = X_text_a['New_description'].tolist()
raw_docs_2 = X_text_a['New_sentiment'].tolist()

processed_docs = [] #for TFIDF
processed_docs_2 = [] #for TFIDF

for doc,doc_2 in zip(raw_docs,raw_docs_2):
    tokens = tokenizer.tokenize(doc)
    tokens_2 = tokenizer.tokenize(doc_2)
    filtered = [word for word in tokens if word.lower() not in stop_words]
    filtered_2 = [w for w in tokens_2 if w.lower() not in stop_words]

    processed_docs.append(" ".join(filtered))
    processed_docs_2.append(" ".join(filtered_2))
    
X_text_a["Processed_desc"] = np.asarray(processed_docs)
X_text_a["Processed_sentiment"] = np.asarray(processed_docs_2)

X_text_a['Processed_desc'].to_csv('processed_docs.txt', index=False, header = False) #save to file
X_text_a['Processed_sentiment'].to_csv('processed_sentiment.txt', index=False, header = False) #save to file

# FASTTEXT

def fastText_vectorize(X,col,textfile,dimension):
    clf = ft.train_unsupervised(input=textfile, \
        ws = 5, minCount=10, epoch=10, \
        minn = 3, maxn = 6, wordNgrams = 3, \
        dim=dimension,thread=1)

    fastText_text_features = []
    for description in col:
        #print(description)
        fastText_text_features.append(clf.get_sentence_vector(description))

    fastText_text_features = pd.DataFrame(fastText_text_features)
    fastText_text_features = fastText_text_features.add_prefix('fastText_Description_'.format(i))

    # Concatenate with main DF:
    #X = pd.concat([X, fastText_text_features], axis=1)
    return fastText_text_features

fastText_h = fastText_vectorize(X_temp_h,X_text_h["sentiment_text"],"train_unsupervised.txt",200)
X_temp_h = pd.concat([X_temp_h, fastText_h], axis=1)
print("Ha's data shape after adding fastText features: {}".format(X_temp_h.shape))

fastText_a = fastText_vectorize(X_temp_a,X_text_a["Description"],"processed_docs.txt",100)
X_temp_a = pd.concat([X_temp_a, fastText_a], axis=1)
print("An's data shape after adding fastText features: {}".format(X_temp_a.shape))
########################################################################################################################
X_temp_a['lv_ratio'] = X_text_a.apply(lambda x: lv.ratio(x['sentiment_text'], x['Description']), axis=1)
########################################################################################################################
# 3.b) TFIDF

def tfidf_vectorize(n_components,X_text,X_temp):
    text_features = []

    # Generate text features:
    for i in X_text.columns:

        #if i != 'Description':

            # Initialize decomposition methods:
            print('generating features from: {}'.format(i))
            svd_ = TruncatedSVD(
                n_components=n_components, random_state=1337)
            nmf_ = NMF(
                n_components=n_components, random_state=1337)

            tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
            svd_col = svd_.fit_transform(tfidf_col)
            svd_col = pd.DataFrame(svd_col)
            svd_col = svd_col.add_prefix('SVD_{}_'.format(i))

            nmf_col = nmf_.fit_transform(tfidf_col)
            nmf_col = pd.DataFrame(nmf_col)
            nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))

            text_features.append(svd_col)
            text_features.append(nmf_col)


    # Combine all extracted features:
    text_features = pd.concat(text_features, axis=1)

    # Concatenate with main DF:
    X_temp = pd.concat([X_temp, text_features], axis=1)
    
    return X_temp

X_temp_a = tfidf_vectorize(16,X_text_a[tfidf_col_a],X_temp_a)
X_temp_h = tfidf_vectorize(50,X_text_h,X_temp_h)

print("Ha's data shape after adding TFIDF features: {}".format(X_temp_h.shape))
print("An's data shape after adding TFIDF features: {}".format(X_temp_a.shape))

X_temp_a.drop(text_columns_a,axis=1, inplace=True)
X_temp_h.drop(text_columns_h+["Description"],axis=1, inplace=True)

########################################################################################################################
# 4. IMAGE PROCESSING 

# 4.a) PREPROCESSING (RESIZE, LOAD IMAGE)
pca = PCA(n_components=1,random_state=1234)
img_size = 256
train_ids = train['PetID'].values
test_ids = test['PetID'].values

#n_batches = len(pet_ids) // batch_size + 1

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

train_tmp = train.set_index(['PetID'], drop = False)
test_tmp = test.set_index(['PetID'], drop = False)    
train_feats_h,train_feats_a = extract_img_features("train_images", train_images,train_ids, train_tmp.copy())
test_feats_h,test_feats_a = extract_img_features("test_images", test_images, test_ids, test_tmp.copy())
feats_h = pd.concat([train_feats_h, test_feats_h], axis=0)
feats_a = pd.concat([train_feats_a, test_feats_a], axis=0)
feats_h["PetID"] = feats_h.index
feats_a["PetID"] = feats_a.index


# MERGING WITH TEXT AND OTHER FEATURES
X_h = pd.merge(X_temp_h, feats_h, how='left', on='PetID')
X_h.drop(to_drop_columns, axis=1,inplace=True)

X_a = pd.merge(X_temp_a, feats_a, how='left', on='PetID')
X_a.drop(to_drop_columns, axis=1,inplace=True)

print("Ha's data shape after adding image features: {}".format(X_h.shape))
print("An's data shape after adding image features: {}".format(X_a.shape))

########################################################################################################################
# 5. SPLIT AND TRAIN DATA

# 5.a) OptimizeRounder, QWK
# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def predict_new(self, X, coef):
        X_p = np.copy(X)
        X_s = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
            X_s[i] = pred
        return X_p, X_s
        
    def coefficients(self):
        return self.coef_['x']
    
def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

########################################################################################################################
# 5.b) SPLITTING DATA

def splitting(X_):
    # Split into train and test again:
    X_train = X_.loc[np.isfinite(X_.AdoptionSpeed), :]
    X_test = X_.loc[~np.isfinite(X_.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed'], axis=1)


    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    assert X_train.shape[0] == train.shape[0]
    assert X_test.shape[0] == test.shape[0]


    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')

    test_cols = X_test.columns.tolist()

    assert np.all(train_cols == test_cols)
    
    return X_train,X_test
X_train_h,X_test_h = splitting(X_h)
X_train_a,X_test_a = splitting(X_a)

X_train_h = X_train_h.fillna(0)
X_test_h = X_test_h.fillna(0)

cat_features = [X_a.columns.get_loc(c) for c in X_a.columns if c in categorical_columns_a]


########################################################################################################################
# 5.c) MODEL BUILDING AND TRAIN

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
          'max_depth': 10,
          'learning_rate': 0.01,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 150,
          'min_child_weight': 0.02,
          'lambda_l2': 0.0475,
          'verbosity': -1,
          'data_random_seed': 17}

# Additional parameters:
early_stop = 1000
verbose_eval = 100
num_rounds = 9500
n_splits = 5

def LGBM_train(X_train,X_test,cat_feature=False):
    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    i = 0
    feature_importance_df = pd.DataFrame()
    j = 1
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):

        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))

        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        if cat_feature == False:
            model = lgb.train(params,
                              train_set=d_train,
                              num_boost_round=num_rounds,
                              valid_sets=watchlist,
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stop)
        else:
            model = lgb.train(params,
                              train_set=d_train,
                              num_boost_round=num_rounds,
                              valid_sets=watchlist,
                              verbose_eval=verbose_eval,
                              categorical_feature=list(cat_features),
                              early_stopping_rounds=early_stop)


        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance()
        fold_importance_df['fold'] = j
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        
        j += 1

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        oof_train[valid_index] = val_pred
        oof_test[:, i] = test_pred

        i += 1
        
    return oof_train,oof_test,feature_importance_df

oof_train_h,oof_test_h,feature_importance_df_h = LGBM_train(X_train_h,X_test_h,cat_feature=False)
oof_train_a,oof_test_a,feature_importance_df_a = LGBM_train(X_train_a,X_test_a,cat_feature=True)


## Fitting again with oof_train
# Compute QWK based on OOF train predictions:

# HA
optR = OptimizedRounder()
optR.fit(oof_train_h, X_train_h['AdoptionSpeed'].values)
coefficients_h = optR.coefficients()
pred_test_y_k_h = optR.predict(oof_train_h, coefficients_h)
print("\nValid Counts = ", Counter(X_train_h['AdoptionSpeed'].values))
print("Predicted Counts = ", Counter(pred_test_y_k_h))
print("Coefficients = ", coefficients_h)
qwk_h = quadratic_weighted_kappa(X_train_h['AdoptionSpeed'].values, pred_test_y_k_h)
print("QWK = ", qwk_h)

coefficients_1 = coefficients_h.copy()
coefficients_1[0] = 1.645
coefficients_1[1] = 2.115
coefficients_1[3] = 2.84

train_predictions_h, train_scores_h = optR.predict_new(oof_train_h, coefficients_1)
qwk_h = quadratic_weighted_kappa(X_train_h['AdoptionSpeed'].values, train_predictions_h)
print('train pred distribution: {}'.format(Counter(train_predictions_h)))
print('new QWK ha: {}'.format(qwk_h))

test_predictions_h, test_scores_h = optR.predict_new(oof_test_h.mean(axis=1), coefficients_1)
print('test pred distribution: {}'.format(Counter(test_predictions_h)))

#######################################################

# AN
optR = OptimizedRounder()
optR.fit(oof_train_a, X_train_a['AdoptionSpeed'].values)
coefficients_a = optR.coefficients()
pred_test_y_k_a = optR.predict(oof_train_a, coefficients_a)
print("\nValid Counts = ", Counter(X_train_a['AdoptionSpeed'].values))
print("Predicted Counts = ", Counter(pred_test_y_k_a))
print("Coefficients = ", coefficients_a)
qwk_a = quadratic_weighted_kappa(X_train_a['AdoptionSpeed'].values, pred_test_y_k_a)
print("QWK = ", qwk_a)

coefficients_2 = coefficients_a.copy()
coefficients_2[0] = 1.5
coefficients_2[1] = 2.06
coefficients_2[2] = 2.45
coefficients_2[3] = 2.89

train_predictions_a, train_scores_a = optR.predict_new(oof_train_a, coefficients_2)
qwk_a = quadratic_weighted_kappa(X_train_a['AdoptionSpeed'].values, train_predictions_a)
print('train pred distribution: {}'.format(Counter(train_predictions_a)))
print('new QWK an: {}'.format(qwk_a))

test_predictions_a, test_scores_a = optR.predict_new(oof_test_a.mean(axis=1), coefficients_2)
print('test pred distribution: {}'.format(Counter(test_predictions_a)))

#######################################################
# SAVE AND SUBMIT

def blend(row, coefficients_blend):
    pred, mod_value = divmod(row["Pred_a"] + row["Pred_h"], 2)
    if mod_value == 0:
        return pred
    else:
        score_blend = (row["Score_a"] + row["Score_h"]) / 2.0
        if score_blend >= coefficients_blend[pred]: 
            return pred + 1
        else:
            return pred
    return pred

coefficients_blend = coefficients_1.copy()
for i in range(4):
    coefficients_blend[i] = (coefficients_1[i] + coefficients_2[i]) / 2.0

# calcualte QWK for blending train
df_an_train = pd.DataFrame({'PetID': train['PetID'].values, 'Pred_a': train_predictions_a.astype(np.int32), 'Score_a': train_scores_a})
df_an_train.to_csv('train_an.csv', index=False)

df_ha_train = pd.DataFrame({'PetID': train['PetID'].values, 'Pred_h': train_predictions_h.astype(np.int32), 'Score_h': train_scores_h})
df_ha_train.to_csv('train_ha.csv', index=False)

df_blend_train = pd.merge(df_an_train, df_ha_train, on='PetID')
df_blend_train["AdoptionSpeed"] = df_blend_train.apply(lambda row: blend(row, coefficients_blend), axis = 1)
print("Predicted Counts Blended Train = ", Counter(df_blend_train["AdoptionSpeed"].values))

qwk_blend = quadratic_weighted_kappa(X_train_a['AdoptionSpeed'].values, df_blend_train['AdoptionSpeed'].values)
print('QWK blend: {}'.format(qwk_blend))

df_blend_train['AdoptionSpeed_Simple'] = (df_blend_train["Pred_a"] + df_blend_train["Pred_h"])/2
df_blend_train['AdoptionSpeed_Simple'] = df_blend_train.AdoptionSpeed_Simple.astype(int)
df_blend_train.to_csv('train_blend.csv', index=False)
print("Predicted Counts Blended Simple Train = ", Counter(df_blend_train["AdoptionSpeed_Simple"].values))

qwk_blend_simple = quadratic_weighted_kappa(X_train_a['AdoptionSpeed'].values, df_blend_train['AdoptionSpeed_Simple'].values)
print('QWK blend simple: {}'.format(qwk_blend_simple))

# generate submission
df_an_submission = pd.DataFrame({'PetID': test['PetID'].values, 'Pred_a': test_predictions_a.astype(np.int32), 'Score_a': test_scores_a})
df_an_submission.to_csv('submission_an.csv', index=False)

df_ha_submission = pd.DataFrame({'PetID': test['PetID'].values, 'Pred_h': test_predictions_h.astype(np.int32), 'Score_h': test_scores_h})
df_ha_submission.to_csv('submission_ha.csv', index=False)

df_blend_submission = pd.merge(df_an_submission, df_ha_submission, on='PetID')
df_blend_submission["AdoptionSpeed"] = df_blend_submission.apply(lambda row: blend(row, coefficients_blend), axis = 1)
df_blend_submission.to_csv('submission.csv', columns = ['PetID', 'AdoptionSpeed'], index=False)