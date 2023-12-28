import sys
import importlib
sys.path.insert(0,'./content/params')
sys.path.insert(1, './content/metrics/')
sys.path.insert(2,'./content/model_class')

import param_rating
import map_at_k
importlib.reload(param_rating)
importlib.reload(map_at_k)
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder

import pandas as pd
import numpy as np
import os
import neural_metrics
from tqdm import tqdm
from tensorflow import keras
from keras.preprocessing import image
from keras.models import  Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.applications import DenseNet121
from keras import regularizers
from keras.metrics import *
from keras.optimizers import Adam
class ModelByImage:
    def __init__(self, image_source, movie_train,movie_test,weight_path=""):
        self.image_source = image_source
        self.weight_path = weight_path
        self.input_shape = (200, 200, 3)
        self.num_classes = 18
        self.learning_rate = 0.01
        self.pre_movies_train = movie_train.copy()
        self.pre_movies_test = movie_test.copy()
        self.model=None
        self.enc = OrdinalEncoder()
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"File {self.image_source} not found!")
        if (not os.path.exists(weight_path)) and len(weight_path)!=0:
            raise FileNotFoundError(f"File {self.weight_path} not found!")
        self.pre_movies_train.reset_index(inplace=True)
        self.pre_movies_test.reset_index(inplace=True)
        #Condition
        self.isTrained=False
        self.isPredict=False
        self.isPreprocess=False
    def check_exist(self, data):
        sourcedir = self.image_source
        delete_list_data = [2085, 47, 3941, 2364, 97, 2848, 3758, 3935, 681, 769, 1421, 571]
        for i in range(data.shape[0]):
            file_name = str(data['movieid'].iloc[i]) + '.jpg'
            flag = True
            for path in os.listdir(sourcedir):
                if file_name == path:
                    flag = False
                    continue
            if (flag):
                delete_list_data.append(data['movieid'][i])
        return delete_list_data

    def preprocessing(self):
        self.isPreprocess=True
        print("Preprocessing...")
        delete_train = self.check_exist(self.pre_movies_train)
        delete_test = self.check_exist(self.pre_movies_test)

        self.movies_train_update = self.pre_movies_train[~self.pre_movies_train['movieid'].isin(delete_train)]
        self.movies_test_update = self.pre_movies_test[~self.pre_movies_test['movieid'].isin(delete_test)]

        new_file_train_path = './content/cleaned_data/movies_train_update_in_code.DAT'
        self.movies_train_update.to_csv(new_file_train_path, sep=',', encoding='latin-1', index=False, header=False)
        new_file_test_path = './content/cleaned_data/movies_test_update_in_code.DAT'
        self.movies_test_update.to_csv(new_file_test_path, sep=',', encoding='latin-1', index=False, header=False)

        self.movies_train = pd.read_csv ( './content/cleaned_data/movies_train_update_in_code.DAT', engine='python', sep=',',
                                              names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False )
        self.movies_test = pd.read_csv ( './content/cleaned_data/movies_test_update_in_code.DAT', engine='python', sep=',',
                                             names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False )
        self.movies_train['genre'] = self.movies_train.genre.str.split('|')
        self.movies_test['genre'] = self.movies_test.genre.str.split('|')

        self.x_train, self.y_train = self.__load_data(self.movies_train)
        self.x_test, self.y_test = self.__load_data(self.movies_test)
        for genre in param_rating.genre2idx.keys():
            self.movies_test[genre] = self.movies_test['genre'].apply(lambda x: 1 if genre in x else 0)
        self.genre_df = pd.DataFrame(self.movies_test['genre'].explode())
    def __load_data(self, data):
        X_dataset = []
        for i in tqdm(range(data.shape[0])):
            img = image.load_img(self.image_source + '/' + str(data['movieid'].iloc[i]) + '.jpg', target_size=self.input_shape)
            img = image.img_to_array(img)
            img = img / 255.
            X_dataset.append(img)
        X = np.array(X_dataset)
        mlb = MultiLabelBinarizer ()
        mlb.fit(data['genre'])
        Y = mlb.transform(data['genre'])
        return X, Y
    def train(self):
        if self.isPreprocess==False:
            raise Exception('preprocessing() needs to be proceeded first!')
        self.isTrained=True
        print("Training...")
        base_model = DenseNet121(include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='binary_crossentropy',
                        metrics=[neural_metrics.f1_m])
        for layer in base_model.layers:
            layer.trainable = False
        if len(self.weight_path)!=0:
            self.model.load_weights(self.weight_path)
        else:# train 7 million params→ takes a lot of time
            _ = self.model.fit(self.x_train, self.y_train, verbose = 1, epochs=50,
                                     validation_data=(self.x_test, self.y_test),batch_size = 64)
    def getValueByImage(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocessing() needs to be proceeded first!')
        
        train_img=pd.DataFrame(self.model.predict(self.x_train))
        submission_img = pd.DataFrame(columns=param_rating.genre2idx.keys())
        for label in param_rating.genre2idx.keys():
            submission_img[label]=train_img[param_rating.genre2idx[label]]
        submission_img['movieid']=self.movies_train['movieid']
        submission_img.to_csv('./content/submission3.csv')
        return train_img
    def predict(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocessing() needs to be proceeded first!')
        self.isPredict=True
        print("Predicting...")
        self.y_pred = self.model.predict(self.x_test)
        test_img=pd.DataFrame(self.y_pred)
        self.submission_imgtest = pd.DataFrame(columns=param_rating.genre2idx.keys())
        for label in param_rating.genre2idx.keys():
            self.submission_imgtest[label]=test_img[param_rating.genre2idx[label]]
        self.sorted_prediction_ids = np.argsort(-self.y_pred, axis=1)
        self.enc.fit_transform(self.genre_df[['genre']])
        self.vectors_labels_test = self.movies_test.drop(columns = ['movieid', 'title', 'genre'], axis = 1)

    def __get_column_names(self, row):
        return list(self.vectors_labels_test.columns[row == 1])
    def evaluate_model(self):
        print("Evaluating...")
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocessing() needs to be proceeded first!')
        if self.isPredict==False:
            self.predict()
        vectors_labels_test_new = self.vectors_labels_test.apply(self.__get_column_names, axis=1).tolist ()
        top_5_prediction_ids = self.sorted_prediction_ids[:, :5]
        original_shape = top_5_prediction_ids.shape
        top_5_predictions = self.enc.inverse_transform(top_5_prediction_ids.reshape(-1, 1))
        top_5_predictions = top_5_predictions.reshape(original_shape)
        print('Map@K score =  {:.3}'.format(map_at_k.mapk(vectors_labels_test_new, top_5_predictions, k = 5)))
    def getFinalPrediction(self):
        submission_imgtest_sub=self.submission_imgtest
        submission_imgtest_sub['movieid']=self.movies_test['movieid']
        return submission_imgtest_sub