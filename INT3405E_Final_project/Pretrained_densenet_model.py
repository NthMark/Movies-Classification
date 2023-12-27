from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import *

from keras.preprocessing import image
import keras.backend as K
from keras.optimizers import Adam

from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from cleanvision.imagelab import Imagelab
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import importlib
import sys
sys.path.insert(0, 'metric/')
import map_k
importlib.reload(map_k)
enc = OrdinalEncoder ()
#%%


class Pretrained_densenet_model:
    def __init__(self, image_source,weight_path, train_path,test_path):
        self.image_source = image_source
        self.weight_path = weight_path
        self.input_shape = (200, 200, 3)
        self.num_classes = 18
        self.learning_rate = 0.01
        self.test_path = test_path
        self.train_path = train_path

        base_model = DenseNet121(weights = 'dataset_cleaned/densenet121_notop.h5',
                                 include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='binary_crossentropy',
                        metrics=[self.f1_m])
        for layer in base_model.layers:
            layer.trainable = False
        self.model.load_weights(self.weight_path)

    def check_exist(self, data):
        sourcedir = self.image_source
        delete_list_data = [2085, 47, 3941, 2364, 97, 2848, 3758, 3935, 681, 769, 1421, 571]
        for i in range(data.shape[0]):
            file_name = str(data['movieid'][i]) + '.jpg'
            flag = True
            for path in os.listdir(sourcedir):
                if file_name == path:
                    flag = False
                    continue
            if (flag):
                delete_list_data.append(data['movieid'][i])
        return delete_list_data

    def preprocessing(self):
        self.pre_movies_train = pd.read_csv(self.train_path, engine='python', sep='::',
                                         names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False)
        self.pre_movies_test = pd.read_csv(self.test_path, engine='python', sep='::',
                                    names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False)
        self.pre_movies_train['genre'] = self.pre_movies_train.genre.str.split('|')
        self.pre_movies_test['genre'] = self.pre_movies_test.genre.str.split('|')

        delete_train = self.check_exist(self.pre_movies_train)
        delete_test = self.check_exist(self.pre_movies_test)

        self.movies_train = self.pre_movies_train[~self.pre_movies_train['movieid'].isin(delete_train)]
        self.movies_test = self.pre_movies_test[~self.pre_movies_test['movieid'].isin(delete_test)]

        # new_file_train_path = 'dataset_cleaned/movies_train_update.DAT'
        # self.movies_train.to_csv(new_file_train_path, sep=',', encoding='latin-1', index=False, header=False)
        # new_file_test_path = '/content/drive/MyDrive/output/movies_test_update.DAT'
        # self.movies_test.to_csv(new_file_test_path, sep=',', encoding='latin-1', index=False, header=False)
        
        genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        for genre in genres:
            self.movies_test[genre] = self.movies_test['genre'].apply(lambda x: 1 if genre in x else 0)
        self.genre_df = pd.DataFrame(self.movies_test['genre'].explode())
        print(self.movies_test)
    def load_data(self, data):
        X_dataset = []
        for i in tqdm(range(data.shape[0])):
            img = image.load_img(self.image_source + '/' + str(data['movieid'][i]) + '.jpg', target_size=self.input_shape)
            img = image.img_to_array(img)
            img = img / 255.
            X_dataset.append(img)
        X = np.array(X_dataset)
        mlb = MultiLabelBinarizer ()
        mlb.fit(data['genre'])
        Y = mlb.transform(data['genre'])
        return X, Y
    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon ())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon ())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon ()))
    def train(self):
        self.x_train, self.y_train = self.load_data(self.movies_train)
        self.x_test, self.y_test = self.load_data(self.movies_test)
        history = self.model.fit(self.x_train, self.y_train, verbose = 1, epochs=50,
                                 validation_data=(self.x_test, self.y_test),batch_size = 64)
    def evaluate(self):
        self.y_pred = self.model.predict(self.x_test)
        self.sorted_prediction_ids = np.argsort(-self.y_pred, axis=1)

        enc.fit_transform(self.genre_df[['genre']])
        self.vectors_labels_test = self.y_test

    def get_column_names(self, row):
        return list(self.vectors_labels_test.columns[row == 1])
    def calculating(self):
        vectors_labels_test_new = self.vectors_labels_test.apply(self.get_column_names, axis=1).tolist ()
        top_5_prediction_ids = self.sorted_prediction_ids[:, :5]
        original_shape = top_5_prediction_ids.shape
        top_5_predictions = enc.inverse_transform(top_5_prediction_ids.reshape(-1, 1))
        top_5_predictions = top_5_predictions.reshape(original_shape)
        print('Map@K score =  {:.3}'.format(map_k.mapk(vectors_labels_test_new, top_5_predictions, k = 5)))


def main():
    train_path = 'pre_data/movies_train.dat'
    test_path = 'pre_data/movies_test.dat'
    weight_path = 'dataset_cleaned/best_weights_32.h5'
    image_source = 'ml1m-images'
    model = Pretrained_densenet_model(image_source,weight_path, train_path, test_path)
    model.preprocessing()
    # model.train()
    # model.evaluate()
    # model.calculating()
#%%
if __name__ == '__main__':
    main()