from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import *

from keras.preprocessing import image
import keras.backend as K
from keras.optimizers import Adam

from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib
import sys
sys.path.insert(0, 'metric/')
import map_k
importlib.reload(map_k)
enc = OrdinalEncoder ()
#%%


class Pretrained_densenet_model:
    def __init__(self, weight_path, data_path):
        self.weight_path = weight_path
        self.input_shape = (200, 200, 3)
        self.num_classes = 18
        self.learning_rate = 0.01
        self.data_path = data_path

        base_model = DenseNet121(include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        self.model.load_weights(self.weight_path)
    def preprocessing(self):
        self.movies_test = pd.read_csv(self.data_path, engine='python', sep=',',
                                    names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False)
        self.movies_test['genre'] = self.movies_test.genre.str.split('|')
        genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        for genre in genres:
            self.movies_test[genre] = self.movies_test['genre'].apply(lambda x: 1 if genre in x else 0)
        self.genre_df = pd.DataFrame(self.movies_test['genre'].explode())
    def load_data_test(self):
        X_dataset = []
        for i in tqdm(range(self.movies_test.shape[0])):
            img = image.load_img('ml1m-images/' + str(self.movies_test['movieid'][i]) + '.jpg', target_size=self.input_shape)
            img = image.img_to_array(img)
            img = img / 255.
            X_dataset.append(img)
        self.x_test = np.array(X_dataset)
        mlb = MultiLabelBinarizer ()
        mlb.fit(self.movies_test['genre'])
        self.y_test = mlb.transform(self.movies_test['genre'])
    def evaluate(self):
        self.y_pred = self.model.predict(self.x_test)
        self.sorted_prediction_ids = np.argsort(-self.y_pred, axis=1)

        enc.fit_transform(self.genre_df[['genre']])
        self.vectors_labels_test = self.movies_test.drop(columns=['movieid', 'title', 'genre'], axis=1)

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
    test_path = 'dataset_cleaned/movies_test_update.DAT'
    weight_path = 'dataset_cleaned/best_weights_32.h5'
    
    model = Pretrained_densenet_model(weight_path, test_path)
    model.preprocessing()
    model.load_data_test()
    model.evaluate()
    model.calculating()
#%%
if __name__ == '__main__':
    main()