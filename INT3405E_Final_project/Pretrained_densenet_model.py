from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import regularizers
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
#%%


class Pretrained_densenet_model:
    def __init__(self, weight_path='dataset_cleaned/best_weights_32.h5', input_shape=(200, 200, 3),
                 num_classes=18, learning_rate=0.001):
        self.weight_path = weight_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

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

    def predict(self, data):
        # Perform prediction here
        return self.model.predict(data)
#%%
def preprocessing(data_path):
    movies_test = pd.read_csv(data_path, engine='python',sep=',', 
                              names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False)
    movies_test['genre'] = movies_test.genre.str.split('|')
    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    for genre in genres:
        movies_test[genre] = movies_test['genre'].apply(lambda x: 1 if genre in x else 0)
    return movies_test
#%%
def load_data_test(data):
  X_dataset = []
  for i in tqdm(range(data.shape[0])):
    img = image.load_img('ml1m-images/' +str(data['movieid'][i])+'.jpg', target_size=(200,200,3))
    img = image.img_to_array(img)
    img = img/255.
    X_dataset.append(img)
  X = np.array(X_dataset)
  mlb = MultiLabelBinarizer()
  mlb.fit(data['genre'])
  y = mlb.transform(data['genre'])
  return X, y

#%%
def main():
    test_path = 'dataset_cleaned/movies_test_update.DAT'
    movies_test = preprocessing(test_path)
    print(movies_test)
    x_test, y_test = load_data_test(movies_test)
    weight_path = 'dataset_cleaned/best_weights_32.h5'
    
    model = Pretrained_densenet_model(weight_path)
    y_pred = model.predict(x_test)
    print(y_pred)
#%%
if __name__ == '__main__':
    main()