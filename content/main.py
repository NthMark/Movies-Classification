import sys
import importlib
sys.path.insert(0,'./content/params')
sys.path.insert(1, './content/metrics/')
sys.path.insert(2,'./content/model_class')

import param_rating
import map_at_k
importlib.reload(param_rating)
importlib.reload(map_at_k)
from  model_by_rating import ModelByRating
from model_by_title import ModelByTitle
from model_by_img import ModelByImage
from bayes_opt import BayesianOptimization

import pandas as pd
import numpy as np

class FinalModel:
    def __init__(self, movie_train, movie_test,user,rating,image_source,weight_path,weight_path_title,data_test_cleaned,data_train_cleaned) :
        self.movie_train=movie_train.copy()
        self.movie_test=movie_test.copy()
        self.user=user.copy()
        self.rating=rating.copy()
        self.image_source=image_source
        self.weight_path=weight_path
        self.weight_path_title=weight_path_title
        self.data_test_cleaned=data_test_cleaned
        self.data_train_cleaned=data_train_cleaned
        self.w1=0      #0.05445067852011942
        self.w2=0      #0.6703042100155795
        self.w3=0      #0.07302313058669374
        self.onceTime=False
        self.onceTimeTrain=False
        #Condition
        self.isTrained=False
        self.isPredict=False
        self.isPreprocess=False
    def preprocess_data(self):
        self.isPreprocess=True
        print("Preprocessing...(final model)!")
        self.movie_test_sub=self.movie_test.copy()
        self.movie_test_sub['genre'] = self.movie_test_sub.genre.str.split('|')
        self.movie_test_sub.reset_index(inplace=True)
        vectors_genre=[]
        for genre in self.movie_test_sub.genre.tolist():
            genre_vector = np.zeros(len(param_rating.genre2idx))
            for g in genre:
                genre_vector[param_rating.genre2idx[g]] = 1
            vectors_genre.append(genre_vector)
        self.movie_test_sub['genre_vectors']=vectors_genre

        genre_test = pd.DataFrame(self.movie_test_sub['genre_vectors'].tolist(), columns=param_rating.genre2idx.keys())
        self.movie_test_sub = pd.concat([self.movie_test_sub, genre_test], axis=1)

        self.movie_train_sub=self.movie_train.copy()
        self.movie_train_sub['genre'] = self.movie_train_sub.genre.str.split('|')
        self.movie_train_sub.reset_index(inplace=True)
        vectors_genre=[]
        for genre in self.movie_train_sub.genre.tolist():
            genre_vector = np.zeros(len(param_rating.genre2idx))
            for g in genre:
                genre_vector[param_rating.genre2idx[g]] = 1
            vectors_genre.append(genre_vector)
        self.movie_train_sub['genre_vectors']=vectors_genre

        genre_train = pd.DataFrame(self.movie_train_sub['genre_vectors'].tolist(), columns=param_rating.genre2idx.keys())
        self.movie_train_sub = pd.concat([self.movie_train_sub, genre_train], axis=1)
    def train_model(self):
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        print("Training...(final model)!")
        self.isTrained=True
        print("Input: User's rating ...........")
        self.model1=ModelByRating(self.movie_train,self.movie_test,self.user,self.rating)
        self.model1.preprocess_data()
        self.model1.train_model()
        self.model1.predict()
        self.model1.evaluate_model()
        print("Input: Title ...........")
        self.model2=ModelByTitle(self.movie_train,self.weight_path_title,self.data_test_cleaned,self.data_train_cleaned)
        self.model2.preprocess_data()
        self.model2.train_model()
        self.model2.predict(movies_test)
        self.model2.evaluate_model()
        print("Input: Image ...........")
        self.model3=ModelByImage(self.image_source,self.movie_train,self.movie_test,self.weight_path)
        self.model3.preprocessing()
        self.model3.train()
        self.model3.predict()
        self.model3.evaluate_model()
        submission1=pd.read_csv('./content/submission1.csv')
        submission2=pd.read_csv('./content/submission2.csv')
        submission3=pd.read_csv('./content/submission3.csv')
        missing_movie_ids1 = submission2[~submission2['movieid'].isin(submission1['movieid'])]
        for label in param_rating.genre2idx.keys():
            missing_movie_ids1[label]=0
        genre_movieid=list(param_rating.genre2idx.keys())
        genre_movieid.append('movieid')
        submission1 = pd.concat([submission1[genre_movieid], missing_movie_ids1[genre_movieid]], ignore_index=True).sort_values(by='movieid')
        missing_movie_ids3 = submission2[~submission2['movieid'].isin(submission3['movieid'])]
        for label in param_rating.genre2idx.keys():
            missing_movie_ids3[label]=0
        genre_movieid=list(param_rating.genre2idx.keys())
        genre_movieid.append('movieid')
        submission3 = pd.concat([submission3[genre_movieid], missing_movie_ids3[genre_movieid]], ignore_index=True).sort_values(by='movieid')
        submission2=submission2.sort_values(by='movieid')
        if self.onceTimeTrain==False:
            submission2=submission2.set_index('movieid')
            submission1=submission1.set_index('movieid')
            submission3=submission3.set_index('movieid')
            self.onceTimeTrain=True
        self.submission_train1=submission1.copy()
        self.submission_train2=submission2.copy()
        self.submission_train3=submission3.copy()
        data_sorted=self.movie_train_sub.sort_values(by='movieid').set_index('movieid')
        self.vectors_labels_train=pd.DataFrame(np.array(data_sorted[param_rating.genre2idx.keys()]))

        #Find Optimal w1,w2,w3
        params_bayes = {'w1': (0,1),
          'w2': (0,1),
           'w3':(0,1) }
        optimizer = BayesianOptimization(f = self.internal_method,
                                        pbounds = params_bayes,
                                        random_state = 7,
                                        verbose=2)
        optimizer.maximize(init_points=5, n_iter=100)
        optimal_value=optimizer.max['params']
        self.w1=optimal_value['w1']
        self.w2=optimal_value['w2']
        self.w3=optimal_value['w3']
    def __get_column_names_train(self,row):
        return list(self.vectors_labels_train.columns[row == 1])
    def internal_method(self,w1,w2,w3):
        if w1+w2+w3>1:
            return -1
        else:
            sorted_prediction__trainids = np.argsort(-(w1*self.submission_train1[param_rating.genre2idx.keys()]
                                                       +w2*self.submission_train2[param_rating.genre2idx.keys()]
                                                       +w3*self.submission_train1[param_rating.genre2idx.keys()]),axis=1)
            top_10_prediction_trainids = sorted_prediction__trainids[:,:5]
            vectors_labels_new=self.vectors_labels_train.apply(self.__get_column_names_train,axis=1).tolist()
            return map_at_k.mapk(vectors_labels_new,top_10_prediction_trainids,k=5)
    def predict(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        self.isPredict=True
        print("Predicting...(final model)!")
        submission1=self.model1.getFinalPrediction()
        submission2=self.model2.getFinalPrediction()
        submission3=self.model3.getFinalPrediction()
        missing_movie_ids1 = submission2[~submission2['movieid'].isin(submission1['movieid'])]
        for label in param_rating.genre2idx.keys():
            missing_movie_ids1[label]=0
        genre_movieid=list(param_rating.genre2idx.keys())
        genre_movieid.append('movieid')
        submission1 = pd.concat([submission1[genre_movieid], missing_movie_ids1[genre_movieid]], ignore_index=True).sort_values(by='movieid')

        missing_movie_ids3 = submission2[~submission2['movieid'].isin(submission3['movieid'])]
        for label in param_rating.genre2idx.keys():
            missing_movie_ids3[label]=0
        genre_movieid=list(param_rating.genre2idx.keys())
        genre_movieid.append('movieid')
        submission3 = pd.concat([submission3[genre_movieid], missing_movie_ids3[genre_movieid]], ignore_index=True).sort_values(by='movieid')

        submission2=submission2.sort_values(by='movieid')
        if self.onceTime==False:
            submission2=submission2.set_index('movieid')
            submission1=submission1.set_index('movieid')
            submission3=submission3.set_index('movieid')
            self.onceTime=True
        self.submission_final=self.w1*submission1[param_rating.genre2idx.keys()]+self.w2*submission2[param_rating.genre2idx.keys()]+self.w3*submission3[param_rating.genre2idx.keys()]
    def __get_column_names(self, row):
        return list(self.vectors_labels.columns[row == 1])
    def evaluate_model(self):
        print("Evaluating...(final model)")
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        if self.isPredict==False:
            self.predict()
        data_sorted=self.movie_test_sub.sort_values(by='movieid').set_index('movieid')
        self.vectors_labels=pd.DataFrame(np.array(data_sorted[param_rating.genre2idx.keys()]))
        sorted_prediction__trainids = np.argsort(-self.submission_final,axis=1)
        top_10_prediction_trainids = sorted_prediction__trainids[:,:5]
        vectors_labels_new=self.vectors_labels.apply(self.__get_column_names,axis=1).tolist()
        print(map_at_k.mapk(vectors_labels_new,top_10_prediction_trainids,k=5))
if __name__=='__main__':
    users = pd.read_csv('./content/dataset/users.dat', sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
    ratings = pd.read_csv('./content/dataset/ratings.dat', engine='python',
                            sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
    movies_train = pd.read_csv('./content/dataset/movies_train.dat', engine='python',
                            sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
    movies_test = pd.read_csv('./content/dataset/movies_test.dat', engine='python',
                            sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')                         
    
    weight_path_img = './content/cleaned_data/best_weights_32.h5'
    image_source = './content/dataset/ml1m-images'
    data_train_cleaned='./content/cleaned_data/movies_train.csv'
    data_test_cleaned='./content/cleaned_data/movies_test.csv'
    weight_path_title = './content/weight/model-fine-tune2.pth'
    
    finalmodel=FinalModel(movies_train,movies_test,users,ratings,image_source,weight_path_img,weight_path_title,data_test_cleaned,data_train_cleaned)
    finalmodel.preprocess_data()
    finalmodel.train_model()
    finalmodel.predict()
    finalmodel.evaluate_model()

    # model1=ModelByRating(movies_train,movies_test,users,ratings)
    # model1.preprocess_data()
    # model1.train_model()
    # model1.getValueByRating()
    # model1.predict()
    # model1.evaluate_model()

    # model2=ModelByTitle(movies_train)
    # model2.preprocess_data()
    # model2.train_model()
    # model2.getValueByTitle()
    # model2.predict(movies_test)
    # model2.evaluate_model()

    # model3=ModelByImage(image_source,movies_train,movies_test,weight_path_img)
    # model3.preprocessing()
    # model3.train()
    # model3.getValueByImage()
    # model3.predict()
    # model3.evaluate_model()

    # datasub=pd.DataFrame(pd.read_csv('./content/cleaned_data/movies_test.csv'))
    # weight_path_title = './content/weight/model-fine-tune2.pth'
    # model = BertModel(weight_path_title, max_len= 7)
    # res = model.predict(datasub)
    # print(np.array(res).shape)