import sys
import importlib
sys.path.insert(0,'./content/params')
sys.path.insert(1, './content/metrics/')
import param_rating
import map_at_k
importlib.reload(param_rating)
importlib.reload(map_at_k)
import pickle

from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler,MultiLabelBinarizer, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import cv2
import os
from nltk import wordpunct_tokenize
import re
import matplotlib.pyplot as plt
import neural_metrics
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

from keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import *
from tensorflow.keras import layers, models
from keras.models import load_model
from keras.optimizers import Adam

class ModelByRating:
    def __init__(self,movie_train,movie_test,user,rating):
        self.movie_train=movie_train
        self.movie_test=movie_test
        self.movie_train['genre'] = self.movie_train.genre.str.split('|')
        self.movie_test['genre'] = self.movie_test.genre.str.split('|')
        self.user=user
        self.rating=rating
        df_1 = pd.merge(self.movie_train,rating,how='inner',on='movieid')
        df_2 = pd.merge(self.movie_test,rating,how='inner',on='movieid')
        self.data1=pd.merge(df_1,user,how='inner',on='userid')
        self.data2 = pd.merge(df_2,user,how='inner',on='userid')
        self.data1_unique=self.movie_train.copy()
        self.data1_unique.reset_index(inplace=True)
        self.data2_unique=self.movie_test.copy()
        self.data2_unique.reset_index(inplace=True)
        self.dataMapping={
            "M":1,
            "F":2
        }
        #Condition
        self.isTrained=False
        self.isPredict=False
        self.isPreprocess=False
    def __preprocess_data_train(self):
        # Convert gender data into numeric values
        self.data1["gender"]=self.data1["gender"].map(self.dataMapping)
        
        self.x_train=self.data1.drop_duplicates(subset="userid",ignore_index=True)
        self.x_train=self.x_train.drop(["movieid","title","timestamp","zip"],axis=1)
        
        vectors_genre=[]
        for genre in self.data1_unique.genre.tolist():
            genre_vector = np.zeros(len(param_rating.genre2idx))
            for g in genre:
                genre_vector[param_rating.genre2idx[g]] = 1
            vectors_genre.append(genre_vector)
        self.data1_unique['genre_vectors']=vectors_genre

        self.vectors_genre1=[]
        self.vectors_labels=[]
        tree = KDTree(self.x_train[param_rating.list_train], leaf_size=2) 
        for i in range(len(self.data1_unique)):
            data1_each=self.data1[self.data1["movieid"]==int(self.data1_unique["movieid"].iloc[i])]
            data1_each=data1_each[(data1_each["rating"]==5) | (data1_each["rating"]==4) ]
            if len(data1_each)!=0:##get movies having rating
                _, ind_train = tree.query(data1_each[param_rating.list_train], k=5)
                x_train_new1=self.x_train.iloc[np.ravel(ind_train)]
                x_train_new_unique1=x_train_new1.drop_duplicates(subset="userid",ignore_index=True)
                x_train_final1=self.data1[self.data1["userid"].isin(x_train_new_unique1["userid"])] 
                x_train_final_high1=x_train_final1[(x_train_final1["rating"]==5) | (x_train_final1["rating"]==4)]
                genre_vector1 = np.zeros(len(param_rating.genre2idx))
                for genre in x_train_final_high1.genre.tolist():
                    for g in genre:
                        genre_vector1[param_rating.genre2idx[g]] += 1
                self.vectors_genre1.append(genre_vector1.astype(int))
                self.vectors_labels.append(self.data1_unique["genre_vectors"].iloc[i].tolist())
        
        self.vectors_labels=pd.DataFrame(self.vectors_labels)
        vectors_genre_table=pd.DataFrame(self.vectors_genre1)
        scaler=MinMaxScaler()
        self.vectors_train=scaler.fit_transform(vectors_genre_table)
    def __preprocess_data_test(self):
        self.data2["gender"]=self.data2["gender"].map(self.dataMapping)
        vectors_genre=[]
        for genre in self.data2_unique.genre.tolist():
            genre_vector = np.zeros(len(param_rating.genre2idx))
            for g in genre:
                genre_vector[param_rating.genre2idx[g]] = 1
            vectors_genre.append(genre_vector)
        self.data2_unique['genre_vectors']=vectors_genre

        self.vectors_genre_test=[]
        self.vectors_labels_test=[]
        tree = KDTree(self.x_train[param_rating.list_train], leaf_size=2) 
        for i in range(len(self.data2_unique)):
            data2_each=self.data2[self.data2["movieid"]==int(self.data2_unique["movieid"].iloc[i])]
            data2_each=data2_each[(data2_each["rating"]==5) | (data2_each["rating"]==4) ]
            if len(data2_each)!=0:##get movies having rating
                _, ind_train = tree.query(data2_each[param_rating.list_train], k=5)
                x_train_new=self.x_train.iloc[np.ravel(ind_train)]
                x_train_new_unique=x_train_new.drop_duplicates(subset="userid",ignore_index=True)
                x_train_final=self.data1[self.data1["userid"].isin(x_train_new_unique["userid"])] 
                x_train_final_high=x_train_final[(x_train_final["rating"]==5) | (x_train_final["rating"]==4)]
                genre_vector = np.zeros(len(param_rating.genre2idx))
                for genre in x_train_final_high.genre.tolist():
                    for g in genre:
                        genre_vector[param_rating.genre2idx[g]] += 1
                self.vectors_genre_test.append(genre_vector.astype(int))
                self.vectors_labels_test.append(self.data2_unique["genre_vectors"].iloc[i].tolist())
        
        self.vectors_labels_test=np.array(self.vectors_labels_test)
        self.vectors_labels_test=pd.DataFrame(self.vectors_labels_test)
        scaler_test=MinMaxScaler()
        self.vectors_test=scaler_test.fit_transform(self.vectors_genre_test)
    def preprocess_data(self):
        self.isPreprocess=True
        print("Preprocessing...")
        self.__preprocess_data_train()
        self.__preprocess_data_test()
    def train_model(self):
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        self.isTrained=True
        print("Training...")
        model_rating=[]
        hype_param_best=[]
            
        for label in range(self.vectors_labels.shape[1]):
            logreg = LogisticRegression(C=0.5)
            smote=BorderlineSMOTE(random_state=27,k_neighbors=5)
            smote_x_train,smote_y_train=smote.fit_resample(self.vectors_train,self.vectors_labels[label])
            print('... Processing {}'.format(list(param_rating.genre2idx.keys())[list(param_rating.genre2idx.values()).index(label)]))
            
            logreg.fit(smote_x_train, smote_y_train)
            model_rating.append(logreg)
            
            y_pred_X = logreg.predict_proba(smote_x_train)[:,1]
            y_pred_X1 = logreg.predict(smote_x_train)
            print('Training accuracy(before) is {}'.format(f1_score(smote_y_train, y_pred_X1)))
            hype_param=0
            interval=0.01
            maxScore=0
            for _ in range(990):
                interval+=0.001
                y_pred_new=(y_pred_X>=interval).astype(int)
                temp_score=f1_score(smote_y_train,y_pred_new)
                if temp_score>maxScore:
                    maxScore=temp_score
                    hype_param=interval
            print(f"{label}: maxScore: {maxScore} best T:{hype_param}")
            hype_param_best.append(hype_param)
            print('Training accuracy(after) is {}'.format(f1_score(smote_y_train, (y_pred_X>=hype_param).astype(int))))

        with open('./content/trained_model_params/modelByRating.pkl', 'wb') as file:
            pickle.dump(model_rating, file)
        with open('./content/trained_model_params/hyperparamByRating.pkl', 'wb') as file:
            pickle.dump(hype_param_best, file)
    def predict(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        self.isPredict=True
        print("Predicting...")
        self.submission_rating=pd.DataFrame()
        hype_param_best=pickle.load(open('./content/trained_model_params/hyperparamByRating.pkl', 'rb'))
        model_rating=pickle.load(open('./content/trained_model_params/modelByRating.pkl','rb'))
        for i in range(self.vectors_labels_test.shape[1]):
            y_pred_X = model_rating[i].predict_proba(self.vectors_test)[:,1]
            self.submission_rating[i]=y_pred_X
            print('Training accuracy is {}'.format(f1_score(self.vectors_labels_test[i], (y_pred_X>=hype_param_best[i]).astype(int))))
    def get_column_names(self,row):
        return list(self.vectors_labels_test.columns[row == 1])
    def evaluate_model(self):
        print("Evaluating...")
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        if self.isPredict==False:
            self.predict()
        sorted_prediction_ids = np.argsort(-self.submission_rating,axis=1)
        top_5_prediction_ids = sorted_prediction_ids[:,:5]
        vectors_labels_test_new=self.vectors_labels_test.apply(self.get_column_names,axis=1).tolist()
        print(map_at_k.mapk(vectors_labels_test_new,top_5_prediction_ids,k=5))

class ModelByTitle:
    def __init__(self,movie_train) :
        self.movie_train=movie_train.copy()
        self.movie_train['genre'] = self.movie_train.genre.str.split('|')
        self.movie_test=None
        self.submission_combined=None
        self.vectors_labels=None
        #Condition
        self.isTrained=False
        self.isPredict=False
        self.isPreprocess=False
    def tokenize(self,text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = wordpunct_tokenize(text)
        tokens = tokens[:-1] # remove last token because it is the year which maybe is not useful
        return tokens

    def create_vocab(self):
        df = self.movie_train.copy()
        arr_title = df['title'].tolist()
        vocab = set()
        for title in arr_title:
            tokens = self.tokenize(title)
            vocab.update(tokens)
        vocab = list(vocab)
        pad_token = '<PAD>'
        unk_token = '<UNK>'
        vocab.append(pad_token)
        vocab.append(unk_token)
        return vocab

    def __preprocess_data_train(self):
        self.movie_train.reset_index(inplace=True)
        vocab=self.create_vocab()
        self.movie_train['title_tokens'] = [self.tokenize(x) for x in self.movie_train.title]
        # create vocab
        pad_token = '<PAD>'
        unk_token = '<UNK>'
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        # Create a binary vector for each word in each sentence
        MAX_LENGTH = 7
        vectors = []
        for title_tokens in self.movie_train.title_tokens.tolist():
            if len(title_tokens) < MAX_LENGTH:
                num_pad = MAX_LENGTH - len(title_tokens)
                for _ in range(num_pad):
                    title_tokens.append(pad_token)
            else:
                title_tokens = title_tokens[:MAX_LENGTH]
            title_vectors = np.zeros(len(vocab))
            for word in title_tokens:
                if word in vocab:
                    title_vectors[token2idx[word]] = 1
                else:
                    title_vectors[token2idx[unk_token]] = 1

            vectors.append(np.array(title_vectors))
        self.movie_train['vectors'] = vectors 

        #preprocess label
        vectors_genre=[]
        for genre in self.movie_train.genre.tolist():
            genre_vector = np.zeros(len(param_rating.genre2idx))
            for g in genre:
                genre_vector[param_rating.genre2idx[g]] = 1
            vectors_genre.append(genre_vector)
        self.movie_train['genre_vectors']=vectors_genre

        genre_df = pd.DataFrame(self.movie_train['genre_vectors'].tolist(), columns=param_rating.genre2idx.keys())
        self.movie_train = pd.concat([self.movie_train, genre_df], axis=1)  

        self.x_train=np.expand_dims(self.movie_train['vectors'], 0)
        self.x_train=np.vstack(np.ravel(np.ravel(self.x_train))) 
        self.y_train=np.expand_dims(self.movie_train['genre_vectors'], 0)
        self.y_train=np.vstack(np.ravel(np.ravel(self.y_train))) 

        
    def __preprocess_data_test(self,movie_test):
        self.movie_test=movie_test.copy()
        self.movie_test['genre'] = self.movie_test.genre.str.split('|')
        self.movie_test.reset_index(inplace=True)
        self.movie_test['title_tokens'] = [self.tokenize(x) for x in self.movie_test.title]
        # create vocab
        vocab=self.create_vocab()

        pad_token = '<PAD>'
        unk_token = '<UNK>'
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        # Create a binary vector for each word in each sentence
        MAX_LENGTH = 7
        vectors = []
        for title_tokens in self.movie_test.title_tokens.tolist():
            if len(title_tokens) < MAX_LENGTH:
                num_pad = MAX_LENGTH - len(title_tokens)
                for _ in range(num_pad):
                    title_tokens.append(pad_token)
            else:
                title_tokens = title_tokens[:MAX_LENGTH]
            title_vectors = np.zeros(len(vocab))
            for word in title_tokens:
                if word in vocab:
                    title_vectors[token2idx[word]] = 1
                else:
                    title_vectors[token2idx[unk_token]] = 1

            vectors.append(np.array(title_vectors))
        self.movie_test['vectors'] = vectors 

        #preprocess label
        vectors_genre=[]
        for genre in self.movie_test.genre.tolist():
            genre_vector = np.zeros(len(param_rating.genre2idx))
            for g in genre:
                genre_vector[param_rating.genre2idx[g]] = 1
            vectors_genre.append(genre_vector)
        self.movie_test['genre_vectors']=vectors_genre

        genre_test = pd.DataFrame(self.movie_test['genre_vectors'].tolist(), columns=param_rating.genre2idx.keys())
        self.movie_test = pd.concat([self.movie_test, genre_test], axis=1)

        self.x_test=np.expand_dims(self.movie_test['vectors'], 0)
        self.x_test=np.vstack(np.ravel(np.ravel(self.x_test))) 
        self.y_test=np.expand_dims(self.movie_test['genre_vectors'], 0)
        self.y_test=np.vstack(np.ravel(np.ravel(self.y_test))) 
    def preprocess_data(self):
        self.isPreprocess=True
        print("Preprocessing...")
        self.__preprocess_data_train()
    def train_model(self):
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        self.isTrained=True
        print("Training...")
        #Logistic Regression with BorderlineSMOTE algorithm
        logreg_list=[]
        for label in param_rating.genre2idx.keys():
            logreg = LogisticRegression(C=1.56)
            smote=BorderlineSMOTE(random_state=27,k_neighbors=5)
            smote_x_train,smote_y_train=smote.fit_resample(self.x_train,self.movie_train[label])
            print('... Processing {}'.format(label))

            logreg.fit(smote_x_train, smote_y_train)
            logreg_list.append(logreg)
            y_pred_X = logreg.predict(smote_x_train)
            print('Training accuracy is {}'.format(f1_score(smote_y_train, y_pred_X)))

        
        with open('./content/trained_model_params/log_borderlineSMOTEmodelByTitle.pkl','wb') as file:
            pickle.dump(logreg_list,file)

        #Logistic Regression with SMOTEENN algorithm 
        logreg_list.clear()
        for label in param_rating.genre2idx.keys():
            logreg = LogisticRegression(C=1.44)
            smote=SMOTEENN(random_state=27)
            smote_x_train,smote_y_train=smote.fit_resample(self.x_train,self.movie_train[label])
            print('... Processing {}'.format(label))

            logreg.fit(smote_x_train, smote_y_train)
            logreg_list.append(logreg)
            y_pred_X = logreg.predict(smote_x_train)
            print('Training accuracy is {}'.format(f1_score(smote_y_train, y_pred_X)))
        with open('./content/trained_model_params/log_borderlineSMOTEENNmodelByTitle.pkl','wb') as file:
            pickle.dump(logreg_list,file)
        
        #Gradient Boosting
        logreg_list.clear()
        for label in param_rating.genre2idx.keys():
            logreg = XGBClassifier(max_depth=6, learning_rate=1e-2) 
            smote=SMOTEENN(random_state=27)
            smote_x_train,smote_y_train=smote.fit_resample(self.x_train,self.movie_train[label])
            print('... Processing {}'.format(label))
            X_train, X_validation, Y_train, Y_validation = train_test_split(smote_x_train, 
                                                              smote_y_train, 
                                                              test_size=0.25)
            logreg.fit(X_train, Y_train, eval_metric="logloss", eval_set=[(X_validation, Y_validation)], early_stopping_rounds=10, verbose=True)
            
            logreg_list.append(logreg)
            y_pred_X = logreg.predict(smote_x_train)
            print('Training accuracy is {}'.format(f1_score(smote_y_train, y_pred_X)))
        with open('./content/trained_model_params/gradientboostingbytitle.pkl','wb') as file:
            pickle.dump(logreg_list,file)

        #Neural network
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.x_train.shape[1],)),
            layers.Dropout(0.8),
            layers.Dense(300, activation='relu'),
            layers.Dropout(0.8),
            layers.Dense(len(param_rating.genre2idx), activation='sigmoid')  # Use 'sigmoid' for multi-label classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',neural_metrics.f1_m,neural_metrics.precision_m, neural_metrics.recall_m])
        model.fit(
            self.x_train, self.y_train,
            epochs=50,  
            batch_size=200
        )
        model.save('./content/trained_model_params/neuralbytitle.h5')

        #SVM
            
        # svm_model=SVC(kernel='rbf', C=0.01, gamma=0.5385, probability=True)
        # multilabel_classifier = MultiOutputClassifier(svm_model, n_jobs=-1)
        # multilabel_classifier = multilabel_classifier.fit(self.x_train, self.y_train)
        # with open('./content/trained_model_params/svm_modelByTitle.pkl', 'wb') as file:
        #     pickle.dump(multilabel_classifier, file)
        
        
        #Classifier Chains Technique for logistic regression
        logreg_list.clear()
        data_classifier=self.movie_train.copy()
        x_train_classfier=self.x_train.copy()
        for label in param_rating.genre2idx.keys():
            logreg_classifier = LogisticRegression(C=1.44)
            print('... Processing {}'.format(label))
            y = data_classifier[label]
            logreg_classifier.fit(x_train_classfier, y)
            logreg_list.append(logreg_classifier)
            y_pred_X = logreg_classifier.predict(x_train_classfier)
            print('Training accuracy is {}'.format(f1_score(y, y_pred_X)))
            x_train_classfier=self.__add_feature(x_train_classfier,y)
        with open('./content/trained_model_params/log_chainsByTitle.pkl', 'wb') as file:
            pickle.dump(logreg_list, file)

    def __add_feature(self,X, feature_to_add):
            return hstack([X, csr_matrix(feature_to_add).T], 'csr')
    
    def predict(self,movie_test):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        self.isPredict=True
        print("Predicting...")
        self.__preprocess_data_test(movie_test)
        log_smote=pickle.load(open('./content/trained_model_params/log_borderlineSMOTEmodelByTitle.pkl', 'rb'))
        log_smoteenn=pickle.load(open('./content/trained_model_params/log_borderlineSMOTEENNmodelByTitle.pkl', 'rb'))
        boosting=pickle.load(open('./content/trained_model_params/gradientboostingbytitle.pkl', 'rb'))
        # neural=pickle.load(open('./content/trained_model_params/neuralbytitle.pkl', 'rb'))
        neural=load_model('./content/trained_model_params/neuralbytitle.h5', custom_objects={'f1_m': neural_metrics.f1_m, 
                                                               'precision_m': neural_metrics.precision_m, 
                                                               'recall_m': neural_metrics.recall_m})
        
        # svm=pickle.load(open('./content/trained_model_params/svm_modelByTitle.pkl', 'rb'))

        log_chains=pickle.load(open('./content/trained_model_params/log_chainsByTitle.pkl', 'rb'))
        submission_binary = pd.DataFrame(columns=param_rating.genre2idx.keys())
        submission_binary_combined = pd.DataFrame(columns=param_rating.genre2idx.keys())
        submission_boost = pd.DataFrame(columns=param_rating.genre2idx.keys())

        # submission_svm = pd.DataFrame(columns=param_rating.genre2idx.keys())

        submission_chains = pd.DataFrame(columns=param_rating.genre2idx.keys())
        #Logistic Regression with BorderlineSMOTE algorithm
        for label in param_rating.genre2idx.keys():
            test_y_prob = log_smoteenn[param_rating.genre2idx[label]].predict_proba(self.x_test)[:,1]
            submission_binary[label] = test_y_prob
        #Logistic Regression with SMOTEENN algorithm 
        for label in param_rating.genre2idx.keys():
            test_y_prob = log_smote[param_rating.genre2idx[label]].predict_proba(self.x_test)[:,1]
            submission_binary_combined[label] = test_y_prob
        #Gradient Boosting    
        for label in param_rating.genre2idx.keys():
            test_y_prob = boosting[param_rating.genre2idx[label]].predict_proba(self.x_test)[:,1]
            submission_boost[label] = test_y_prob
        #Neural network
        submission_neural = neural.predict(self.x_test)
        self.test_neural=submission_neural
        #SVM
        # y_test_pred = svm.predict_proba(self.x_test)
        # y_test_pred_svm=np.array(y_test_pred.copy())
        # submission_svm = pd.DataFrame(columns=param_rating.genre2idx.keys())
        # for label in param_rating.genre2idx.keys():
        #     submission_svm[label]=y_test_pred_svm[param_rating.genre2idx[label]][:,1]

        #Classifier Chains Technique for logistic regression
        x_test_classfier=self.x_test.copy()
        for label in param_rating.genre2idx.keys():
            test_y = log_chains[param_rating.genre2idx[label]].predict(x_test_classfier)
            test_y_prob = log_chains[param_rating.genre2idx[label]].predict_proba(x_test_classfier)[:,1]
            submission_chains[label] = test_y_prob
            x_test_classfier = self.__add_feature(x_test_classfier, test_y)
        self.submission_combined=(submission_binary_combined[param_rating.genre2idx.keys()]+submission_binary[param_rating.genre2idx.keys()]+submission_boost[param_rating.genre2idx.keys()]+submission_neural
                     +submission_chains[param_rating.genre2idx.keys()])/5 #+submission_svm[param_rating.genre2idx.keys()]
    def __get_column_names(self,row):
        return list(self.vectors_labels.columns[row == 1])
    def evaluate_model(self):
        print("Evaluating...")
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        if self.isPredict==False:
            self.predict()
        self.vectors_labels=pd.DataFrame(np.array(self.movie_test[param_rating.genre2idx.keys()]))
        sorted_prediction_ids = np.argsort(-self.submission_combined,axis=1)
        top_5_prediction_ids = sorted_prediction_ids[:,:5]
        vectors_labels_test_new=self.vectors_labels.apply(self.__get_column_names,axis=1).tolist()
        print(map_at_k.mapk(vectors_labels_test_new,top_5_prediction_ids,k=5))       


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

        new_file_train_path = './content/dataset_cleaned/movies_train_update_in_code.DAT'
        self.movies_train_update.to_csv(new_file_train_path, sep=',', encoding='latin-1', index=False, header=False)
        new_file_test_path = './content/dataset_cleaned/movies_test_update_in_code.DAT'
        self.movies_test_update.to_csv(new_file_test_path, sep=',', encoding='latin-1', index=False, header=False)

        self.movies_train = pd.read_csv ( './content/dataset_cleaned/movies_train_update_in_code.DAT', engine='python', sep=',',
                                              names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False )
        self.movies_test = pd.read_csv ( './content/dataset_cleaned/movies_test_update_in_code.DAT', engine='python', sep=',',
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
    def predict(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocessing() needs to be proceeded first!')
        self.isPredict=True
        print("Predicting...")
        self.y_pred = self.model.predict(self.x_test)
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
    # movies_train['genre'] = movies_train.genre.str.split('|')
    # movies_test['genre'] = movies_test.genre.str.split('|')
    model1=ModelByRating(movies_train,movies_test,users,ratings)
    model1.preprocess_data()
    model1.train_model()
    model1.predict()
    model1.evaluate_model()
    # model2=ModelByTitle(movies_train)
    # model2.preprocess_data()
    # model2.train_model()
    # model2.predict(movies_test)
    # model2.evaluate_model()