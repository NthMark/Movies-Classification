import sys
import importlib
sys.path.insert(0,'./content/params')
sys.path.insert(1, './content/metrics/')
sys.path.insert(2,'./content/model_class')
from bert_model import BertModel
import param_rating
import map_at_k
importlib.reload(param_rating)
importlib.reload(map_at_k)
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from nltk import wordpunct_tokenize
import re
import matplotlib.pyplot as plt
import neural_metrics
from scipy.sparse import csr_matrix, hstack
from tensorflow import keras
from keras.metrics import *
from keras import layers, models
from keras.models import load_model

class ModelByTitle:
    def __init__(self,movie_train,weight_path_title,data_test_cleaned,data_train_cleaned) :
        self.movie_train=movie_train.copy()
        self.movie_train['genre'] = self.movie_train.genre.str.split('|')
        self.movie_test=None
        self.submission_combined=None
        self.vectors_labels=None
        self.weight_path_title=weight_path_title
        self.data_test_cleaned=data_test_cleaned
        self.data_train_cleaned=data_train_cleaned
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
    
    def getValueByTitle(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        log_smote=pickle.load(open('./content/trained_model_params/log_borderlineSMOTEmodelByTitle.pkl', 'rb'))
        log_smoteenn=pickle.load(open('./content/trained_model_params/log_borderlineSMOTEENNmodelByTitle.pkl', 'rb'))
        boosting=pickle.load(open('./content/trained_model_params/gradientboostingbytitle.pkl', 'rb'))
        neural=load_model('./content/trained_model_params/neuralbytitle.h5', custom_objects={'f1_m': neural_metrics.f1_m, 
                                                               'precision_m': neural_metrics.precision_m, 
                                                               'recall_m': neural_metrics.recall_m})
        log_chains=pickle.load(open('./content/trained_model_params/log_chainsByTitle.pkl', 'rb'))
        submission_binary = pd.DataFrame(columns=param_rating.genre2idx.keys())
        submission_binary_combined = pd.DataFrame(columns=param_rating.genre2idx.keys())
        submission_boost = pd.DataFrame(columns=param_rating.genre2idx.keys())
        submission_chains = pd.DataFrame(columns=param_rating.genre2idx.keys())
        change_namelabel=dict([(value,key) for key,value in param_rating.genre2idx.items()])
        #Bert Model
        datasub=pd.DataFrame(pd.read_csv(self.data_train_cleaned))
        model = BertModel(self.weight_path_title, max_len= 7)
        res = model.predict(datasub)
        res=pd.DataFrame(res)
        res.rename(columns=change_namelabel,inplace=True)
        #Logistic Regression with BorderlineSMOTE algorithm
        for label in param_rating.genre2idx.keys():
            test_y_prob = log_smoteenn[param_rating.genre2idx[label]].predict_proba(self.x_train)[:,1]
            submission_binary[label] = test_y_prob
        #Logistic Regression with SMOTEENN algorithm 
        for label in param_rating.genre2idx.keys():
            test_y_prob = log_smote[param_rating.genre2idx[label]].predict_proba(self.x_train)[:,1]
            submission_binary_combined[label] = test_y_prob
        #Gradient Boosting    
        for label in param_rating.genre2idx.keys():
            test_y_prob = boosting[param_rating.genre2idx[label]].predict_proba(self.x_train)[:,1]
            submission_boost[label] = test_y_prob
        #Neural network
        submission_neural = pd.DataFrame(neural.predict(self.x_train))
        submission_neural.rename(columns = change_namelabel, inplace = True)
        x_train_classfier=self.x_train.copy()
        for label in param_rating.genre2idx.keys():
            train_y = log_chains[param_rating.genre2idx[label]].predict(x_train_classfier)
            train_y_prob = log_chains[param_rating.genre2idx[label]].predict_proba(x_train_classfier)[:,1]
            submission_chains[label] = train_y_prob
            x_train_classfier = self.__add_feature(x_train_classfier, train_y)
        submission_title=(submission_binary_combined[param_rating.genre2idx.keys()]
                        +submission_binary[param_rating.genre2idx.keys()]
                        +submission_boost[param_rating.genre2idx.keys()]
                        +submission_neural[param_rating.genre2idx.keys()]
                        +submission_chains[param_rating.genre2idx.keys()]
                        +res[param_rating.genre2idx.keys()])/6 
        submission_title['movieid']=self.movie_train['movieid']
        submission_title.to_csv('./content/submission2.csv')
        return submission_title
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
        change_namelabel=dict([(value,key) for key,value in param_rating.genre2idx.items()])
        #Bert Model
        datasub=pd.DataFrame(pd.read_csv(self.data_test_cleaned))
        model = BertModel(self.weight_path_title, max_len= 7)
        res = model.predict(datasub)
        res=pd.DataFrame(res)
        res.rename(columns=change_namelabel,inplace=True)
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
        submission_neural = pd.DataFrame(neural.predict(self.x_test))
        submission_neural.rename(columns = change_namelabel, inplace = True)
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
        self.submission_combined=(submission_binary_combined[param_rating.genre2idx.keys()]
                                +submission_binary[param_rating.genre2idx.keys()]
                                +submission_boost[param_rating.genre2idx.keys()]
                                +submission_neural[param_rating.genre2idx.keys()]
                                +submission_chains[param_rating.genre2idx.keys()]
                                +res[param_rating.genre2idx.keys()])/6 #+submission_svm[param_rating.genre2idx.keys()]
        
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
    def getFinalPrediction(self):
        submission_combined_sub=self.submission_combined
        submission_combined_sub['movieid']=self.movie_test['movieid']
        return submission_combined_sub