import sys
import importlib
sys.path.insert(0,'./content/params')
sys.path.insert(1, './content/metrics/')
import pandas as pd
import numpy as np
import param_rating
import map_at_k
importlib.reload(param_rating)
importlib.reload(map_at_k)
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN
from sklearn.metrics import f1_score
class ModelByRating:
    def __init__(self,movie_train,movie_test,user,rating):
        self.movie_train=movie_train.copy()
        self.movie_test=movie_test.copy()
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
        self.vector_movieids=[]
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
                self.vector_movieids.append(int(self.data1_unique["movieid"].iloc[i]))

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
        self.vector_movieids_test=[]
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
                self.vector_movieids_test.append(int(self.data2_unique["movieid"].iloc[i]))

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
    def getValueByRating(self):
        if self.isTrained==False and self.isPreprocess==True:
            raise Exception('train_model() needs to be proceeded!')
        if self.isPreprocess==False:
            raise Exception('preprocess_data() needs to be proceeded first!')
        train_rating=pd.DataFrame()
        model_rating=pickle.load(open('./content/trained_model_params/modelByRating.pkl','rb'))
        for i in range(self.vectors_labels.shape[1]):
            y_pred_X = model_rating[i].predict_proba(self.vectors_train)[:,1]
            train_rating[list(param_rating.genre2idx.keys())[list(param_rating.genre2idx.values()).index(i)]]=y_pred_X
        train_rating["movieid"]=self.vector_movieids
        train_rating.to_csv('./content/submission1.csv')
        return train_rating
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
            self.submission_rating[list(param_rating.genre2idx.keys())[list(param_rating.genre2idx.values()).index(i)]]=y_pred_X
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
    def getFinalPrediction(self):
        submission_rating_sub=self.submission_rating.copy()
        submission_rating_sub['movieid']=self.vector_movieids_test
        return submission_rating_sub