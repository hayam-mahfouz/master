# author: yasser hifny


#!pip install nilearn
import glob
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

#sklearn - basic ML tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn import metrics

input_path = '/lfs01/workdirs/hlwn030u1/abide_dataset/ABIDE_pcp/cpac/nofilt_noglobal'

def split(out_path, fold):
    train_file = "%s/train_ds%d.csv" % (input_path, fold)
    dev_file = "%s/dev_ds%d.csv" % (input_path, fold)
    
    metadata_df = pd.read_csv(train_file, sep=",", header=None, quoting=3)
    metadata_df.columns = ["file_name", "label"]
    df_train = metadata_df[["file_name", "label"]]

    metadata_df = pd.read_csv(dev_file, sep=",", header=None, quoting=3)
    metadata_df.columns = ["file_name", "label"]
    df_val = metadata_df[["file_name", "label"]]


   
    df_all_data = pd.concat([df_train, df_val])
    
    site_dic = defaultdict(list)
    
    for index, row in df_all_data.iterrows():
        site = os.path.basename(row['file_name']).split('_')[0]
        site_dic[site].append((row['file_name'], row['label']))
    
    print (site_dic.keys())
    print (len(site_dic['CMU']))

    for site in ['NYU', 'USM', 'UM', 'Olin', 'Pitt', 'SDSU', 'Stanford', 'Trinity', 'Caltech', 'Leuven', 'Yale', 'MaxMun', 'UCLA', 'KKI', 'SBL', 'OHSU']:
        print(site)
        print (len(site_dic[site]))
        print (site_dic[site])
        
        X = np.array([i[0] for i in site_dic[site]])
        y = np.array([i[1] for i in site_dic[site]])        
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=1234)
        
        k=0
        results_train   = []
        results_test    = []
        for train_index, test_index in skf.split(X, y):
            k = k+1
            print (k)

            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("X_train")
            print(X_train, y_train)
            print("X_test")
            print(X_test, y_test)            
            file = open(out_path+'/train_ds%s.csv' % k, 'a')
            for i,j in zip(X_train, y_train):
                file.write(f'{i},{j}\n')
            file.close()    
            file = open(out_path+'/dev_ds%s.csv' % k, 'a')
            for i,j in zip(X_test, y_test):
                file.write(f'{i},{j}\n')
            file.close()    

    for site in ['CMU']:
        print(site)
        print (len(site_dic[site]))
        print (site_dic[site])
        
        X = np.array([i[0] for i in site_dic[site]])
        y = np.array([i[1] for i in site_dic[site]])        
        skf = StratifiedKFold(n_splits=2, shuffle = True, random_state=1234)
        
        k=0
        results_fold_train   = []
        results_fold_dev    = []
        for train_index, test_index in skf.split(X, y):
            k = k+1
            print (k)

            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("X_train")
            print(X_train, y_train)
            print("X_test")
            print(X_test, y_test)
            results_fold_train.append((X_train,y_train))
            results_fold_dev.append((X_test,y_test))
        
        print (len(results_fold_train))
        print (len(results_fold_dev))
        
        for w in range(10):
            w= w +1
            print(w, w%2)
            file = open(out_path+'/train_ds%s.csv' % w, 'a')
            print ((results_fold_train))
            print ((results_fold_dev))
            
            for i,j in zip(results_fold_train[w%2][0], results_fold_train[w%2][1]):
                print (i,j)
                file.write(f'{i},{j}\n')
            file.close()    
            file = open(out_path+'/dev_ds%s.csv' % w, 'a')
            for i,j in zip(results_fold_dev[w%2][0],results_fold_dev[w%2][1]):
                print (i,j)
                file.write(f'{i},{j}\n')
            file.close()


out_path='/lfs01/workdirs/hlwn030u1/abide_dataset/ABIDE_pcp/cpac/nofilt_noglobal'
split(out_path,1)
