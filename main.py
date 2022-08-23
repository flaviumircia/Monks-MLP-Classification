import pandas as pd
import sklearn.metrics
from sklearn.neural_network import MLPClassifier

#headers for the dataframe
headers=['class','a1','a2','a3','a4','a5','a6','data_id']

#get the files from the database already splitted in train and test
monks1_test=pd.read_csv("monks-1.test", sep=' ',names=headers)
monks1_train=pd.read_csv("monks-1.train", sep=' ',names=headers)

#dropping the unnecessary column "data_id"
monks1_train.drop("data_id",axis=1,inplace=True)
monks1_test.drop("data_id",axis=1,inplace=True)

monks2_test=pd.read_csv("monks-2.test", sep=' ',names=headers)
monks2_train=pd.read_csv("monks-2.train", sep=' ',names=headers)

#dropping the unnecessary column "data_id"
monks2_train.drop("data_id",axis=1,inplace=True)
monks2_test.drop("data_id",axis=1,inplace=True)

monks3_test=pd.read_csv("monks-3.test", sep=' ',names=headers)
monks3_train=pd.read_csv("monks-3.train",sep=' ',names=headers)

#dropping the unnecessary column "data_id"
monks3_train.drop("data_id",axis=1,inplace=True)
monks3_test.drop("data_id",axis=1,inplace=True)

#getting the class column from monks1
monks1_train_labels=monks1_train["class"]
monks1_test_labels=monks1_test["class"]

#dropping the column data from monks1
monks1_train.drop("class",inplace=True,axis=1)
monks1_test.drop("class",inplace=True,axis=1)

#getting the class column from monks2
monks2_train_labels=monks2_train["class"]
monks2_test_labels=monks2_test["class"]

#dropping the column data from monks2
monks2_train.drop("class",inplace=True,axis=1)
monks2_test.drop("class",inplace=True,axis=1)

#getting the class column from monks3
monks3_train_labels=monks3_train["class"]
monks3_test_labels=monks3_test["class"]

#dropping the column data from monks3
monks3_train.drop("class",inplace=True,axis=1)
monks3_test.drop("class",inplace=True,axis=1)

hidden_layers=[1,2]
learning_rate=[0.1,0.01]
# numbers on neurons = (initially) with numbers of attributes
neurons_on_hidden_layers=[7,3]
print("RESULTS FOR MONK 1 DATA SET: ")
for i in hidden_layers:
    for j in learning_rate:
        if i==1:
            for k in neurons_on_hidden_layers:
                model=MLPClassifier(hidden_layer_sizes=(k),learning_rate_init=j,max_iter=1000)
                model.fit(monks1_train,monks1_train_labels)
                predicted_labels=model.predict(monks1_test)
                print("Learning rate = ",j,", layers = ",i,", neurons on hidden layers = ",k, ", accuracy = ",sklearn.metrics.accuracy_score(monks1_test_labels,predicted_labels))
        else:
            for k in neurons_on_hidden_layers:
                for l in range(1,3):
                    k2=int(k/l)
                    model = MLPClassifier(hidden_layer_sizes=(k,k2), learning_rate_init=j, max_iter=1000)
                    model.fit(monks1_train, monks1_train_labels)
                    predicted_labels = model.predict(monks1_test)
                    print("Learning rate = ",j,", layers = ",i,", neurons on hidden layers = (",k,",",k2,"), accuracy = ",sklearn.metrics.accuracy_score(monks1_test_labels,predicted_labels))

print("\nRESULTS FOR MONK 2 DATA SET: ")

for i in hidden_layers:
    for j in learning_rate:
        if i==1:
            for k in neurons_on_hidden_layers:
                model=MLPClassifier(hidden_layer_sizes=(k),learning_rate_init=j,max_iter=2000)
                model.fit(monks2_train,monks2_train_labels)
                predicted_labels=model.predict(monks2_test)
                print("Learning rate = ",j,", layers = ",i,", neurons on hidden layers = ",k, ", accuracy = ",sklearn.metrics.accuracy_score(monks2_test_labels,predicted_labels))
        else:
            for k in neurons_on_hidden_layers:
                for l in range(1,3):
                    k2=int(k/l)
                    model = MLPClassifier(hidden_layer_sizes=(k,k2), learning_rate_init=j, max_iter=2000)
                    model.fit(monks2_train, monks2_train_labels)
                    predicted_labels = model.predict(monks2_test)
                    print("Learning rate = ",j,", layers = ",i,", neurons on hidden layers = (",k,",",k2,"), accuracy = ",sklearn.metrics.accuracy_score(monks2_test_labels,predicted_labels))

print("\nRESULTS FOR MONK 3 DATA SET: ")

for i in hidden_layers:
    for j in learning_rate:
        if i==1:
            for k in neurons_on_hidden_layers:
                model=MLPClassifier(hidden_layer_sizes=(k),learning_rate_init=j,max_iter=1000)
                model.fit(monks3_train,monks3_train_labels)
                predicted_labels=model.predict(monks3_test)
                print("Learning rate = ",j,", layers = ",i,", neurons on hidden layers = ",k, ", accuracy = ",sklearn.metrics.accuracy_score(monks3_test_labels,predicted_labels))
        else:
            for k in neurons_on_hidden_layers:
                for l in range(1,3):
                    k2=int(k/l)
                    model = MLPClassifier(hidden_layer_sizes=(k,k2), learning_rate_init=j, max_iter=1000)
                    model.fit(monks3_train, monks3_train_labels)
                    predicted_labels = model.predict(monks3_test)
                    print("Learning rate = ",j,", layers = ",i,", neurons on hidden layers = (",k,",",k2,"), accuracy = ",sklearn.metrics.accuracy_score(monks3_test_labels,predicted_labels))

