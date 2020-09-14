# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:10:34 2020

@author: elias
"""

def main():

    import torch
    import torch.nn as nn
    
    import random
    import string
    
    def generate_data(n_samples):
        filenames = [''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.wav' for i in range(n_samples)]
        labels = [random.choice(['HD', 'preHD', 'control']) for i in range(n_samples)]
    
        emb = {}
        for filename in filenames:
            factor = torch.randint(1,1000, (1,))
            nb_lines = torch.randint(100, 200, (1,))
            emb[filename] = factor * torch.randn((nb_lines, 512))
        
        return filenames, labels, emb
    
    filenames, labels, emb = generate_data(85)
    
    from sklearn.model_selection import train_test_split
    
    train_files, test_files, train_labels, test_labels = train_test_split(filenames, labels, test_size=0.2)
    
    def prepare_numerical_data(files, labels, embeddings):
        X = torch.tensor([])
        y = []
        mapping = {'HD' : 0, 'preHD' : 1, 'control' : 2}
        for i, filename in enumerate(files):
            add_rows = embeddings[filename] # torch.from_numpy(embeddings[filename])
            add_labels = [mapping[labels[i]]] * add_rows.shape[0]
            X = torch.cat((X, add_rows), 0)
            y += add_labels
        return X, torch.tensor(y)
    
    X_train, y_train = prepare_numerical_data(train_files, train_labels, emb)
    X_test, y_test = prepare_numerical_data(test_files, test_labels, emb)
    
    from imblearn.datasets import make_imbalance
    from collections import Counter
    
    def resample(X, y):
        n_ = Counter(y.tolist()).most_common()[-1][-1]
        newX, newY = make_imbalance(X, y, sampling_strategy={0: n_, 1: n_, 2: n_})
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        newX = torch.from_numpy(newX).type(dtype)
        newY = torch.from_numpy(newY)
        return newX, newY
    
    X_train, y_train = resample(X_train, y_train)
    
    class DNN(nn.Module):
        def __init__(self):
            super(DNN, self).__init__()
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 3)
            self.softmax = nn.Softmax(dim=-1)
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = self.softmax(self.fc2(x))
            return x
    
    net = DNN()
    
    import torch.optim as optim
    
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    for _ in range(epochs):
        optimizer.zero_grad()
        output = net(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    
    def predict(clf, files, embeddings):
        y_predict = []
        mapping = {0:'HD', 1:'preHD', 2:'control'}
        for filename in files:
            X_test = embeddings[filename]
            voting_y_predict = torch.argmax(clf(X_test), dim=1).tolist()
            int_mode = Counter(voting_y_predict).most_common()[0][0]
            label = mapping[int_mode]
            y_predict.append(label)
        return y_predict
    
    predict_labels = predict(net, test_files, emb)
    
    from sklearn.metrics import classification_report
    
    print(' \n classification report : \n') 
    print(classification_report(test_labels, predict_labels))
    
    
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame
    
    cols = ['HD predicted', 'control predicted', 'preHD predicted']
    idx = ['HD True', 'control True', 'preHD True']
    
    mat = confusion_matrix(test_labels, predict_labels, labels=['HD', 'control', 'preHD'])
    mat = DataFrame(mat, columns=cols, index=idx)
    print(' \n confusion matrix : \n') 
    print(mat)
    
    y_predict = torch.argmax(net(X_test), dim=1)
    
    from pandas import Series
    
    def reverse_transform(y):
        mapping = {0:'HD', 1:'preHD', 2:'control'}
        map_func = lambda integer : mapping[integer]
        y_list = Series(y.tolist()).apply(map_func).tolist()
        return y_list
    
    y_test = reverse_transform(y_test)
    y_predict = reverse_transform(y_predict)
        
    mat = confusion_matrix(y_test, y_predict, labels=['HD', 'control', 'preHD'])
    mat = DataFrame(mat, columns=cols, index=idx)
    print(' \n confusion matrix for embeddings : \n') 
    print(mat)

if __name__ == '__main__':
    main()


