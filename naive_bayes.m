
clc

clear 

close all

rng(0)

%% dataset
T=readtable('archive/all_currencies.csv');

allcurrecies = table2cell(T(:,3));  % extract all the currencies

currencies =  unique(allcurrecies);  % compute the unique currencies 

trend_direction = 'upward';        % specify the trend direction

window  = 7;  % Number of days to use for prediction
    
AUC =[];
for c =1:length(currencies)   % loop through the number of currencies 
    
    tic;
    
    TT=T(strcmp(allcurrecies,allcurrecies(c)),:);  % extract subset data for each currency 

    close_data = TT.Close;  % extract the closing data
    
    %% extracting the dataset for training 
    data  =  [];  % create an empty dataset

    for i = 1:window:length(close_data)-2*window  % looping through windows
        
        if strcmp(trend_direction,'upward')
            trend = close_data(i)>close_data(i+window);    % upward trend 
            
        elseif  strcmp(trend_direction,'downward')
            
            trend = close_data(i)<close_data(i+window);    % upward trend 
        end
        
        data = [data;close_data(i:i+window-1)' trend];  % data for training 

    end
    
    %% direction 
    
    X=data(:,1:end-1);  % extract X from the dataset
    y=data(:,end);       % extracX from the dataset

    cv=cvpartition(size(X,1),'holdout',0.25);    % cv partition for splitting the dataset
    
    Xtrain = X(~cv.test,:);  % extract training input
    ytrain = y(~cv.test);    % extract training target

    Xtest = X(cv.test,:);    % extract testing input
    ytest = y(cv.test);      % extract testing target

    mdl = fitcnb(Xtrain,ytrain);  % model training

    [pred,prob]=predict(mdl,Xtest);      % model prediction
    
    pred_train=predict(mdl,Xtrain);      % model prediction
    
    acc_train(c)=sum(pred_train==ytrain)/length(ytrain);  % compute the accuracy

    acc_test(c)=sum(pred==ytest)/length(ytest);  % compute the accuracy

    cmat = confusionmat(pred,ytest);
    TP = cmat(1,1);
    TN = cmat(2,2);
    
    FN = cmat(1,2);
    FP = cmat(2,1);
    
    prec(c) = TP/(TP+FP);
    recall(c) = TP/(TP+FN);
    
    f1score(c) = (2*prec(c)*recall(c))/(prec(c) + recall(c));
    
    [~,~,auc]=perfcurve(ytest,prob(:,2),1);
    
    AUC(c) = auc(1);
    
    avg_time(c)=toc;
end

%% Result
[~,idx]=sort(acc_test,'descend');  % sort the accuracies and extract the index

currencies_sorted = currencies(idx);  % extract the currencies from the sorted index
accuracy_test_sorted = (acc_test(idx))';        % extract the accuracies of the sorted index
accuracy_train_sorted = (acc_train(idx))';        % extract the training accuracies of the sorted index
precision_sorted = (prec(idx))';       % extract the precision of the sorted index
recall_sorted = (recall(idx))';        % extract the recall of the sorted index
f1score_sorted = (f1score(idx))';      % extract the f1 score of the sorted index
AUC_sorted = (AUC(idx))';              % extract the AUC of the sorted index
avg_time_sorted = (avg_time(idx))';
T=table(currencies_sorted,accuracy_train_sorted ,accuracy_test_sorted,precision_sorted,recall_sorted,f1score_sorted,AUC_sorted,avg_time_sorted)  % create table

writetable(T,'naive bayes.xlsx')

% disp the best accuracies
disp('Currency that has the best accuracy')
currencies(idx(1))
acc_test(idx(1))
