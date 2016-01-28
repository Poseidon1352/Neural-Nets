n_train = size(iristrain,1); %Used MATLAB's interface to import csv
n_test = size(iristest,1);
iristrain_mat = cell2mat(iristrain(:,1:4));
iristest_mat = cell2mat(iristest(:,1:4));
ztrain = zscore(iristrain_mat);
%ztest = zscore(iristest_mat);
ztest = iristest_mat - repmat(mean(iristrain_mat),[n_test 1]); %Using training mean to zscore
std_train = std(iristrain_mat); %Using training std to zscore
for i = 1:4
    ztest(:,i) = ztest(:,i)/std_train(i);
end

train_target = ones(n_train,1);
test_target = ones(n_test,1);
for i = 1:size(iristrain_mat,1)
    if strcmp(iristrain{i,5},'Iris-versicolor')
        train_target(i) = 0;
    end
end
for i = 1:size(iristest_mat,1)
    if strcmp(iristest{i,5},'Iris-versicolor')
        test_target(i) = 0;
    end
end
labels = {'sepal length','sepal width','petal length','petal width'};
for i = 1:4
    for j = i+1:4
        gscatter(ztrain(:,i),ztrain(:,j),iristrain(:,5));xlabel(labels{i});ylabel(labels{j});
        print(sprintf('fig%i%i',i,j),'-dpng');
    end
end
w = zeros(5,1);
n_err = 10;
eta = 1;
while n_err > 0 %Since data is linearly seperable, training till zero error
    for i = 1:n_train %Iterating over each datapoint instead of "randomly choosing"
        out = [1,ztrain(i,:)]*w >= 0;
        if out ~= train_target(i)
            if train_target(i) == 1
                w = w + eta*[1,ztrain(i,:)]';
            else
                w = w - eta*[1,ztrain(i,:)]';
            end
        end
    end
    output = [ones(n_train,1),ztrain]*w >= 0;
    n_err = sum(abs(output - train_target));
end

test_output = [ones(n_test,1),ztest]*w >= 0;
test_err = sum(abs(test_output - test_target));