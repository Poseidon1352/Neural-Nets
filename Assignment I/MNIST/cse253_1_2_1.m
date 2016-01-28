n_train = 20000;n_test = 2000;eta = 1/n_train;
n_iter = 1000;
W = zeros(10,785); % Weight Matrix [No of classifiers X Input Size]
%Logistic Regression
for class_lbl = 0:9 %Computing weight for each class
    class_lbl
    for i = 1:n_iter
        y = (1./(1 + exp(-W(class_lbl+1,:)*train_images)))'; %Softmax output given current weights
        t = double((train_labels == class_lbl)); % 0-1 vector of targets for each image. 1 if image belongs to class 'class_lbl'
        err_sum = train_images*(y-t); %Sum of error over all training images
        W(class_lbl+1,:) = W(class_lbl+1,:) - eta*err_sum'; %Gradient descent based weight update
    end
end
%Calculating Individual Classifier Error
dec_mat = W*test_images;
classifier_err = zeros(10,1);
for classifier = 0:9
    dec = (1./(1 + exp(-dec_mat(classifier+1,:)))) > 0.5;
    trgt = (test_labels == classifier);
    classifier_err(classifier+1) = sum(abs(dec'-trgt))/n_test;
end
%Calculating Test Error
[~,I] = max(dec_mat,[],1);
test_output = (I - 1)';
test_err_LR = sum(1-(test_output == test_labels))/n_test;