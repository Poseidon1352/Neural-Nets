n_train = 20000;n_test = 2000;eta = 1/n_train;
n_iter = 3000;
W = zeros(10,785); % Weight Matrix [No of classes X Input Size]
T = zeros(10,n_train); %Target matrix.
for sample = 1:n_train
    T(train_labels(sample)+1,sample) = 1;% Each column has a '1' in the location of the true class of that image
end
train_error = zeros(n_iter,1); %Training error over iterations
% Softmax Regression
for iter = 1:n_iter
    Y = exp(W*train_images); %Computing class activations 'a'
    for sample = 1:n_train
        Y(:,sample) = Y(:,sample)/sum(Y(:,sample)); %Normalizing activations
    end
    W = W - eta*(Y - T)*train_images'; %Gradient descent based weight update
    %Training error over iterations
    [~,I] = max(Y,[],1);
    train_output = (I - 1)';
    train_error(iter) = sum(1-(train_output == train_labels))/n_train;
end
plot(1 - train_error); ylabel('training accuracy'); xlabel('No of iterations'); print('figlast','-dpng');
%Calculating training error
[~,I] = max(W*test_images,[],1); test_output = (I-1)';
test_error_SR = sum(1-(test_output == test_labels))/n_test;