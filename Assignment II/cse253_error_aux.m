eps = 10^-5; n = 1000; X = X(:,1:n); T = T(:,1:n);
EIH = zeros(NI,NHU);EHO = zeros(NHU+1,NO);
A = (WIH')*X;[Z,H_] = act(A,'tanh');Z = [ones(1,n);Z]; Y = exp((WHO')*Z);
for sample = 1:n
    Y(:,sample) = Y(:,sample)/sum(Y(:,sample));
end
Del_k = T - Y;
EIHr = -X*((H_.*(WHO(2:NHU+1,:)*Del_k))');
EHOr = -Z*(Del_k');
for i = 1:NI
    for j = 1:NHU
        WIH(i,j) = WIH(i,j) + eps;
        Yf = exp((WHO')*[ones(1,n);act((WIH')*X,'tanh')]);
        WIH(i,j) = WIH(i,j) - 2*eps;
        Yb = exp((WHO')*[ones(1,n);act((WIH')*X,'tanh')]);
        for sample = 1:n
            Yf(:,sample) = Yf(:,sample)/sum(Yf(:,sample));
            Yb(:,sample) = Yb(:,sample)/sum(Yb(:,sample));
        end
        EIH(i,j) = -sum(sum(T.*(log(Yf) - log(Yb))))/(2*eps);
        WIH(i,j) = WIH(i,j) + eps;      
    end
end
for i = 1:NHU + 1
    for j = 1:NO
        WHO(i,j) = WHO(i,j) + eps;
        Yf = exp((WHO')*[ones(1,n);act((WIH')*X,'tanh')]);
        WHO(i,j) = WHO(i,j) - 2*eps;
        Yb = exp((WHO')*[ones(1,n);act((WIH')*X,'tanh')]);
        for sample = 1:n
            Yf(:,sample) = Yf(:,sample)/sum(Yf(:,sample));
            Yb(:,sample) = Yb(:,sample)/sum(Yb(:,sample));
        end
        EHO(i,j) = -sum(sum(T.*(log(Yf) - log(Yb))))/(2*eps);
        WHO(i,j) = WHO(i,j) + eps;
    end
end
WIHErr = sum(sum(abs(EIH - EIHr)))/(NI*NHU);
WHOErr = sum(sum(abs(EHO - EHOr)))/(NO*(NHU+1));