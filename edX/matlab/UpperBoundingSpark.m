%% ===================================
% Creating the matrix A
%rng(120);
%n=50; m=150; 
%A=randn(n,m); 
%Spark=13; 
%A(:,m)=mean(A(:,1:Spark-1),2);

n = 4;
m = 4;
A = [16 -2 15 13; 5 6 8 8; 9 4 11 12; 4 12 10 1];

% Normalizing the columns
for k=1:1:m
    A(:,k)=A(:,k)/norm(A(:,k)); 
end

%% ===================================
% Evaluating the Spark by the mutual coherence
G=A'*A; 
G=abs(G); 
for k=1:1:m
    G(k,k)=0; 
end
mu=max(G(:)); 
SparkEst1=ceil(1+1/mu); 
disp('A lower bound on the Spark via the Mutual Coherence is:'); 
disp(SparkEst1); 

%% ===================================
% Evaluating the Spark by the Babel function
G=A'*A; 
G=abs(G); 
for k=1:1:m
    G(k,:)=sort(G(k,:),'descend'); 
end
G=G(:,2:end); 
G=cumsum(G,2); 
mu1=zeros(20,1); 
for k=1:1:20
    mu1(k)=max(G(:,k)); 
end
SparkEst2=find(mu1>1,1)+1; 
disp('A lower bound on the Spark via the Babel-Function is:'); 
disp(SparkEst2); 


%% ===================================
% Evaluating the Spark by the upper-bound
% options = optimoptions('linprog','Algorithm','dual-simplex',...
%                  'Display','none','OptimalityTolerance',1.0000e-07);
Z=zeros(m,m); 
Zcount=zeros(m,1); 
h=waitbar(0,'Sweeping through the LP problems');
set(h,'Position',[500 100 270 56]);
for k=1:1:m
    waitbar(k/m); 
    % We convert the problem min ||z||_1 s.t. Az=0 ^ z_k=1
    % to Linear Programming by splitting z into the
    % positive and negative entries z=u-v, u,v>=0
    c=ones(2*m,1); 
    Aeq=[A,-A];
    indicator=zeros(1,2*m); 
    indicator(k)=1; 
    Aeq=[Aeq; indicator]; % forcing zk=uk=1
    indicator=zeros(1,2*m); 
    indicator(m+k)=1; 
    Aeq=[Aeq; indicator]; % forcing vk=0 to avoid trivial solution
    beq=[zeros(n,1); 1; 0];
    lb=zeros(2*m,1); 
    ub=ones(2*m,1)*100; 
    Solution=linprog(c,[],[],Aeq,beq,lb,ub,[]); % options); 
    Solution=Solution.*(abs(Solution)>1e-7); 
    Z(:,k)=Solution(1:m)-Solution(m+1:2*m); 
    Zcount(k)=nnz(Solution); 
end
close(h); 
SparkEst3=Zcount(find(Zcount==min(Zcount),1)); 
disp('An upper bound on the Spark is:'); 
disp(SparkEst3); 
