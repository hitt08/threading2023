%% sample particle trajectories from a simple SDPP

Data = load('model_Q1.mat');
% config
T = 5; % time steps
N = size(Data.Q1,2); % labels (positions)
D = 51;  % similarity feature dimension
k = 20;  % number of trajectories to sample

% init model
clear model;
model.T = T;
model.N = N;

model.Q1 = Data.Q1;
model.Q = ones(1,N);

% enforce smoothness
Data = load('model_A.mat');
model.A = Data.A;

% similarity features
Data = load('model_G.mat');
model.G = Data.G;

%save('doc_thread_model.mat','model')

% sample
C = decompose_kernel(bp(model,'covariance'));
save('cov.mat','C')
sdpp_sample = sample_sdpp(model,C,k);

save('doc_thread_sdpp.mat','sdpp_sample')