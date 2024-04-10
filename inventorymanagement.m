clear;

%% Initialisation
% Import the dataset
actual_demand = importdata('demand-2024_15.txt');

% Plot the curve of the demand to see if any patterns are noticeable
plot(linspace(1,104,104), actual_demand, 'b-', 'LineWidth', 2);
xlabel('Time');
ylabel('Demand');
title('Demand Data Visualization');
grid on;

%% Visualise demand distribution
% Via visual inspection it appears that our demand is gamma rather than
% normally distributed.
columns = 20;
[h_counts, h_centers] = hist(actual_demand, columns);
bin_width = h_centers(2) - h_centers(1);
pdf = h_counts / (sum(h_counts) * bin_width);
bar(h_centers, pdf, 'hist');
xlabel('Demand');
ylabel('Probability Density');
title('Probability Density Function (PDF) of Demand Data');

%% Exercise 1 

% Declare initial parameters used for the optimisation functions later on
% as x0 (like initial brackets). Use conventional values.
alpha = 0.1;
alpha_H = 0.2;
beta_H = 0.053;
alpha_SE = 0.05;
level_in = mean(actual_demand(1:10));
trend_in = mean(actual_demand(2:11)-actual_demand(1:10));

% Put the input required for SES in one vector as the function requires.
input_SES = [alpha,level_in];
% Define a handle function used to find the minimum RMSE.
% Only use the training set not to overfit!
% Use fmincon so that we can set upper/lower bounds.
opt_SES = @(input) optimise_RMSE_SES(actual_demand,input,70);
optimal_SES = fmincon(opt_SES,input_SES,[],[],[],[],[0,0],[0.3,inf]);

% Do the same as with SES, only input vector is longer.
input_Holt = [alpha_H,beta_H,level_in,trend_in];
opt_Holt = @(input) optimise_RMSE_Holt(actual_demand,input,70);
optimal_Holt = fmincon(opt_Holt,input_Holt,[],[],[],[],[0.02,0.005,0,0],[0.51,0.176,inf,inf]);

% Call the get_smoothing_SES function and optimise its output, namely
% the mean squared smoothed error of the training set so that we extract
% the optimal smoothing coefficient alpha.
smoothing_SES = @(alphaSE) get_smoothing_SES(actual_demand(1:70),alphaSE,optimal_SES);
optimal_smooth_SES = fmincon(smoothing_SES,alpha_SE,[],[],[],[],0,1);

% Repeat for Holt.
smoothing_Holt = @(alphaSE) get_smoothing_Holt(actual_demand(1:70),alphaSE,optimal_Holt);
optimal_smooth_Holt = fmincon(smoothing_Holt,alpha_SE,[],[],[],[],0,1);

% Calling the designated functions, we generate the naive and analytical
% estimators for the forecast standard deviations.
[sigma_n_SES,sigma_a_SES,smoothed_RMSE_SES] = sigma_est_SES(actual_demand,optimal_smooth_SES,optimal_SES,70);
[sigma_n_H,sigma_a_H] = sigma_est_Holt(actual_demand,optimal_smooth_Holt,optimal_Holt,70);

% Observing that SES showcases a relatively stable performance in terms of
% st.dev. and also simplifies further computations, we opt for the
% predictions as extrapolated via SES.
[predictions_ses,~,all_periods_SES] = SES(actual_demand,optimal_SES);
[predictions_holt,~,all_periods_holt] = Holt(actual_demand,optimal_Holt);
% We visualise how the SES and Holt estimates's curve compares to the actual 
% demand flow. No noticeable overfit.
figure;
time = 1:104;
plot(time,all_periods_SES(1:end-1))
hold on
plot(time,actual_demand)
xlabel('Week')
ylabel('Demand')
title('SES Prediction vs Actual Demand')
legend('SES Prediction','Actual')

figure;
time = 1:104;
plot(time,all_periods_holt)
hold on
plot(time,actual_demand)
xlabel('Week')
ylabel('Demand')
title('Holt Prediction vs Actual Demand')
legend('Holt Prediction','Actual')

%% Exercise 2
% Declaration of all related parameters as suggested by the exercise.
L = 2; 
R = 4; 
S = 32000;
v = 120;
P = 132;
r = 0.25;
A = 500;
B2 = 0.05;
P2 = 0.99;

% Because we don't know precisely which week falls into review and which
% into lead-time, we need to account for any feasible scenario. Thereby,
% we know that the sigma for all 6 weeks is given by the last entry in
% either estimator, so we can use it as our estimated st.dev. for R+L.
% We opt for the analytical sigma as it rests on less assumptions and
% we believe it would capture more noise in our sporadic demand shape.
mu_RL = predictions_ses(end);
sigma_RL = sigma_a_SES(end);

% Based on the review+lead-time parameters, in the (R,S)-setting it is
% rather straightforward how to enumerate the other parameters.
mu_R = mu_RL / (1+(L/R));
sigma_R = sigma_RL / sqrt(1+(L/R));
mu_L = mu_R *L / R ;
sigma_L = sigma_R*sqrt(L/R);

% We generate an anonymous function that gives the safety factor
% corresponding to a certain level of S. We'll need it later as well.
k = @(S) (S-mu_RL)/sigma_RL;
k_initial = k(S);

% Based on the previously declared mu's and sigma's, we generate the rho
% and lambda parameters for the gamma distribution.
rho_R = (mu_R/sigma_R)^2;
rho_L = rho_R*L/R;
rho_RL = (R+L)*rho_R / R;
lambda = mu_R / (sigma_R^2);

% We know generate handle functions that compute various KPIs for our 
% initial order-up-to level S below. For the ESPRC we implement the 'with
% correction for shortage at the start of the RC' version as with such
% erratic demand cycles shortages are possible to appear and we'd better
% account for them.
P1_calc = @(S) gamcdf(S,rho_RL,1/lambda);
P1_initial = P1_calc(S);

ESPRC_calc = @(S) loss_gamma(S,rho_RL,lambda)-loss_gamma(S,rho_L,lambda);
ESPRC_initial = ESPRC_calc(S);

P2_calc = @(S) 1 - (ESPRC_calc(S)/mu_R);
P2_initial = P2_calc(S);

% The ordering costs are contingent only on the review period R as they're
% concerned with the order count per year.
order_c = @(S) A*52/R;
order_initial = order_c(S);

% The holding costs capture the average on-hand stock level and the
% respective expenses for its storage.
hold_c = @(S) (k(S)*sigma_RL+(1/2)*mu_R)*v*r;
hold_initial = hold_c(S);

% For now we use the formulation featuring B2-costs as they've been
% provided to us.
shortage_c = @(S) B2*v*ESPRC_calc(S)*(52/R);
shortage_initial = shortage_c(S);

% The total costs simply combine the previous three together.
total_c = @(S) order_c(S) + hold_c(S) + shortage_c(S);
total_initial=total_c(S);

fprintf('P1 initial: %d\n', P1_initial);
fprintf('ESPRC initial: %d\n', ESPRC_initial);
fprintf('P2 initial: %d\n', P2_initial);
fprintf('Ordering costs initial: %d\n', order_initial);
fprintf('Holding costs initial: %d\n', hold_initial);
fprintf('Shortage costs initial: %d\n', shortage_initial);
fprintf('Total costs initial: %d\n', total_initial);

%% Exercise 3
% Using fminsearch we recognise the minimum total costs value. Having
% established which S yields it, using the expected demand during R+L as
% initial guess, we perform relevant KPIs using the anonymous functions
% constructed previously.
S_opt_TC = ceil(fminsearch(total_c,mu_RL));
P1_opt_TC = P1_calc(S_opt_TC);
P2_opt_TC = P2_calc(S_opt_TC);
order_opt_TC = order_c(S_opt_TC);
hold_opt_TC = hold_c(S_opt_TC);
shortage_opt_TC = shortage_c(S_opt_TC);
total_opt_TC = total_c(S_opt_TC);

fprintf('S optimal TC: %d\n', S_opt_TC);
fprintf('P1 optimal TC: %d\n', P1_opt_TC);
fprintf('P2 optimal TC: %d\n', P2_opt_TC);
fprintf('Ordering costs optimal TC: %d\n', order_opt_TC);
fprintf('Holding costs optimal TC: %d\n', hold_opt_TC);
fprintf('Shortage costs optimal TC: %d\n', shortage_opt_TC);
fprintf('Total costs optimal TC: %d\n', total_opt_TC);
%% Exercise 4
% In order to tackle this question, we need to introduce the service
% equation so that we can seek the S which yields the desired fill rate.
% Setting the equation equal to 0 fetches the corresponding order-up-to 
% level, which enables us to quantify KPIs again using the same functions.
service_eq = @(S) ESPRC_calc(S) - (1-P2)*mu_R;
S_opt_P2 = ceil(fzero(service_eq, mu_RL));
P1_opt_P2 = P1_calc(S_opt_P2);
P2_opt_P2 = P2_calc(S_opt_P2);
order_opt_P2 = order_c(S_opt_P2);
hold_opt_P2 = hold_c(S_opt_P2);
shortage_opt_P2 = shortage_c(S_opt_P2);
total_opt_P2 = total_c(S_opt_P2);

fprintf('S optimal for P2=0.99: %d\n', S_opt_P2);
fprintf('P1 for P2=0.99: %d\n', P1_opt_P2);
fprintf('Ordering costs optimal for P2=0.99: %d\n', order_opt_P2);
fprintf('Holding costs optimal for P2=0.99: %d\n', hold_opt_P2);
fprintf('Shortage costs optimal for P2=0.99: %d\n', shortage_opt_P2);
fprintf('Total costs optimal for P2=0.99: %d\n', total_opt_P2);

%% Exercise 5
% We first define an anonymous function which yields the corresponding B1
% costs in Mr.Stark's problem. The derivation is supplied in the report.
B1_find = @(S) (v*r*R)/(52*gampdf(S,rho_RL,1/lambda));

% We then generate a matrix B1 where we store the B1 cost for each fill
% rate P2 between 1% and 99% (all integer percentages). We exclude the
% extremes (0% and 100%) for they are unrealistic in practice and produce
% asymptotic results. In S_serv we store the order-up-to level which solves
% the service equation as constructed in the for loop.
B1 = zeros(99,1);
P_2 = linspace(0.01,0.99,99);
S_serv = zeros(99,1);
for i = 1:99
    serv = @(S) ESPRC_calc(S) - (1-P_2(i))*mu_R;
    S_serv(i) = fzero(serv, mu_RL);
    B1(i) = B1_find(S_serv(i));
    B1(i) = B1(i) / ESPRC_calc(S_serv(i));
end

% We plot the B1 values corresponding to P2 figures in the 25-99% range (we
% omit the lesser values for they generate extremely high costs and are
% also obsolete from Mr.Stark's perspective - he wouldn't opt for such low
% fill rates anyway.
plot(P_2(25:end),B1(25:end),'b-', 'LineWidth', 2);
xlabel('Fill Rate (%)');
ylabel('Goodwill Costs per Item Short');
title('Relationship between Fill Rate and Goodwill Costs');
grid on;
% Highlight the P2-rate of 99%
hold on;
plot(P_2(end), B1(end), 'ro', 'MarkerSize', 8);
text(P_2(end), B1(end), ' Target (99%)', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
hold off;

% We compute the B1 costs associated with the initial value of S and with
% the order-up-to level which achieves a fill rate of 99%.
B1_initial = B1_find(S);
B1_P2 = B1_find(S_opt_P2);

%% Exercise 6
% We first declare a number of handle functions, all of which dependent onn
% the review period R, and fix the order-up-to level at the initial S=32000
% figure.
Mu_RL = @ (R) predictions_ses(1)*(R+L);
Mu_R = @ (R) Mu_RL(R)/(1+(L/R));
Mu_L = @ (R) (L/R)*Mu_R(R);

Sigma = @(R) smoothed_RMSE_SES*sqrt(cumsum((1 + linspace(0,R+L-1,R+L)*optimal_SES(1)).^2));
Sigma_RL = @(R) get_last_entry(Sigma(R));
Sigma_R = @(R) Sigma_RL(R) / sqrt(1+(L/R));
Sigma_L = @(R) sqrt(L/R)*Sigma_R(R);

Rho_R = @ (R) (Mu_R(R)/Sigma_R(R))^2;
Rho_L = @ (R) Rho_R(R)*L/R;
Rho_RL = @ (R) (R+L)*Rho_R(R) / R;
Lambda = @ (R) Mu_R(R) / (Sigma_R(R)^2);

% We update the cost functions so that they now also depend on the 
% change in R rather than S alone. For now, we exclude the goodwill 
% losses from the shortage costs function prescription. We restrict the 
% holding costs to be at least 0 (they can't plunge into the negatives)
% via the max function.
P1_new = @(R,S) gamcdf(S,Rho_RL(R),1/Lambda(R));
ESPRC_new = @(R,S) loss_gamma(S,Rho_RL(R),Lambda(R))-loss_gamma(S,Rho_L(R),Lambda(R));
order_new = @(R,S) A*52/R;
hold_new = @(R,S) max(S - Mu_RL(R) + (1/2)*Mu_R(R),0)*v*r;
shortage_new = @(R,S) B2*v*ESPRC_new(R,S)*(52/R);
total_c_new = @(R,S) order_new(R,S) + hold_new(R,S) + shortage_new(R,S);

% For each review period between 1 week and 52 weeks (1 year), we verify
% which R spurs the minimal total costs incurred by Mr.Stark. The optimal
% vaue will be displayed by R_optimal.
Optimal_New_Costs = total_c_new(1,S);
R_optimal = 1;
for i=2:52
    current = total_c_new(i,S);
    if current<Optimal_New_Costs
        Optimal_New_Costs = current;
        R_optimal = i;
    end
end

% To compute the S which adheres to the optimal R as resulting from above,
% we rectify the initial service equation. Solving the service
% equation provides us with the desired optimal S.
service_new_no_goodwill = @(S) ESPRC_new(R_optimal,S) - (1-P2)*Mu_R(R_optimal);
S_opt_P2_no_goodwill = ceil(fzero(service_new_no_goodwill, Mu_RL(R_optimal)));
P1_opt_P2_new = P1_new(R_optimal,S_opt_P2_no_goodwill);
order_opt_P2_new = order_new(R_optimal,S_opt_P2_no_goodwill);
hold_opt_P2_new = hold_new(R_optimal,S_opt_P2_no_goodwill);
shortage_opt_P2_new = shortage_new(R_optimal,S_opt_P2_no_goodwill);
goodwill_P2_new=0;
total_opt_P2_new = total_c_new(R_optimal,S_opt_P2_no_goodwill);

fprintf('The optimal review period when omitting goodwill costs is: %d\n', R_optimal)
fprintf('S optimal for new R without goodwill: %d\n', S_opt_P2_no_goodwill);
fprintf('P1 for new R without goodwill: %d\n', P1_opt_P2_new);
fprintf('Ordering costs optimal for new R without goodwill: %d\n', order_opt_P2_new);
fprintf('Holding costs optimal for new R without goodwill: %d\n', hold_opt_P2_new);
fprintf('Shortage costs optimal for new R without goodwill: %d\n', shortage_opt_P2_new);
fprintf('Goodwill costs for new R without goodwill: %d\n', goodwill_P2_new);
fprintf('Total costs optimal for new R without goodwill: %d\n', total_opt_P2_new);

% We now proceed to account for goodwill costs as well by including them in
% the shortage cost equation. Consequently, the total cost function is to
% be updated. The sequence of steps is identical to the computation with no
% goodwill costs. The optimal R is now R_optimal_2.
goodwill = @(R,S) B1(end)*(1-gamcdf(S,Rho_RL(R),1/Lambda(R)))*ESPRC_new(R,S)*(52/R);
shortage_B1 = @(R,S) shortage_new(R,S) + goodwill(R,S);
total_c_B1 = @(R,S) order_new(R,S) + hold_new(R,S) + shortage_B1(R,S);
Optimal_New_Costs_2 = total_c_B1(1,S);
R_optimal_2 = 1;
for i=2:52
    current_2 = total_c_B1(i,S);
    if current_2<Optimal_New_Costs_2
        Optimal_New_Costs_2 = current_2;
        R_optimal_2 = i;
    end
end

% We finally compute the new KPIs, now addressing goodwill as well.
service_new_with_goodwill = @(S) ESPRC_new(R_optimal_2,S) - (1-P2)*Mu_R(R_optimal_2);
S_opt_P2_with_goodwill = ceil(fzero(service_new_with_goodwill, Mu_RL(R_optimal_2)));
P1_opt_P2_new_2 = P1_new(R_optimal_2,S_opt_P2_with_goodwill);
order_opt_P2_new_2 = order_new(R_optimal_2,S_opt_P2_with_goodwill);
hold_opt_P2_new_2 = hold_new(R_optimal_2,S_opt_P2_with_goodwill);
shortage_opt_P2_new_2 = shortage_new(R_optimal_2,S_opt_P2_with_goodwill);
goodwill_P2_new_2 = goodwill(R_optimal_2,S_opt_P2_with_goodwill);
total_opt_P2_new_2 = total_c_B1(R_optimal_2,S_opt_P2_with_goodwill);

fprintf('The optimal review period when considering goodwill costs is: %d\n', R_optimal_2)
fprintf('S optimal for new R with goodwill: %d\n', S_opt_P2_with_goodwill);
fprintf('P1 for new R with goodwill: %d\n', P1_opt_P2_new_2);
fprintf('Ordering costs optimal for new R with goodwill: %d\n', order_opt_P2_new_2);
fprintf('Holding costs optimal for new R with goodwill: %d\n', hold_opt_P2_new_2);
fprintf('Shortage costs optimal for new R with goodwill: %d\n', shortage_opt_P2_new_2);
fprintf('Goodwill costs for new R with goodwill: %d\n', goodwill_P2_new_2);
fprintf('Total costs optimal for new R with goodwill: %d\n', total_opt_P2_new_2);


%% Defined funcs
% The last_entry function simply extracts the last item in a vector
% supplied as input.
function last_entry = get_last_entry(items)
    last_entry = items(end);
end


% The optimise_RMSE_SES function calculates the root-mean-squared error of
% the SES estimates on the training set only. The input arguments are the
% dataset, the SES smoothing constant alpha and initial level, and the
% desired training set size (70 in our case). The return variable is the
% RMSE of the training set.
function RMSE_SES = optimise_RMSE_SES(data,input,size_training_set)
% We run SES on the dataset and extract its squared errprs.
[~,squared_e,~] = SES(data,input);

% We calculate the RMSE of the training set.
RMSE_SES = sqrt(mean(squared_e(1:size_training_set)));
end


% The optimise_RMSE_Holt function serves the exact same purpose as the one
% above, but this time focuses on Holt rather than SES estimates. Input now
% extends so that it features the second smoothing parameter (beta) and the
% trend estimate. Returned variable again is the training RMSE.
function RMSE_Holt = optimise_RMSE_Holt(data,input,size_training_set)
% Again, we extract the squared errors for the Holt procedure and callucate
% the training RMSE.
[~,squared_e] = Holt(data,input);
RMSE_Holt = sqrt(mean(squared_e(1:size_training_set)));
end
     

% The SES function generates the SES forecasts on our dataset. It requires
% the dataset (data) and a vector of the relevant parameters (pars), which
% in the case of SES are alpha and the initial level. The returned variables are the predictions for the upcoming 6
% weeks (predictions_SES), the squared errors for the past (sq_error)
% predictions and the past predictions themselves (all_SES).
function [predictions_SES,sq_error,all_SES] = SES(data,pars)
m = size(data,1);
alpha = pars(1);
level_in = pars(2);
SES = ones(m+1,1);
SES(1) = level_in;
for i=1:m
    SES(i+1)=alpha*data(i)+(1-alpha)*SES(i);
end
errors = data-SES(1:end-1);
sq_error = errors.^2;
predictions_SES = cumsum(SES(end)*ones(1,6));
all_SES = SES;
end


% The Holt function generates the Holt forecasts on our dataset. It requires
% the dataset (data) and a vector of the relevant parameters (pars), which
% in the case of Holt are alpha,beta, the initial level and the trend
% estimate. The returned variables are the predictions for the upcoming 6
% weeks (predictions_Holt), the squared errors for the past (sq_error)
% predictions and the past predictions themselves (all_Holt).
function [predictions_Holt,sq_error,all_Holt] = Holt(data,pars)
m = size(data,1);
alpha = pars(1);
beta = pars(2);
level_in = pars(3);
trend_in = pars(4);
all_Holt = ones(m,1);
level = ones(m,1);
trend = ones(m,1);
for i=1:m
    if i==1 
    level(i)=alpha*data(i)+(1-alpha)*(level_in+trend_in);
    trend(i)=beta*(level(i)-level_in)+(1-beta)*trend_in;
    all_Holt(i) = level_in+trend_in;
    else
    level(i)=alpha*data(i)+(1-alpha)*(level(i-1)+trend(i-1));
    trend(i)=beta*(level(i)-level(i-1))+(1-beta)*trend(i-1);
    all_Holt(i) = level(i-1)+trend(i-1);
    end
end
errors = data-all_Holt;
sq_error = errors.^2;
predictions_Holt = cumsum(level(end)+(trend(end)*linspace(1,6,6)));
end


% The get_smoothing_SES function calculates the smoothed mean squared
% error of the SES forecasts on a supplied dataset (data). Apart from data,
% it requires the input for SES (alpha,level) and the 
% smoothing constant alphaSE. It returns the smoothed RMSE.
function smoothed_final_SES = get_smoothing_SES(data,alphaSE,input)
[~,sq_errors,~] = SES(data,input);
m = size(sq_errors,1);
smoothed_se = zeros(m,1);
smoothed_se_in = mean(sq_errors);
for i = 1:m
    if i == 1
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se_in;    
    else
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se(i-1);
    end
end 
smoothed_final_SES = sqrt(mean(smoothed_se));
end


% The get_smoothing_Holt function calculates the smoothed mean squared
% error of the Holt forecasts on a supplied dataset (data). Apart from data,
% it requires the input for Holt (alpha,beta,level,trend) and the 
% smoothing constant alphaSE. It returns the smoothed RMSE.
function smoothed_final_Holt = get_smoothing_Holt(data,alphaSE,input)
[~,sq_errors] = Holt(data,input);
m = size(sq_errors,1);
smoothed_se = zeros(m,1);
smoothed_se_in = mean(sq_errors);
for i = 1:m
    if i == 1
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se_in;    
    else
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se(i-1);
    end
end 
smoothed_final_Holt = sqrt(mean(smoothed_se));
end


% The sigma_est_SES provides us with the naive and analytical estimators
% for the forecast error standard deviation. It uses as initial sigma for
% the one-period-ahead forecast the training RMSE of the SES estimates and
% then performs SES on the squared errors as extrapolated from the SES
% forecasting procedure on the entire past data. The formulas for the naive
% and the analytical sigmas of the 6-weeks-ahead forecasts are provided in
% the report. The function takes as parameters the dataset, the smoothing
% alphaSE, the SES input (alpha,level) and the size of the training set.
% The returned variables are the naive and the analytical sigma estimators,
% alongside the smoothed RMSE.
function [sigma_naive,sigma_analytical,smoothed_RMSE] = sigma_est_SES(data,alphaSE,input,range)
[~,sq_errors,~] = SES(data,input);
RMSE = sqrt(mean(sq_errors(1:range)));
m = size(sq_errors,1);
smoothed_se = zeros(m,1);
smoothed_se_in = RMSE^2;
for i = 1:m
    if i == 1
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se_in;    
    else
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se(i-1);
    end
end 

% St_dev naive
periods = linspace(1,6,6);
sigma_naive = sqrt(smoothed_se(end)*periods);

% St_dev analytical
j = linspace(0,5,6);
C = 1 + j.*input(1);
C_2 = C.^2;
smoothed_RMSE = sqrt(smoothed_se(end));
sigma_analytical = smoothed_RMSE*sqrt(cumsum(C_2));
end   


% The sigma_est_Holt provides us with the naive and analytical estimators
% for the forecast error standard deviation. It uses as initial sigma for
% the one-period-ahead forecast the training RMSE of the Holt estimates and
% then performs SES on the squared errors as extrapolated from the Holt
% forecasting procedure on the entire past data. The formulas for the naive
% and the analytical sigmas of the 6-weeks-ahead forecasts are provided in
% the report. The function takes as parameters the dataset, the smoothing
% alphaSE, the SES input (alpha,beta,level,trend) and the size of the training set.
% The returned variables are the naive and the analytical sigma estimators.
function [sigma_naive,sigma_analytical] = sigma_est_Holt(data,alphaSE,input,range)

% % Training and test
[~,sq_errors] = Holt(data,input);
alpha = input(1);
beta = input(2);
RMSE = sqrt(mean(sq_errors(1:range)));
m = size(sq_errors,1);
periods = linspace(1,6,6);
smoothed_se = zeros(m,1);
smoothed_se_in = RMSE.^2;
for i=1:m
    if i==1
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se_in;        
    else
        smoothed_se(i) = alphaSE*sq_errors(i)+(1-alphaSE)*smoothed_se(i-1);        
    end
end
smoothed_RMSE = sqrt(smoothed_se(end));

% St_dev naive
sigma_naive = smoothed_RMSE*sqrt(periods);

% St_dev analytical
j = linspace(0,5,6);
C = 1 + j.*(alpha+0.5*beta*(j+1));
C_2 = C.^2;
sigma_analytical = smoothed_RMSE*sqrt(cumsum(C_2));
end