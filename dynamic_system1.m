%% Mass-spring-damper dynamic system 

clear
clc

% System parameter initialization
m = 10;
b = 0.3;
k = 1.5;

% Initial state
x0(1) = 0;
x0(2) = 0;

% External force
u = @(t) 10 * sin(3*t) + 5; 

% Simulation & measurements characteristics
resolution = 0.1;
t_span = 0:resolution:10;


%% Simulate & collect input-output measurements

% Solve the ODE for every t
[t, state] = ode15s(@(t, state)dynamics(t, state, m, b, k, u), t_span, x0);

% Input-output                                        
U = u(t(:));
Y = state(:, 1);


%% Mean Squares Method

% Poles of the stable filter
syms m_ b_ k_; 
p = [0.1, 0.4];

% Parameters vector, after linear factorization
theta = [b_ / m_ - (p(1) + p(2)); k_ / m_ - (p(1) * p(2)); 1 / m_];

% Input vector, after linear factorization
zeta1 = lsim(tf([-1, 0], [1, p(1) + p(2), p(1) * p(2)]), state(:, 1), t);
zeta2 = lsim(tf([-1], [1, p(1) + p(2), p(1) * p(2)]), state(:, 1), t);
zeta3 = lsim(tf([1], [1, p(1) + p(2), p(1) * p(2)]), U, t);
                                            
zeta = [zeta1, zeta2, zeta3]';

% Estimation of the unknown parameters
to_solve = theta == compute_optimal(state, zeta);
solved = solve(to_solve, [m_, b_, k_]);
m_ = double(solved.m_);
b_ = double(solved.b_);
k_ = double(solved.k_);


%% Result evaluation & deviation from the real model

% Output of the approximated model 
[t, state_approx] = ode15s(@(t, state_approx) dynamics(t, state_approx, ...
                                               m_, b_, k_, u), t_span, x0);

error = zeros(length(state), 1); 
for i = 1:length(state)
    error(i) = abs((state(i) - state_approx(i)) / state(i));
end


%% Plots display

figure(1);
plot(t, error, 'Linewidth', 1);
ylabel('error: $\big| \frac{y(t) - \hat{y}(t)}{y(t)} \big|$', ...
        'interpreter', 'latex', 'FontWeight', 'bold');
xlabel('$t(s)$', 'interpreter', 'latex', 'FontWeight', 'bold');

figure(2);
plot(t, state(:, 1), 'Linewidth', 1);
ylabel('output $y(t)$', 'interpreter', 'latex', 'FontWeight', 'bold');
xlabel('$t(s)$', 'interpreter', 'latex', 'FontWeight', 'bold');


%% Dynamics of the system

function dx = dynamics(t, x, m, b, k, u)
    dx(1) = x(2);
    dx(2) = (1 / m) * (-k * x(1) - b * x(2) + u(t));
    dx = dx';
end


%% Optimal parameters computation via optimization of the MSE

function theta = compute_optimal(state, zeta)
    sum_denom = 0;
    sum_numer = 0;

    num_samples = length(state);
    for i = 1:num_samples
        sum_denom = sum_denom + zeta(:, i) * zeta(:, i)';
        sum_numer = sum_numer + zeta(:, i) * state(i, 1);
    end
    
    theta = sum_denom \ sum_numer;
end

