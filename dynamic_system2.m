%% RLC circuit dynamic system

clear
clc

% Initial state
[x0(1), x0(2)] = v(0);

% Voltage from the two sources
u1 = @(t) 3 * sin(2*t);
u2 = 2;

% Simulation & measurements characteristics
t_span = 0:1e-5:5;
N = length(t_span);
state = zeros(N, 2);


%% measurements

% Get measurements for voltages vC and vR
for i = 1:N
    [state1, state2] = v(t_span(i));
    state(i, 1) = state2;
    state(i, 2) = state2;
end

Vc = state(:, 1);

% Arbitrarily add outlier values into the measurements
% Vc(100000) =  Vc(100000) + 400 *  Vc(100000);
% Vc(160000) =  Vc(160000) + 75 *  Vc(160000);
% Vc(290000) =  Vc(290000) + 180 *  Vc(290000);

U1 = double(u1(t_span))';
U2 = ones(N, 1) .* u2;


%% Mean Squares Method

% Poles of the stable filter
p = [200, 300];

% Input vector, after linear factorizetaation
zeta1 = lsim(tf([-1, 0], [1, p(1) + p(2), p(1) * p(2)]), Vc, t_span');
zeta2 = lsim(tf([-1], [1, p(1) + p(2), p(1) * p(2)]), Vc, t_span');
zeta3 = lsim(tf([1, 0], [1, p(1) + p(2), p(1) * p(2)]), U2, t_span');
zeta4 = lsim(tf([1], [1, p(1) + p(2), p(1) * p(2)]), U2, t_span');
zeta5 = lsim(tf([1, 0], [1, p(1) + p(2), p(1) * p(2)]), U1, t_span');
zeta6 = lsim(tf([1], [1, p(1) + p(2), p(1) * p(2)]), U1, t_span');

zeta = [zeta1, zeta2, zeta3, zeta4, zeta5, zeta6]';

% Estimation of the unknown parameters
res = compute_optimal(state, zeta) + [p(1) + p(2); p(1) * p(2); ...
                                        0; 0; 0; 0];
RC = res(1)
LC = res(2)


%% Result evaluation & deviation from the real model

% Output of the approximated model 
[t, Vc_approx] = ode45(@(t, Vc_approx) dynamics(t, Vc_approx, RC, LC, ...
                                                u1, u2), t_span, x0);
sum = 0;
for i = 1: N
    sum = sum + abs((Vc(i) - Vc_approx(i, 1)));
end
error = sum / N


%% Plots display

figure(1);
plot(t_span, Vc, 'Linewidth', 0.1);
ylabel('system output $V_C(t)$', 'interpreter', 'latex', 'FontWeight', 'bold');
xlabel('$t(s)$', 'interpreter', 'latex', 'FontWeight', 'bold');


%% Dynamics of the system

function dx = dynamics(t, x, RC_inv, LC_inv, u1, u2)
    dx(1)  = x(2);
    dx(2) = -RC_inv * x(2) - LC_inv * x(1) + LC_inv * u2 + RC_inv * u1(t);
    dx = dx';
end


%% Optimal parameters computation via optimization of the MSE

function theta = compute_optimal(state, zeta)
    sum1 = 0;
    sum2 = 0;

    for i = 1:length(state)
        sum1 = sum1 + zeta(:, i) * zeta(:, i)';
        sum2 = sum2 + zeta(:, i) * state(i, 1);
    end
    
    theta = sum1 \ sum2;
end
