
clear; clc;

g = 9.8; 
L = 1;
theta0 = deg2rad(30); 
omega0 = 0;           
tspan = [0 20];
y0 = [theta0; omega0];
f_pendulum = @(t, y) [y(2); -(g/L)*sin(y(1))];
[t_sol, y_sol] = ode45(f_pendulum, tspan, y0);

theta_sol = y_sol(:, 1); 
omega_sol = y_sol(:, 2); 

fig = figure;
plot(t_sol, rad2deg(theta_sol), 'b-', 'LineWidth', 2);
title('Graph of pendulum angle with RK4/5 (init angle = 30 deg, g = 9.8 m/s^2, L = 1 m)');
xlabel('Time (secs)');
ylabel('Angle (degrees)');
grid on;
exportgraphics(fig,'Matlab/pendulum/graph.png','Resolution',300);
disp(['Graph is saved!'])