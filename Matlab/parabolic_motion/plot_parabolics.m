x = 0:0.01:10;
v = 10;
angle = pi/4;
x = 0:0.1:10;
g = 9.8;
y = x .* tan(angle) - (g.*x.^2)/(2*v.^2.*(cos(angle)).^2);
fig = figure;
plot(x,y);
title('Parabolic Motion');
xlabel('x');
ylabel('y');
grid on;
exportgraphics(fig,'Matlab/parabolic_motion/graph.png','Resolution',300);
disp(['Graph is saved!'])