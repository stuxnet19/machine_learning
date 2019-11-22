x = load('data/ex2x.dat');
y = load('data/ex2y.dat');

function j = cost_equation (x,y,theta)
	m = length(y);
	% compute the hypethesis matrix
	h = x * theta;
	% calculate the coast
	j = 1 / (2*m)*sum((h-y).^2);
end

function [theta,j_histoty] = gradient_descent(x,y,theta,alpha,iterations)
	%prepare variables
	m = length(y);
	j_histoty = zeros(iterations,1);
	h = x * theta;
	first_theta1 = theta(1) - alpha * (1/m) * sum(h-y)
	first_theta2 = theta(2) - alpha * (1/m) * sum((h-y).* x(:,2))
	for i = 1:iterations,
		% multiplication d'une matrice (2,n) par un vecteur a n dimention = vecteur a n dimention 
		h = x * theta;
		t1 = theta(1) - alpha * (1/m) * sum(h-y);
		% quand on fiat (h-y).*x(:,2) on multiplie les 2 vecteur ligne par ligne
		t2 = theta(2) - alpha * (1/m) * sum((h-y).* x(:,2));
		theta(1) = t1;
		theta(2) = t2;
		j_histoty(i) = cost_equation(x,y,theta);
		
	end
end
figure % open a new figure window
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')

m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x

alpha = 0.07 ;% learning rate
theta = [ 0;0 ];
iterations = 15000;

% running gradient descent

[theta,j_history] = gradient_descent(x,y,theta,alpha,iterations);
hold on
plot(x(:,2),x*theta,'-');
legend('Training data', 'Linear regression')
predicted_y = theta(1)+3.5*theta(2)

J_vals = zeros(100, 100);   % initialize Jvals to 100x100 matrix of 0's

theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);

for i = 1:length(theta0_vals),
	  for j = 1:length(theta1_vals),
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = (0.5/m) .* (x * t - y)' * (x * t - y);
    end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals'
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')