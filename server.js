const express = require('express');
const app = express();
const port = 3000; // Server runs on port 3000

const responseText1 = `clc
clear all

%% Inputs
a = [1, 1; 3, 4; -1, -1]; % Coefficients of constraints
b = [20; 72; -5];         % Right-hand side values
c = [4, 5];               % Coefficients of objective function

%% Plotting
x1 = 0:0.5:max(b); % Generate x1 values for plotting

% Calculate x2 values for each constraint
x21 = (b(1) - a(1,1)*x1) / a(1,2);
x22 = (b(2) - a(2,1)*x1) / a(2,2);
x23 = (b(3) - a(3,1)*x1) / a(3,2);

% Ensure x2 values are non-negative
x21(x21 < 0) = NaN;
x22(x22 < 0) = NaN;
x23(x23 < 0) = NaN;

% Plot the constraints
plot(x1, x21, 'r', x1, x22, 'k', x1, x23, 'm');
title('Graph of Constraints');
xlabel('x1');
ylabel('x2');
legend('x1 + x2 = 20', '3x1 + 4x2 = 72', 'x1 + x2 = 5');
grid on
hold on

%% Intersection Points with Axes
pts = [0, 0]; % Initialize intersection points

for i = 1:size(a, 1)
    if a(i, 1) ~= 0
        x1_intersect = b(i) / a(i, 1);
        if x1_intersect >= 0
            pts = [pts; x1_intersect, 0];
        end
    end
    if a(i, 2) ~= 0
        x2_intersect = b(i) / a(i, 2);
        if x2_intersect >= 0
            pts = [pts; 0, x2_intersect];
        end
    end
end

%% Intersection Points of Constraints
for i = 1:size(a, 1)
    for j = i+1:size(a, 1)
        a4 = a([i, j], :);
        b4 = b([i, j]);
        if det(a4) ~= 0
            X = a4 \ b4;
            if all(X >= 0)
                pts = [pts; X'];
            end
        end
    end
end

%% Feasible Region
feasible_pts = [];
for i = 1:size(pts, 1)
    if all(a * pts(i, :)' <= b)
        feasible_pts = [feasible_pts; pts(i, :)];
    end
end

%% Objective Value and Points
value = feasible_pts * c'; % Calculate objective function values
table(feasible_pts, value) % Display feasible points and their objective values
[obj, index] = min(value); % Find the minimum value (since it's minimization)
x1 = feasible_pts(index, 1);
x2 = feasible_pts(index, 2);
fprintf('Objective Value is %f at (%f, %f)\n', obj, x1, x2);`;


const responseText2 = `bfs code 

clc 
clear all

A=[1 0 0 1 0 0 0;0 1 0 0 1 0 0; -1 1 0 0 0 1 0; -1 0 2 0 0 0 1];
B=[4;4;6;4];
C=[-1 2 -1 0 0 0 0];
n=size(A,2);
m=size(A,1);

if n>m
    ncm = nchoosek(n, m);
    p = nchoosek(1:n, m);
    sol=[];
    for i=1:ncm
        y = zeros(n,1);
        A1 = A(:,p(i,:));
        if det(A1)~=0
            X = A1\B;
            if all(X>=0)
                y(p(i,:)) = X;
                sol = [sol y];
            end
        end
    end
else
    error('no of avriables are less than number of constraints')
end

z = C*sol;
[obj,index] = max(z);
BFS = sol(:,index);
optimal_value = [BFS' obj]`

// Route for /1
app.get('/1', (req, res) => {
    res.type('text/plain');
    res.send(responseText1);
});

// Route for /2
app.get('/2', (req, res) => {
    res.type('text/plain');
    res.send(responseText2);
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
