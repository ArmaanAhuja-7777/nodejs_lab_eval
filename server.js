const express = require('express');
const app = express();
const port = 3000; // Server runs on port 3000

const responseText1 = `clc
clear all
% Input Parameters
Z = @(x1, x2) 3*x1-x2;
C1 = @(x1, x2) x1+x2-6;
C2 = @(x1, x2) x1+x2-4;

C = [3 -1];
A = [1 1 ; 1 1];
b = [6 ; 4];

n = size(A, 1);
m = size(A, 2);

% Plotting

x1 = 0 : 0.1 : max(b./A(:,1));

for i = 1:n
    x2 = (b(i)-A(i,1)*x1)/(A(i,2));
    x2 = max(0,x2);
    plot(x1,x2)
    hold on
end

% Intersection Points

pt = [];

A = [A ; eye(2)];
b = [b ; zeros(2,1)];

for i = 1 : size(A, 1)
    for j = i+1 : size(A,1)
        AA = [A(i,:) ; A(j,:)];
        bb = [b(i) ; b(j)];

        if det(AA) ~= 0
            X = inv(AA)*bb;
            if X >= 0
                pt = [pt X];
            end
        end
    end
end

% disp(pt)

% Finding Points

fp = [];

for i = 1:length(pt)
    pt1 = pt(1,i);
    pt2 = pt(2,i);

    if C1(pt1,pt2) >= 0 && C2(pt1,pt2) <= 0
        fp = [fp pt(:,i)];
        plot(pt1, pt2, "*r", "markerSize", 10)
    end
end

%disp(fp)

% Optimal Solution

if length(fp) == 0
    disp("No feasible solution")
else
    z = [];

    for i = 1 : length(fp)
        cost = Z(fp(1,i), fp(2,i));
        z = [z cost];
    end

    [optimal_value, index] = max(z)
    optimal_sol = fp(:,index)
end

hold off`;


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
