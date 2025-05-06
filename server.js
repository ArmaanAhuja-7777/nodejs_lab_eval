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

const lcm = `clc
clear all

cost = [11 13 17 14 ; 16 18 14 10 ; 21 24 13 10]
Supply = [8 5 9]
demand = [8 7 5 4]
[m,n] = size(cost)
if sum(Supply)==sum(demand)
    disp('BAL')
elseif sum(Supply)<sum(demand)
        cost(end + 1 , :)= zeros(1,n)
        Supply (end +1)= sum(demand)-sum(Supply)
else
    cost(:, end + 1)= zeros(m,1)
        demand (end +1)= sum(Supply)-sum(demand)
end
[m,n] = size(cost)
X = zeros(m,n)
cost_in = cost
while any(Supply)~=0 || any(demand) ~=0
min_cost = min(cost(:))
[r,c]=find(cost==min_cost)
y = min(Supply(r),demand(c))
[aloc,index] =  max(y)
rr = r(index)
cc = c(index)
X(rr, cc)=aloc
Supply(rr)=Supply(rr)-aloc
demand(cc)=demand(cc)-aloc
cost(rr,cc)=inf
end
cost_x = X.*cost_in
final_cost = sum(cost_x(:))
`

const dualSimplex = `A = [-3 -1 1 0 0 ; -4 -3 0 1 0 ; -1 -2 0 0 1];
b = [-3 ; -6 ; -3];
C = [-2 -1 0 0 0];

[m, n] = size(A);

Y = [A b];
bIndex = n-m+1 : n;

for s = 1:50
    Cb = C(bIndex);
    Xb = Y(:, end);
    ZjCj = Cb*Y(:, 1:n) - C;
    Z = Cb*Xb;
    
    table = [ZjCj Z ; Y]
    
    if Xb >= 0
        disp('Feasibility Achieved')
        Xb
        basicVar = bIndex
        disp('Optimal Objective Value')
        Z
        break
    else
        [a, LV] = min(Xb);
    
        if Y(LV, 1:n) >= 0
            disp('No Feasible Solution')
            break
    
        else
            for j = 1:n
                if Y(LV, j) < 0
                    ratio(j) = abs(ZjCj(j)/Y(LV, j));
                else
                    ratio(j) = inf;
                end
            end
        end
    
        [K, EV] = min(ratio);
        bIndex(LV) = EV;
    end
    
    pivot = Y(LV, EV);
    
    Y(LV, :) = Y(LV, :)/pivot;
    
    for i = 1:m
        if i ~= LV
            Y(i, :) = Y(i, :) - Y(i, EV)*Y(LV, :);
        end
    end
end
`

const simplex = `A = [1 1 1 0 ; 2 -1 0 1];
b = [6 ; 9];
C = [-3 2 0 0];

[m, n] = size(A);

Y = [A b];   % m*(n+1)
bIndex = n-m+1 : n;

for s = 1:50
    Cb = C(bIndex);   % 1*m
    Xb = Y(:, end);   % end -> m+1
    ZjCj = Cb*Y(:, 1:n) - C;   % 1*(n+1)
    Z = Cb*Xb;
    
    table = [ZjCj Z ; Y]
    
    disp(table)
    
    if ZjCj >= 0
        disp('Optimal Solution')
        Xb
        bIndex
        Z
        break
    else
        [a, EV] = min(ZjCj);    % stores value and index of entering variable
    
        if Y(:, EV) < 0
            disp('Unbounded Solution')
            break
    
        else
            for j = 1:m
                if Y(j, EV) > 0
                    ratio(j) = Y(j, end)/Y(j, EV);  % Y(j, end) or Xb(j) - same thing
                else
                    ratio(j) = inf;
                end
            end
        end
    
        [K, LV] = min(ratio);
        bIndex(LV) = EV;
    end
    
    pivot = Y(LV, EV);
    
    Y(LV, :) = Y(LV, :)/pivot;
    
    for i = 1:m
        if i ~= LV
            Y(i, :) = Y(i, :) - Y(i, EV)*Y(LV, :);
        end
    end
end
`

const bigm = `format short
clear all
clc
% Cost=[-4 -5 0 0 -1000 -1000 0]
% A=[3 1 1 0 0 0 27; 3 2 0 -1 1 0 3; 5 5 0 0 0 1 60]
% % BV=[3 5 6]
Cost=[-2 -1 0 0 -10000 -10000 0]
A=[3 1 0 0 1 0 3; 4 3 -1 0 0 1 6 ;1 2 0 1 0 0 3]
BV=[5 6 4]
 
ZjCj=Cost(BV)*A-Cost
 zcj=[Cost;ZjCj;A];
    bigmtable=array2table(zcj);
    bigmtable.Properties.VariableNames(1:size(zcj,2))={'x_1','x_2','s_1','s_2','A_1','A_2','sol'}
 
RUN= true;
while RUN
    ZC=ZjCj(1:end-1)
    if any(ZC<0)
        fprintf('  The current BFS is not optimal
')
        [ent_col,pvt_col]=min(ZC)
        fprintf('Entering Col =%d 
' , pvt_col);
        sol=A(:,end)
        Column=A(:,pvt_col)
        if Column<=0
            error('LPP is unbounded');
        else
            for i=1:size(A,1)
                if Column(i)>0
                    ratio(i)=sol(i)./Column(i)
                else
                    ratio(i)=inf
                end
            end
            [MinRatio,pvt_row]=min(ratio)
            fprintf('leaving Row=%d 
', pvt_row);
        end
        BV(pvt_row)=pvt_col;
        pvt_key=A(pvt_row,pvt_col);
        A(pvt_row,:)=A(pvt_row,:)./ pvt_key;
        for i=1:size(A,1)
            if i~=pvt_row
                A(i,:)=A(i,:)-A(i,pvt_col).*A(pvt_row,:);
            end
        end
        ZjCj=ZjCj-ZjCj(pvt_col).*A(pvt_row,:)
        ZCj=[ZjCj;A]
        TABLE=array2table(ZCj);
        TABLE.Properties.VariableNames(1:size(ZCj,2))={'x_1','x_2','s_1','s_2','A_1','A_2','sol'}
    else
        RUN=false;
        fprintf('  Current BFS is Optimal 
');
    end
end
`

const steep1 = `clc;
clear all;

x0 = 1;
y0 = 0.5;

f = @(x, y) x.^2 - x.*y + y.^2;

grad = @(x, y) [2*x - y, 2*y - x];


alpha = 0.01;      
epsilon = 1e-6;   
max_iter = 1000;

for k = 1:max_iter
    g = grad(x0, y0); 
    x = x0 - alpha * g(1);
    y = y0 - alpha * g(2);
    t = abs(f(x,y) - f(x0,y0));
    
    if t < epsilon
        break;
    else
        x0=x;
        y0=y;
    end
    


    fprintf('Iter %d: x = %.6f, y = %.6f, f = %.6f, ||grad|| = %.6e\n', ...
            k, x, y, f(x, y), t);
end

fprintf('\nMinimum found at x = %.6f, y = %.6f\n', x, y);
fprintf('Function value at minimum f(x, y) = %.6f\n', f(x, y));

% If maximum is asked, multiply the objective function by -1.
`

const steep2 = `syms l;
f = @(x) (x(1)-2)^2 + (x(2)-3)^2;
grad_f = @(x) [2*(x(1)-2); 2*(x(2)-3)];

x = [0; 0];
tol = 1e-6;
max_iter = 1000;

for i = 1:max_iter
    grad = grad_f(x);
    d = -grad;
    
    x_sym = x + l * d;
    f_l = f(x_sym);
    df_dl = diff(f_l, l);
    lambda_star = double(solve(df_dl == 0, l));
    
    x_new = x + lambda_star * d;
    
    if norm(x_new - x) < tol
        break;
    end
    
    x = x_new;
end

fprintf('Minimum found at x = [%f, %f] after %d iterations\n', x(1), x(2), i);
fprintf('Minimum function value f(x) = %f\n', f(x));`
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

app.get('/lcm', (req, res) =>{
    res.type('text/plain'); 
    res.send(lcm); 
})
app.get('/dualsimplex', (req, res) =>{
    res.type('text/plain'); 
    res.send(dualSimplex); 
})
app.get('/simplex', (req, res) =>{
    res.type('text/plain'); 
    res.send(simplex); 
})
app.get('/bigm', (req, res) =>{
    res.type('text/plain'); 
    res.send(bigm); 
})
app.get('/steep1', (req, res) =>{
    res.type('text/plain'); 
    res.send(steep1); 
})
app.get('/steep2', (req, res) =>{
    res.type('text/plain'); 
    res.send(steep2); 
})
// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
