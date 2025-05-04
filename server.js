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
format short
% Matlab Code of Least Cost Method (LCM)
% Input Information
%% Input Phase
Cost=[11 20 7 8; 21 16 10 12; 8 12 18 9]
A=[50 40 70]
B=[30 25 35 40]
%% To check unbalanced/balanced Problem
if sum(A)==sum(B)
    fprintf('Given Transportation Problem is Balanced \n')
else
   fprintf('Given Transportation Problem is Unbalanced \n') 
   if sum(A)<sum(B)
       Cost(end+1,:)=zeros(1,size(B,2))
       A(end+1)=sum(B)-sum(A)
   elseif sum(B)<sum(A)
   Cost(:,end+1)=zeros(1,size(A,2))
       B(end+1)=sum(A)-sum(B)  
   end
end
ICost=Cost
X=zeros(size(Cost))   % Initialize allocation
[m,n]=size(Cost)      % Finding No. of rows and columns
BFS=m+n-1             % Total No. of BFS
%% Finding the cell(with minimum cost) for the allocations
for i=1:size(Cost,1)
    for j=1:size(Cost,2)
hh=min(Cost(:))   % Finding minimum cost value
[Row_index, Col_index]=find(hh==Cost)  % Finding position of minimum cost cell
x11=min(A(Row_index),B(Col_index))
[Value,index]=max(x11)            % Find maximum allocation
ii=Row_index(index)       % Identify Row Position
jj=Col_index(index)        % Identify Column Position
y11=min(A(ii),B(jj))        % Find the value
X(ii,jj)=y11
A(ii)=A(ii)-y11
B(jj)=B(jj)-y11
Cost(ii,jj)=inf
    end
end
%% Print the initial BFS
fprintf('Initial BFS =\n')
IBFS=array2table(X)
disp(IBFS)
%% Check for Degenerate and Non Degenerate
TotalBFS=length(nonzeros(X))
if TotalBFS==BFS
    fprintf('Initial BFS is Non-Degenerate \n')
else
    fprintf('Initial BFS is Degenerate \n')
end
%% Compute the Initial Transportation cost
InitialCost=sum(sum(ICost.*X))
fprintf('Initial BFS Cost is = %d \n',InitialCost)


`

const dualSimplex = `format short
clc
clear all

C=[-2 0 -1 0 0 0]
A=[-1 -1 1 1 0 -5; -1 2 -4 0 1 -8]
ib=[4 5]
zjcj=C(ib)*A-C
RUN=true;
while RUN
    if any(A(:,size(A,2))<0)
        fprintf('the current BFS is not feasible')
        [lvg_val, pvt_row]=min(A(:,size(A,2)))
for i=1:size(A,2)-1
    if A(pvt_row,i)<0
        m(i)=zjcj(i)/A(pvt_row,i)
    else
         m(i)=-inf
     end
end
[ent_val, pvt_col]=max(m)
A(pvt_row,:)=A(pvt_row,:)/A(pvt_row,pvt_col)
for i=1:size(A,1)
     if i~=pvt_row
         A(i,:)=A(i,:)-A(i,pvt_col).*A(pvt_row,:)
     end 
end
ib(pvt_row)=pvt_col;
zjcj=zjcj-zjcj(pvt_col).*A(pvt_row,:)
ZCj=[zjcj;A]
        TABLE=array2table(ZCj);
        TABLE.Properties.VariableNames(1:size(ZCj,2))={'x_1','x_2','x3','s_1','s_2','sol'}
else
    RUN=false;
 fprintf('    current BFS is Feasible and Optimal   \n')
end
end

`

const simplex = `% Simplex Method
%max z=2x1+5X2
%x1+4x2<=24
%3x1+1x2<=21
%x1+x2<=9
clc
clear all
format short
Noofvariables=2;
C=[2 5];
a=[1 4; 3 1; 1 1]
b=[24; 21; 9]
s=eye(size(a,1))
A=[a s b]
cost=zeros(1,size(A,2))
cost(1:Noofvariables)=C
bv= Noofvariables+1:1:size(A,2)-1
zjcj=cost(bv)*A-cost
zcj=[zjcj; A]
simptable=array2table(zcj);
simptable.Properties.VariableNames(1:size(zcj,2))={'x_1','x_2','s_1','s_2','s_3','sol'}
RUN=true;
while RUN
if any(zjcj<0); %check for (most) negative value
    fprintf(' the current BFS is not optimal \n')
   zc=zjcj(1:end-1);
   [Enter_val, pvt_col]= min(zc) 
   if all(A(:,pvt_col)<=0)
    error('LPP is Unbounded all enteries are <=0 in column %d',pvt_col);
   else
       sol=A(:,end)
       column=A(:,pvt_col)
       for i=1:size(A,1)
         if column(i)>0
            ratio(i)= sol(i)./column(i)
         else
            ratio(i)=inf
         end
       end
       [leaving_val, pvt_row]=min(ratio)
   end
bv(pvt_row)=pvt_col
pvt_key=A(pvt_row, pvt_col)
A(pvt_row,:)=A(pvt_row,:)./pvt_key
for i=1:size(A,1)
    if i~=pvt_row
        A(i,:)=A(i,:)-A(i, pvt_col).*A(pvt_row,:)
    end
end
    zjcj=zjcj-zjcj(pvt_col).*A(pvt_row,:)
    zcj=[zjcj;A]
    table=array2table(zcj)
    table.Properties.VariableNames(1:size(zcj,2))={'x_1','x_2','s_1','s_2','s_3','sol'}
else
    RUN=false;
    fprintf('The current BFS is optimal \n')
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
        fprintf('  The current BFS is not optimal\n')
        [ent_col,pvt_col]=min(ZC)
        fprintf('Entering Col =%d \n' , pvt_col);
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
            fprintf('leaving Row=%d \n', pvt_row);
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
        fprintf('  Current BFS is Optimal \n');
    end
end

`

const stepdesc = `clc;
clear all;
format short

a=0
b=0

%syms x y a1;
%f= @(x,y) x*y;

f= @(x,y) 3*x^2-4*x*y+2*y^2+4*x+6

grad=@(x,y) [6*x-4*y+4 , -4*x+4*y]
for k=1:4
grad(a,b)
d=-grad(a,b)/norm(grad(a,b))

%fun=@(z)(a+z*d(1))*(b+z*d(2)) 
fun=@(z) 3*(a+z*d(1))^2-4*(a+z*d(1))*(b+z*d(2))+2*(b+z*d(2))^2+4*(a+z*d(1))+6;
x1 = fminbnd(fun,0,10000)
a=a+x1*d(1)
b=b+x1*d(2)
f(a,b)
end

`
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
app.get('/step', (req, res) =>{
    res.type('text/plain'); 
    res.send(stepdesc); 
})
// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
