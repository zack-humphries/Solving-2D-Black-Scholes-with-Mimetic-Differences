%% 2D Black-Scholes PDE
% Zachary Humphries
% COMP 797 - Dr. Miguel Dumett
% Spring 2024

clear all;
close all;
clc;

 addpath('mole\mole_MATLAB','-end')


%% parameters depend on strike price value
T = 1;
strike = 1;                                 % Strike Price
omega = [0.3 0.05; 0.05 0.3];

r = 0.1;                                    % Risk free interest rate

k = 4;                                      % Order of accuracy

a = 0;                                      % Minimum Value of Option for Asset X (must be zero)
b = round(10*strike);                       % Maximum Value of Option for Asset X (Recommended between 8*strike and 12*strike)
c = 0;                                      % Minimum Value of Option for Asset Y (must be zero)
d = b;                                      % Maximum Value of Option for Asset X

m = (2*k)^2 +1;                                  % 2*k+1 = Minimum number of cells to attain the desired accuracy
n = m;                                      % Number of cells along the y-axis

dx = (b-a)/m;                               % Step length along the x-axis
dy = (d-c)/n;                               % Step length along the y-axis

dt = 0.05*((1/dx)^2 + (1/dy)^2)^(-1);       % Von Neumann stability criterion for explicit scheme dx^2/(4*alpha)
dt = -dt;

%% mesh
xgrid = [a a+dx/2 : dx : b-dx/2 b];
ygrid = [c c+dy/2 : dy : d-dy/2 d];
[X, Y] = meshgrid(xgrid, ygrid);

%% initial condition
F = max(((X+Y)/2)-strike, 0); 

Fvec = reshape(F', (m+2)*(n+2), 1);
Xvec = reshape(X', (m+2)*(m+2), 1);
Yvec = reshape(Y', (n+2)*(n+2), 1);

%% operator matrix

small_I = speye((m+2)*(n+2), (m+2)*(n+2));
I = speye((m+2)*(n+2)+(m+2)*(n+2), (m+2)*(n+2)+(m+2)*(n+2));

XI = spdiags(Xvec, 0, (m+2)*(n+2), (m+2)*(n+2));
YI = spdiags(Yvec, 0, (m+2)*(n+2), (m+2)*(n+2));

XYI = spdiags([Xvec; Yvec], 0, (m+2)*(n+2)+(m+2)*(n+2), (m+2)*(n+2)+(m+2)*(n+2));

XY = [XI,YI];

O = [omega(1,1).^2 *small_I, omega(1,2).^2*small_I; omega(2,1).^2*small_I, omega(2,2).^2*small_I]; % MD

%% Setting Up Gradient and Divergence Matricies

G = grad2D(k, m, dx, n, dy);
D = div2D(k, m, dx, n, dy);

%% Setting Up Interpolation Matrices

IFC = interpolFacesToCentersG2D(k, m, n);
ICF = interpolCentersToFacesD2D(k, m, n);

%% Combine for Black-Scholes Matrix

A = -(r*XY*IFC*G) - 0.5*(D*ICF*XYI*O*XYI*IFC*G) + r*small_I;


%% Plot Initial Conditions
figure(1)
plotBlackScholes(Fvec, X, Y, m, n, T, a, b, c, d, strike);

%% auxiliary variablesb, d
len = length(T : dt : 0);
Fsol = zeros(numel(Fvec), len); % to store solutions
Fsol(:,1) = Fvec;

Faux = Fvec; % to jump start time discretization

%% Forward Euler (First Order)

[farFieldX, farFieldY] = farFieldBC(b, d, xgrid, ygrid, strike, r, T+dt);

[closeFieldX,closeFieldY] = closeFieldBC_Order1(k, m, n, b, d, dx, dy, dt, xgrid, ygrid, strike, r, omega, Faux, T+dt);
Faux = (small_I - dt*A)\(Fsol(:,1));

Faux(1:m+2:(end-(m+1))) = closeFieldX;
Faux(1:m+2) = closeFieldX;

Faux(m+2:m+2:end) = farFieldY;
Faux(end-m-1:end) = farFieldX;
Faux(1) = 0;

Fsol(:,2) = Faux;

%% Forward Euler
count = 2;
for t = T+(2*dt) : dt : 0

    [farFieldX, farFieldY] = farFieldBC(b, d, xgrid, ygrid, strike, r, t);
    [closeFieldX,closeFieldY] = ...
        closeFieldBC_Order2(k, m, n, b, d, dx, dy, dt, xgrid, ygrid, strike, r, omega, Fsol(:, count-1), Fsol(:, count), t);

    Faux = (small_I - ((2*dt)/3)*A)\((4/3)*Fsol(:,count) - (1/3)*Fsol(:, count-1));
    

    % Adjust Faux to Account for Close-Field and FarField Boundary Conditions
    Faux(1:m+2:(end-(m+1))) = closeFieldX;
    Faux(1:n+2) = closeFieldY;
    
    Faux(m+2:m+2:end) = farFieldY;
    Faux(end-m-1:end) = farFieldX;
    Faux(1) = 0;

%    plotBlackScholes(Faux, X, Y, m, n, t, a, b, c, d, strike);

    count = count+1;
    Fsol(:,count) = Faux;

end

figure(2)
plotBlackScholes(Faux, X, Y, m, n, 0, a, b, c, d, strike);


%% Far-Field Coundary Condition

function [farFieldX, farFieldY] = farFieldBC(b, d, xgrid, ygrid, strike, r, time)

    farFieldX = (b + ygrid)/2 - strike*exp(-r*(time));
    farFieldY = (d + xgrid)/2 - strike*exp(-r*(time));  
end


%% Close-Field Boundary Condition

function [closeFieldX, closeFieldY] = ...
    closeFieldBC_Order1(k, m, n, b, d, dx, dy, dt, xgrid, ygrid, strike, r, omega, Faux, time)

    Gx = grad(k, m, dx);
    Gy = grad(k, n, dy);

    IFCx = interpolFacesToCentersG1D(k,m);
    IFCy = interpolFacesToCentersG1D(k,n);

    Lx = lap(k, m, dx);
    Ly = lap(k, n, dy);

    Ix = speye((m+2), (m+2));
    Iy = speye((n+2), (n+2));

    XI = diag(xgrid);
    YI = diag(ygrid);

    Mx = -(r*XI*IFCx*Gx) - (omega(1,1)*omega(1,1)/2 * XI*XI * Lx) + r*Ix;
    closeFieldX = (Ix-dt*Mx)\Faux(1:m+2:(end-(m+1)));
    closeFieldX(end) = b/2 - strike*exp(-r*(time));

    My = -(r*YI*IFCy*Gy) - (omega(2,2)*omega(2,2)/2 * YI*YI * Ly) + r*Iy;
    closeFieldY = (Ix-dt*My)\Faux(1:n+2);
    closeFieldY(end) = d/2 - strike*exp(-r*(time));
end

function [closeFieldX, closeFieldY] = ...
    closeFieldBC_Order2(k, m, n, b, d, dx, dy, dt, xgrid, ygrid, strike, r, omega, Faux_minus, Faux, time)
    
    Gx = grad(k, m, dx);
    Gy = grad(k, n, dy);

    IFCx = interpolFacesToCentersG1D(k,m);
    IFCy = interpolFacesToCentersG1D(k,n);

    Lx = lap(k, m, dx);
    Ly = lap(k, n, dy);

    Ix = speye((m+2), (m+2));
    Iy = speye((n+2), (n+2));

    XI = diag(xgrid);
    YI = diag(ygrid);
 
    Mx = -(r*XI*IFCx*Gx) - (omega(1,1)*omega(1,1)/2 * XI*XI * Lx) + r*Ix;
    closeFieldX = (Ix - ((2*dt)/3)*Mx)\((4/3)*Faux(1:m+2:(end-(m+1))) - (1/3)*Faux_minus(1:m+2:(end-(m+1))));
    closeFieldX(end) = b/2 - strike*exp(-r*(time));

    My = -(r*YI*IFCy*Gy) - (omega(2,2)*omega(2,2)/2 * YI*YI * Ly) + r*Iy;
    closeFieldY = (Iy - ((2*dt)/3)*My)\((4/3)*Faux(1:n+2) - (1/3)*Faux_minus(1:n+2));
    closeFieldY(end) = d/2 - strike*exp(-r*(time));
end

function plotBlackScholes(Fvec, X, Y, m, n, t, a, b, c, d, strike)
    
    surf(X, Y, reshape(Fvec, m+2, n+2))
    title(['2D Black-Scholes \newlineTime = ' num2str(t, '%1.4f')])
    xlabel('x')
    ylabel('y')
    zlabel('F')
    colorbar
    axis([a b c d]);
%    axis([0 (5*strike/3) 0 (5*strike/3) 0 5/3 *strike])
    drawnow
end

