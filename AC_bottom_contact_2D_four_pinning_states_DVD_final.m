%% 2D constrained Allen-Cahn with dynamic boundary: four pinned states
% Spatial discretization is the SAME discrete-energy / DVD-type discretization
% used in time.m:
%   - x periodic second-order difference;
%   - y gradient-energy stiffness with endpoint rows;
%   - trapezoidal bulk quadrature in y;
%   - dynamic-boundary metric m_0 = 1 + epsilon*hy/2;
%   - top Neumann metric m_N = epsilon*hy/2;
%   - AVF discrete gradients in time;
%   - exact discrete H_h constraint.
%
% The computational domain and the chemically patterned substrate are kept
% from AC_bottom_contact_2D_multiple_metastable_states(3).m.
% The first three initial states are unchanged.  A fourth centered,
% over-spread droplet is added to test receding contact-line pinning.
% Its final pinned state is not prescribed by the initialization.
%
% This script performs ONLY the computation and saves the data.
% Plot with: fig_plot_pinning_four_cases_final.m

clear; clc; close all;

%% Parameters
Nx = 256;
Ny = 128;
Ny1 = Ny + 1;

Lx = 1.0;
Ly = 1.0;
left   = -Lx;
right  =  Lx;
bottom = 0.0;
top    = Ly;

hx = 2*Lx/Nx;
hy = Ly/Ny;

epsilon = 0.02;
tau = 1.0e-2;
T = 20;
Nt = round(T/tau);
T = Nt*tau;

maxNewton = 25;
tolNewton = 1.0e-10;

angleEvery = 5;

%% Grid
x = linspace(left,right-hx,Nx);
y = linspace(bottom,top,Ny1);
[X,Y] = meshgrid(x,y);
Ntot = Nx*Ny1;

%% Chemical pattern: k = 2
% Material A: theta_A = 120 deg
% Material B: theta_B = 60 deg
thetaA = 2*pi/3;
thetaB = pi/3;

thetaY = thetaA*ones(size(x));

isB = ...
    ((x >= -0.58) & (x < -0.46)) | ...
    ((x >= -0.34) & (x < -0.22)) | ...
    ((x >=  0.22) & (x <  0.34)) | ...
    ((x >=  0.46) & (x <  0.58));

thetaY(isB) = thetaB;
ctheta = cos(thetaY);
ctheta_b = ctheta(:);

junctions = [-0.58,-0.46,-0.34,-0.22,0.22,0.34,0.46,0.58];

%% ===============================================================
%% Spatial discretization: exactly the time.m logic
%% ===============================================================

% x direction: periodic
ex = ones(Nx,1);
Ax = spdiags([-ex,2*ex,-ex],[-1,0,1],Nx,Nx)/hx^2;
Ax(1,Nx) = -1/hx^2;
Ax(Nx,1) = -1/hx^2;

% y direction: gradient-energy stiffness
ey = ones(Ny1,1);
Ay = spdiags([-ey,2*ey,-ey],[-1,0,1],Ny1,Ny1)/hy^2;
Ay(1,1) = 1/hy^2;
Ay(end,end) = 1/hy^2;

% trapezoidal weights in y
wy = ones(Ny1,1);
wy(1) = 0.5;
wy(end) = 0.5;
Wy = spdiags(wy,0,Ny1,Ny1);

% discrete gradient-energy matrix
Ix = speye(Nx);
Kgrad = epsilon*hx*hy * ...
    (kron(Ax,Wy) + kron(Ix,Ay));

% bulk quadrature weights
OmegaWeight = hx*hy*repmat(wy,Nx,1);

% dynamic-boundary metric matrix
my = epsilon*hy*ones(Ny1,1);
my(1) = 1 + epsilon*hy/2;
my(end) = epsilon*hy/2;

Mvec = hx*repmat(my,Nx,1);
M = spdiags(Mvec,0,Ntot,Ntot);

% bottom boundary indices
bottom_id = 1:Ny1:Ntot;

%% Energy / invariant / mass
Efun = @(U) discrete_energy_2d(U,hx,hy,wy,epsilon,ctheta);
Hfun = @(U) discrete_H_2d(U,hx,hy,wy);
Massfun = @(U) discrete_mass_2d(U,hx,hy,wy);

%% Four initial cap templates with the same sharp-interface area
A_target = 0.30;

caseName = { ...
    'I: symmetric narrow', ...
    'II: heterogeneous wetting', ...
    'III: symmetric wide', ...
    'IV: over-spread receding'};

d_case2 = sqrt(2*A_target/pi);

xc0 = [0.00, 0.06, 0.00, 0.00];
d0  = [0.31, d_case2, 0.55, 0.70];
Ncase = numel(caseName);

theta0 = zeros(1,Ncase);
R0 = zeros(1,Ncase);
yc0 = zeros(1,Ncase);
Uraw = cell(Ncase,1);

for icase = 1:Ncase
    theta0(icase) = theta_from_area_halfwidth(A_target,d0(icase));
    R0(icase) = d0(icase)/sin(theta0(icase));
    yc0(icase) = -R0(icase)*cos(theta0(icase));

    phi0 = sqrt((X-xc0(icase)).^2 + (Y-yc0(icase)).^2) - R0(icase);
    Uraw{icase} = tanh(-phi0/(sqrt(2)*epsilon));
end

% Use Case I to define the common discrete H_h target.
Htarget = Hfun(Uraw{1});

% Correct Cases II--IV by a scalar level-set shift so all four states
% have the same DISCRETE H_h value.
Uinit = cell(Ncase,1);
levelShift = zeros(1,Ncase);
Uinit{1} = Uraw{1};

for icase = 2:Ncase
    phi0 = sqrt((X-xc0(icase)).^2 + (Y-yc0(icase)).^2) - R0(icase);
    [Uinit{icase},levelShift(icase)] = ...
        match_H_by_level_shift(phi0,Htarget,Hfun,epsilon);
end

%% Check initial states
fprintf('============================================================\n');
fprintf('Pinning test: four wetting dynamics on the same patterned substrate\n');
fprintf('Same substrate + same discrete H_h + time.m spatial discretization\n');
fprintf('Nx=%d, Ny=%d, eps=%.4g, tau=%.4g, T=%.4g\n', ...
        Nx,Ny,epsilon,tau,T);
fprintf('theta_A=%.1f deg, theta_B=%.1f deg\n', ...
        thetaA*180/pi,thetaB*180/pi);
fprintf('target sharp area A=%.4f\n',A_target);
fprintf('common discrete H target = %.16e\n\n',Htarget);

for icase = 1:Ncase
    [xl0,xr0] = contact_points_bottom(Uinit{icase},x);
    fprintf('%s\n',caseName{icase});
    fprintf('  cap theta0       = %.8f deg\n',theta0(icase)*180/pi);
    fprintf('  nominal contacts = (%.5f, %.5f)\n', ...
            xc0(icase)-d0(icase),xc0(icase)+d0(icase));
    fprintf('  actual contacts  = (%.8f, %.8f)\n',xl0,xr0);
    fprintf('  level shift      = %+.8e\n',levelShift(icase));
    fprintf('  H-Htarget        = %.3e\n',Hfun(Uinit{icase})-Htarget);
    fprintf('  E(initial)       = %.16e\n\n',Efun(Uinit{icase}));
end

%% Snapshot settings
snapshot_times = [0,0.005,0.5,1]*T;
snapshot_steps = round(snapshot_times/tau);
snapshot_steps(1) = 0;
snapshot_steps(end) = Nt;

%% Storage for all four runs
results = struct([]);

for icase = 1:Ncase

    fprintf('\n============================================================\n');
    fprintf('Running %s\n',caseName{icase});
    fprintf('============================================================\n');

    u = Uinit{icase};
    uvec = u(:);

    Ehist = zeros(Nt+1,1);
    Hhist = zeros(Nt+1,1);
    Masshist = zeros(Nt+1,1);
    Lamhist = zeros(Nt,1);
    NewtonHist = zeros(Nt,1);
    SpeedHist = zeros(Nt,1);

    LeftContactHist = nan(Nt+1,1);
    RightContactHist = nan(Nt+1,1);
    ThetaAppHist = nan(Nt+1,1);
    MinHist = nan(Nt+1,1);
    MaxHist = nan(Nt+1,1);

    Ehist(1) = Efun(u);
    Hhist(1) = Hfun(u);
    Masshist(1) = Massfun(u);

    [LeftContactHist(1),RightContactHist(1)] = ...
        contact_points_bottom(u,x);
    ThetaAppHist(1) = apparent_contact_angle(u,x,y,hy);
    MinHist(1) = min(u(:));
    MaxHist(1) = max(u(:));

    H0 = Htarget;
    lambda_old = 0;

    Usnap = cell(length(snapshot_steps),1);
    Usnap{1} = u;
    snap_id = 2;

    %% Time integration: same Newton structure as time.m
    for n = 1:Nt
        uold = uvec;
        unew = uold;
        lambda = lambda_old;
        converged = false;

        for iter_step = 1:maxNewton
            [F1,F2,J11,J12,J21] = nonlinear_system( ...
                unew,uold,lambda, ...
                M,Mvec,Kgrad,OmegaWeight, ...
                bottom_id,hx,epsilon,ctheta_b,H0,tau);

            res = residual_norm(F1,F2,Mvec);

            if res < tolNewton
                converged = true;
                break
            end

            J = [J11,J12; J21,sparse(1,1)];
            delta = J\[-F1;-F2];

            unew = unew + delta(1:end-1);
            lambda = lambda + delta(end);
        end

        if ~converged
            warning('%s: Newton does not converge at step %d, residual=%.3e.', ...
                    caseName{icase},n,res);
        end

        SpeedHist(n) = sqrt(sum(Mvec.*(((unew-uold)/tau).^2)));

        uvec = unew;
        lambda_old = lambda;

        Umat = reshape(uvec,Ny1,Nx);

        Ehist(n+1) = Efun(Umat);
        Hhist(n+1) = Hfun(Umat);
        Masshist(n+1) = Massfun(Umat);
        Lamhist(n) = lambda;
        NewtonHist(n) = iter_step;

        [LeftContactHist(n+1),RightContactHist(n+1)] = ...
            contact_points_bottom(Umat,x);

        if mod(n,angleEvery) == 0 || n == Nt
            ThetaAppHist(n+1) = apparent_contact_angle(Umat,x,y,hy);
        end

        MinHist(n+1) = min(Umat(:));
        MaxHist(n+1) = max(Umat(:));

        if snap_id <= length(snapshot_steps) && n == snapshot_steps(snap_id)
            Usnap{snap_id} = Umat;
            snap_id = snap_id + 1;
        end

        if mod(n,max(1,round(Nt/10))) == 0 || n == 1
            fprintf(['n=%5d/%5d, Newton=%2d, lambda=%+.5e, ', ...
                     'E=%.12e, H-H0=%.3e, contacts=(%.5f,%.5f)\n'], ...
                    n,Nt,iter_step,lambda,Ehist(n+1), ...
                    Hhist(n+1)-H0, ...
                    LeftContactHist(n+1),RightContactHist(n+1));
        end
    end

    u = reshape(uvec,Ny1,Nx);
    time = (0:Nt)'*tau;

    overshootUpper = max(MaxHist-1,0);
    overshootLower = max(-1-MinHist,0);
    MBPerr = max(overshootUpper,overshootLower);

    thetaFinal = apparent_contact_angle(u,x,y,hy);

    results(icase).name = caseName{icase};
    results(icase).u0 = Uinit{icase};
    results(icase).uT = u;
    results(icase).Usnap = Usnap;
    results(icase).Ehist = Ehist;
    results(icase).Hhist = Hhist;
    results(icase).Masshist = Masshist;
    results(icase).Lamhist = Lamhist;
    results(icase).NewtonHist = NewtonHist;
    results(icase).SpeedHist = SpeedHist;
    results(icase).LeftContactHist = LeftContactHist;
    results(icase).RightContactHist = RightContactHist;
    results(icase).ThetaAppHist = ThetaAppHist;
    results(icase).MinHist = MinHist;
    results(icase).MaxHist = MaxHist;
    results(icase).MBPerr = MBPerr;
    results(icase).thetaFinal = thetaFinal;

    fprintf('\nFinal diagnostics for %s:\n',caseName{icase});
    fprintf('E(0)               = %.16e\n',Ehist(1));
    fprintf('E(T)               = %.16e\n',Ehist(end));
    fprintf('E(T)-E(0)          = %.16e\n',Ehist(end)-Ehist(1));
    fprintf('max |H-H0|         = %.16e\n',max(abs(Hhist-H0)));
    fprintf('final left contact = %.8f\n',LeftContactHist(end));
    fprintf('final right contact= %.8f\n',RightContactHist(end));
    if isfinite(LeftContactHist(end)) && isfinite(RightContactHist(end))
        distL = min(abs(junctions-LeftContactHist(end)));
        distR = min(abs(junctions-RightContactHist(end)));
        fprintf('distance to nearest junction = (%.3e, %.3e)\n',distL,distR);
    end
    fprintf('final apparent angle = %.8f deg\n',thetaFinal);
    fprintf('final speed norm   = %.16e\n',SpeedHist(end));
    fprintf('max MBP overshoot  = %.16e\n',max(MBPerr));
    fprintf('mean Newton        = %.3f\n',mean(NewtonHist));
end

%% Summary table in command window
fprintf('\n\n');
fprintf('================================================================================================================\n');
fprintf('SUMMARY: same patterned substrate + same discrete H_h\n');
fprintf('================================================================================================================\n');
fprintf('%-24s %12s %12s %14s %14s %14s\n', ...
        'case','x_L(T)','x_R(T)','E(T)','max|H-H0|','speed(T)');
fprintf('----------------------------------------------------------------------------------------------------------------\n');

for icase = 1:Ncase
    fprintf('%-24s %12.6f %12.6f %14.7e %14.3e %14.3e\n', ...
        caseName{icase}, ...
        results(icase).LeftContactHist(end), ...
        results(icase).RightContactHist(end), ...
        results(icase).Ehist(end), ...
        max(abs(results(icase).Hhist-Htarget)), ...
        results(icase).SpeedHist(end));
end
fprintf('================================================================================================================\n');

%% Save all data for a separate plotting program
save('pinning_four_cases_DVD_data.mat', ...
    'x','y','X','Y','left','right','bottom','top', ...
    'Nx','Ny','hx','hy','epsilon','tau','T','time', ...
    'thetaA','thetaB','thetaY','ctheta','junctions', ...
    'A_target','caseName','xc0','d0','theta0','R0','yc0', ...
    'Htarget','levelShift','snapshot_times','results');

fprintf('\nSaved: pinning_four_cases_DVD_data.mat\n');

%% ===============================================================
%% Local functions
%% ===============================================================

function [F1,F2,J11,J12,J21] = nonlinear_system( ...
    u,uold,lambda, ...
    M,Mvec,Kgrad,OmegaWeight, ...
    bottom_id,hx,epsilon,ctheta_b,H0,dt)

n = length(u);

% AVF discrete variational derivative of the energy
umid = (u+uold)/2;
Ehat_grad = Kgrad*umid;

What = ...
    1/4*(u.^3 + u.^2.*uold + u.*uold.^2 + uold.^3) ...
    - 1/2*(u+uold);
Ehat_pot = OmegaWeight.*(What/epsilon);

Ehat_bdy = zeros(n,1);
Ehat_bdy(bottom_id) = hx*ghat( ...
    u(bottom_id),uold(bottom_id),ctheta_b);

Ehat = Ehat_grad + Ehat_pot + Ehat_bdy;

% AVF discrete gradient of H
Hhat = OmegaWeight.*Hhat_point(u,uold);

% nonlinear system
F1 = M*(u-uold)/dt + Ehat - epsilon*lambda*Hhat;
F2 = discrete_H_vec(u,OmegaWeight)-H0;

% Jacobian of Ehat
dWhat = 1/4*(3*u.^2 + 2*u.*uold + uold.^2) - 1/2;

dE = 1/2*Kgrad + ...
    spdiags(OmegaWeight.*dWhat/epsilon,0,n,n);

db = zeros(n,1);
db(bottom_id) = hx*ghat_du( ...
    u(bottom_id),uold(bottom_id),ctheta_b);
dE = dE + spdiags(db,0,n,n);

% Jacobian of Hhat
dHhat_point = -sqrt(2)/6*(2*u+uold);
dHhat = spdiags(OmegaWeight.*dHhat_point,0,n,n);

% Jacobian blocks
J11 = M/dt + dE - epsilon*lambda*dHhat;
J12 = sparse(-epsilon*Hhat);

Hprime = sqrt(2)/2*(1-u.^2);
J21 = sparse((OmegaWeight.*Hprime).');

end

function H = discrete_H_vec(u,OmegaWeight)
    H = sum(OmegaWeight.*Hfun_point(u));
end

function r = residual_norm(F1,F2,Mvec)
    r = sqrt(sum(F1.^2./Mvec) + F2^2);
end

function theta = theta_from_area_halfwidth(A,d)
    % Circular cap above y=0:
    % A = R^2(theta-sin(theta)cos(theta)), d=R sin(theta).
    f = @(th) d^2*(th-sin(th)*cos(th))/(sin(th)^2) - A;

    lo = 5*pi/180;
    hi = 175*pi/180;
    flo = f(lo);
    fhi = f(hi);

    if flo*fhi > 0
        error('Cannot bracket theta for A=%.4g and d=%.4g.',A,d);
    end

    for k = 1:80
        mid = 0.5*(lo+hi);
        fm = f(mid);
        if flo*fm <= 0
            hi = mid;
            fhi = fm;
        else
            lo = mid;
            flo = fm;
        end
    end

    theta = 0.5*(lo+hi);
end

function [U,shift] = match_H_by_level_shift(phi,Htarget,Hfun,epsilon)
    % U(shift)=tanh(-(phi-shift)/(sqrt(2)*epsilon)).
    % Positive shift expands the liquid region.
    fun = @(s) Hfun(tanh(-(phi-s)/(sqrt(2)*epsilon))) - Htarget;

    sL = -0.10;
    sR =  0.10;
    fL = fun(sL);
    fR = fun(sR);

    count = 0;
    while fL*fR > 0 && count < 12
        sL = 1.5*sL;
        sR = 1.5*sR;
        fL = fun(sL);
        fR = fun(sR);
        count = count + 1;
    end

    if fL*fR > 0
        error('Failed to bracket the level shift for matching H.');
    end

    for k = 1:80
        sm = 0.5*(sL+sR);
        fm = fun(sm);

        if abs(fm) < 1e-14
            shift = sm;
            U = tanh(-(phi-shift)/(sqrt(2)*epsilon));
            return
        end

        if fL*fm <= 0
            sR = sm;
            fR = fm;
        else
            sL = sm;
            fL = fm;
        end
    end

    shift = 0.5*(sL+sR);
    U = tanh(-(phi-shift)/(sqrt(2)*epsilon));
end

function val = Wfun(u)
    val = 0.25*(u.^2-1).^2;
end

function val = Hfun_point(u)
    val = sqrt(2)/2*(u-u.^3/3);
end

function val = Hhat_point(u,v)
    val = sqrt(2)/2*(1-(u.^2+u.*v+v.^2)/3);
end

function val = gfun(u,ctheta)
    val = -sqrt(2)/6 .* ctheta .* (3*u-u.^3);
end

function val = ghat(u,v,ctheta)
    val = -sqrt(2)/2*ctheta ...
        + sqrt(2)/6*ctheta.*(u.^2+u.*v+v.^2);
end

function val = ghat_du(u,v,ctheta)
    val = sqrt(2)/6*ctheta.*(2*u+v);
end

function E = discrete_energy_2d(U,hx,hy,wy,epsilon,ctheta)
    [~,Nx] = size(U);

    Uxp = U(:,[2:Nx,1]);
    dxU = (Uxp-U)/hx;

    Ex = epsilon/2 * hx*hy * ...
        sum(sum((wy.*ones(1,Nx)).*dxU.^2));

    dyU = (U(2:end,:)-U(1:end-1,:))/hy;
    Ey = epsilon/2 * hx*hy * sum(dyU(:).^2);

    Epot = (hx*hy/epsilon) * ...
        sum(sum((wy.*ones(1,Nx)).*Wfun(U)));

    Eb = hx * sum(gfun(U(1,:),ctheta));

    E = Ex + Ey + Epot + Eb;
end

function H = discrete_H_2d(U,hx,hy,wy)
    [~,Nx] = size(U);
    H = hx*hy * ...
        sum(sum((wy.*ones(1,Nx)).*Hfun_point(U)));
end

function Mass = discrete_mass_2d(U,hx,hy,wy)
    [~,Nx] = size(U);
    Mass = hx*hy * ...
        sum(sum((wy.*ones(1,Nx)).*U));
end

function [xl,xr] = contact_points_bottom(U,x)
    ub = U(1,:);
    Nx = length(x);
    xcross = [];

    for i = 1:Nx-1
        if ub(i) == 0
            xcross = [xcross,x(i)]; %#ok<AGROW>
        elseif ub(i)*ub(i+1) < 0
            xx = x(i) - ub(i)*(x(i+1)-x(i))/(ub(i+1)-ub(i));
            xcross = [xcross,xx]; %#ok<AGROW>
        end
    end

    if length(xcross) < 2
        xl = NaN;
        xr = NaN;
        return
    end

    xl = min(xcross);
    xr = max(xcross);
end

function thetaDeg = apparent_contact_angle(U,x,y,hy)
    [xp,yp] = zero_contour_points(U,x,y);

    if isempty(xp)
        thetaDeg = NaN;
        return
    end

    keep = yp > 2*hy;
    xp = xp(keep);
    yp = yp(keep);

    if length(xp) < 6
        thetaDeg = NaN;
        return
    end

    A = [2*xp(:),2*yp(:),ones(length(xp),1)];
    rhs = xp(:).^2 + yp(:).^2;
    coef = A\rhs;

    xc = coef(1);
    yc = coef(2);
    R2 = coef(3) + xc^2 + yc^2;

    if R2 <= 0
        thetaDeg = NaN;
        return
    end

    R = sqrt(R2);
    c = -yc/R;
    c = max(-1,min(1,c));
    thetaDeg = acos(c)*180/pi;
end

function [xp,yp] = zero_contour_points(U,x,y)
    C = contourc(x,y,U,[0 0]);

    xp = [];
    yp = [];

    if isempty(C)
        return
    end

    k = 1;
    nbest = 0;

    while k < size(C,2)
        npts = C(2,k);

        if npts <= 0 || k+npts > size(C,2)
            break
        end

        pts = C(:,k+1:k+npts);

        if npts > nbest
            xp = pts(1,:);
            yp = pts(2,:);
            nbest = npts;
        end

        k = k+npts+1;
    end
end
