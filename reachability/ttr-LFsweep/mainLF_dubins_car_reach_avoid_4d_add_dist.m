function phi = mainLF_dubins_car_reach_avoid_4d(dx, N, Min, Max, a_max, omega_max, d_max_xy, d_max_theta, d_max_alpha, d_max_beta)

epsilon = 1e-6;

dim = 4;

xs = gridGeneration4d(dim, Min, Max, dx, N);

% Retrieve Obstacles
data = load('/local-scratch/anjianl/project/WayPtNav/reachability/data_tmp/reach_avoid_map_4d.mat');
phi = data.goal_reach_avoid_map_4d;
obs = data.obstacle_reach_avoid_map_4d;

% disp(dx);
% disp(size(xs));
% disp(size(phi));
% disp(size(obs));


%LF sweeping
% mex mexLFsweep_dubins_car_reach_avoid_4d_add_dist.cpp;
% disp('dubins car 4d reach avoid mexing done!');

numIter = 50;
TOL = 0.5;

startTime = cputime;
% tic;
mexLFsweep_dubins_car_reach_avoid_4d_add_dist(phi,xs,dx,a_max,omega_max,d_max_xy,d_max_theta,d_max_alpha,d_max_beta,numIter,TOL,obs);
% toc;


endTime = cputime;
fprintf('TTR computation takes %g seconds\n', endTime - startTime);

% Augment theta dimension. copy theta 0 afterwards
phi = cat(3, phi, phi(:,:,1,:));


end
