function phi = mainLF_dubins_car_avoid_4d_dxdy_circular(dx, N, Min, Max, a_max, omega_max, d_max_xy, d_max_theta, data_tmp_path)

dim = 4;

xs = gridGeneration4d(dim, Min, Max, dx, N);

% Retrieve Obstacles
data = load(data_tmp_path);
phi = data.obstacle_avoid_map_4d;

% disp(dx);
% disp(size(xs));
% disp(size(phi));
% disp(size(obs));


%LF sweeping
% mex mexLFsweep_dubins_car_avoid_4d_dxdy_circular.cpp;
% disp('dubins car 4d avoid mexing done!');

numIter = 50;
TOL = 0.5;

% startTime = cputime;
% tic;
mexLFsweep_dubins_car_avoid_4d_dxdy_circular(phi,xs,dx,a_max,omega_max,d_max_xy,d_max_theta,numIter,TOL);
% toc;

% endTime = cputime;
% fprintf('4d avoid TTR computation takes %g seconds\n', endTime - startTime);

% Augment theta dimension. copy theta 0 afterwards
phi = cat(3, phi, phi(:,:,1,:));

end
