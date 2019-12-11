clear
clc
close all
addpath('helper_functions')
% addpath('own_helper_functions')

%%
% Setup
% path to the validation images folder
test_img_dir = 'data/tracking/test/img';
valid_img_dir = 'data/tracking/validation/img';

% path to object ply file
object_path = 'data/teabox.ply';
% path to results folder
test_results_dir = 'data/tracking/test/results';
valid_results_dir = 'data/tracking/validation/results';

% Read the object's geometry
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);
faces = faces + 1;

% Create directory for results
if ~exist(test_results_dir,'dir')
    mkdir(test_results_dir);
end

if ~exist(valid_results_dir,'dir')
    mkdir(valid_results_dir);
end

% Load Ground Truth camera poses for the validation sequence
% Camera orientations and locations in the world coordinate system
load('gt_valid.mat')

% TODO: setup camera parameters (camera_params) using cameraParameters()
focalLength = 2960.37845;
principalPoint_cx = 1841.68855;
principalPoint_cy = 1235.23369;
imageSize = [2456 3680];
camera_params = cameraParameters('IntrinsicMatrix',[focalLength 0 0; ...
    0 focalLength 0; principalPoint_cx principalPoint_cy 1], ...
    'ImageSize',imageSize);

% Get all filenames in images folder
FolderInfo = dir(fullfile(valid_img_dir, '*.JPG'));
Filenames = fullfile(valid_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);


% Detect SIFT keypoints in all images
%   ___             ___ ___ ___ _____
%  | _ \_  _ _ _   / __|_ _| __|_   _|
%  |   / || | ' \  \__ \| || _|  | |
%  |_|_\\_,_|_||_| |___/___|_|   |_|

% Place SIFT keypoints and corresponding descriptors for all images here
rerun_sift = 0;
sift_files = ["sift_descriptors.mat", "sift_keypoints.mat"];
all_files_exist = 1;
for i=1:size(sift_files,2)
    all_files_exist = all_files_exist && isfile(sift_files(i));
end

if all_files_exist && ~rerun_sift
    for i=1:size(sift_files,2)
        load(sift_files(i));
    end
else
    % Place SIFT keypoints and corresponding descriptors for all images here
    keypoints = cell(num_files,1);
    descriptors = cell(num_files,1);
    
    for i=1:length(Filenames)
        fprintf('Calculating sift features for image: %d \n', i)
        
        %    TODO: Prepare the image (img) for vl_sift() function
        I = imread(Filenames{i});
        img = single(rgb2gray(I));
        [keypoints{i}, descriptors{i}] = vl_sift(img) ;
    end
    
    % Save sift features and descriptors and load them when you rerun the code to save time
    save('sift_descriptors.mat', 'descriptors')
    save('sift_keypoints.mat', 'keypoints')
end

%% Initialize Pose of First image with Labeled corners for better initialization

% Label images
num_points = 8;
relabel = 0;
 
if isfile('labeled_points.mat') && ~relabel
    load('labeled_points.mat');
else
    % We only need to do it for the first image
    labeled_points = mark_image(Filenames{1}, num_points);
    
    % Save labeled points and load them when you rerun the code to save time
    save('labeled_points.mat', 'labeled_points');
end

% Call estimateWorldCameraPose to perform PnP

% Place estimated camera orientation and location here to use
% visualisations later
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

max_reproj_err = 4;

% We only need to do this for the first image
fprintf('Estimating pose for image: %d \n', 1)
% find index for entries with NaN values
index = find(isnan(labeled_points(:,1,1)));
% delete index in image and world points to ensure that 2d-3d correspondence is kept
image_points = labeled_points(:,:,1);
image_points(index, :) = [];
world_points = vertices;
world_points(index, :) = [];

[cam_in_world_orientations(:,:,1),cam_in_world_locations(:,:,1)] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', max_reproj_err);

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure()
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
hold on;
title(sprintf('Initial Image Camera Pose'));
%   Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
% points = worldToImage(camera_params, init_orientation, init_location, world_points);
for j=1:12
    plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
end
hold off;

save('workspace_vars.mat')
% load('workspace_vars.mat')
%% IRLS nonlinear optimisation

threshold_irls = 0.005; % update threshold for IRLS
max_iter = 50; % number of iterations
threshold_ubcmatch = 2.5; % matching threshold for vl_ubcmatch()


for i=1:(num_files-1)
    fprintf('Optimizing energy function between image %d and image %d\n', i, i+1)
    %   ___          _                 _        _   _
    %  | _ ) __ _ __| |___ __ _ _ ___ (_)___ __| |_(_)___ _ _
    %  | _ \/ _` / _| / / '_ \ '_/ _ \| / -_) _|  _| / _ \ ' \
    %  |___/\__,_\__|_\_\ .__/_| \___// \___\__|\__|_\___/_||_|
    %                   |_|         |__/
    %1) Project all SIFT keypoints from previous image,that have been matched, to 3D object. Use ray
    %intersection.
    
    R = cam_in_world_orientations(:,:,i);
    t = cam_in_world_locations(:,:,i);
    K = camera_params.IntrinsicMatrix;
    
    backProjected_point.coord3d = [];
    backProjected_point.descriptors = [];
    keypoint_3Dmodel = [];
    %indx_keypoints = [];
    
    P = K.' * [R -R*t.'];
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q; % this corresponds to C
    
    %===========================================
    %      _             _
    %   __| |_  ___  ___| |_   _ _ __ _ _  _ ___
    %  (_-< ' \/ _ \/ _ \  _| | '_/ _` | || (_-<
    %  /__/_||_\___/\___/\__| |_| \__,_|\_, /__/
    %                                   |__/
    % We need to back project all sift points found on image i
    % to find which one intersects with the 3D model
    for siftpoint_ = 1:size(keypoints{i},2)
        
        % Create homogeneous coordinate point for one sift point
        point = [keypoints{i}(1, siftpoint_) keypoints{i}(2, siftpoint_) 1];
        % calculate ray from camera center through random selected keypoint
        lambda = norm(inv(Q)*point.');
        ray_m = orig + lambda * inv(Q)*point.';
        direction = (ray_m - orig)/norm(ray_m -orig);
        
        % iterate through faces
        for face_=1:size(faces,1)
            % calculate intersection ray/face (same as in hw1.1_task1)
            [intersect, T, U, V, xcoor] = TriangleRayIntersection(orig', direction',...
                vertices(faces(face_, 1), :), vertices(faces(face_, 2), :), vertices(faces(face_, 3), :), ...
                'planeType','one sided');
            
            % save 3d intersection coordinates and descriptors of keypoint
            if intersect
                % keypoints that have intersected the 3D model
                keypoint_3Dmodel = [keypoint_3Dmodel, keypoints{i}(:,siftpoint_)];
                % index of those keypoints with respect to the keypoints
                % matrix
                % indx_keypoints = [indx_keypoints; h];
                backProjected_point.coord3d = [backProjected_point.coord3d; xcoor];
                backProjected_point.descriptors = [backProjected_point.descriptors descriptors{i}(:, siftpoint_)];
                %disp(xcoor);
            end
        end       
    end
    
    % Calculate the matches
    sift_matches{i+1} = vl_ubcmatch(backProjected_point.descriptors, descriptors{i+1}, threshold_ubcmatch);
    
    %================================================
    %   ___                   _        _   _
    %  | _ \___ _ __ _ _ ___ (_)___ __| |_(_)___ _ _
    %  |   / -_) '_ \ '_/ _ \| / -_) _|  _| / _ \ ' \
    %  |_|_\___| .__/_| \___// \___\__|\__|_\___/_||_|
    %          |_|         |__/
    % 3) Reproject these matches using corresponding 3D coordinates from the
    % step 1 and the initial camera pose back to the subsequent frame (image i+1)
    
    % Get the 3D coordinates of the backprojected points coming from the
    % initial frame
    backProjectedPoints_3Dcoord = backProjected_point.coord3d(sift_matches{i+1}(1,:),:);

    % Project match points from 3D model to the current frame (i+1) (2D image)
    reprojected_points_prev = project3d2image(backProjectedPoints_3Dcoord',camera_params, R, t);
%     reprojected_points = reprojected_points.';
    %========================================================
    %   ___      _ _   _      _   ___
    %  |_ _|_ _ (_) |_(_)__ _| | | __|_ _ _ _ ___ _ _
    %   | || ' \| |  _| / _` | | | _|| '_| '_/ _ \ '_|
    %  |___|_||_|_|\__|_\__,_|_| |___|_| |_| \___/_|
    % 4) Compute the reprojection error between pixels from SIFT
    % matches for the subsequent frame (image i+1) and from reprojected matches
    % from step 3
    image_points = [keypoints{i+1}(1,sift_matches{i+1}(2,:));keypoints{i+1}(2,sift_matches{i+1}(2,:))];
    [init_orientations,init_locations,inlieridx,~] = estimateWorldCameraPose(image_points', backProjectedPoints_3Dcoord, ...
        camera_params, 'MaxReprojectionError', 20);
    
    R = init_orientations;
    t = init_locations;

    % Project match points from 3D model to the current frame (i+1) (2D image)
    reprojected_points = project3d2image(backProjectedPoints_3Dcoord',camera_params, R, t);

    % Figure to see the comparison between the points that were matched
    % between the SIFT points of the previous and current frame, vs the
    % reprojected matched points from the 3D model.
        
%     figure(3)
%     imshow(char(Filenames(i+1)))
%     title(['Frame ', num2str(i+1), ' - Reprojected points from 3D model vs Matched points'])
%     hold on;
%     scatter(reprojected_points_prev(1,:)',reprojected_points_prev(2,:)','g*')    
%     scatter(reprojected_points(1,:)',reprojected_points(2,:)','rx')
%     scatter(image_points(1,:)',image_points(2,:)','bo')
%     legend('Reprojected points','2D correspondences with previous image')
%     hold off;


    %=======================================================================
    %    ___                     _          _____           _    _
    %   / __|___ _ __  _ __ _  _| |_ ___   _ | |__ _ __ ___| |__(_)__ _ _ _
    %  | (__/ _ \ '  \| '_ \ || |  _/ -_) | || / _` / _/ _ \ '_ \ / _` | ' \
    %   \___\___/_|_|_| .__/\_,_|\__\___|  \__/\__,_\__\___/_.__/_\__,_|_||_|
    %                 |_|
    % 5) Compute Jacobian of the reprojection error with respect to the pose
    % parameters and apply IRLS to iteratively update the camera pose for the subsequent frame (image i+1)

    rotationVector = rotationMatrixToVector(R);
    
    % Jacobian of the reprojection error with respect to the pose
    %     J = Jacobian_function(backProjectedPoints_3Dcoord, image_points.',...
    %       camera_params, cam_in_world_locations(:,:,i), rotationVector);
    %====================
    %   ___ ___ _    ___
    %  |_ _| _ \ |  / __|
    %   | ||   / |__\__ \
    %  |___|_|_\____|___/
    
    theta = [rotationVector, t].';
    
    iter = 0;
    lambda = 0.000001;
    u = threshold_irls + 1;
    
    while iter < max_iter && u > threshold_irls
        [e, sigma] = get_residuals(reprojected_points, image_points);
        energy = compute_energy(backProjectedPoints_3Dcoord, image_points.', camera_params, theta);
        W = compute_W(e ./ sigma);
        
        % Finite differences
        J = Jacob_fd(camera_params, rotationVector, t, backProjectedPoints_3Dcoord);
        % Symbolic Jacobian
%         J = Jacobian_function(backProjectedPoints_3Dcoord, image_points.',...
%             camera_params, t, rotationVector);
        
        delta = -inv(J' * W * J + lambda * eye(6)) * (J' * W * e);
        
        theta_updated = theta + delta;
        energy_new = compute_energy(backProjectedPoints_3Dcoord, image_points.', camera_params, theta_updated);
        
        if energy_new > energy
            lambda = 10 * lambda;
        else
            lambda = lambda / 10;
            theta = theta_updated;
        end
        
        u = norm(delta);
        iter = iter + 1;
    end
    R_ = rotationVectorToMatrix([theta(1); theta(2); theta(3)]);
    t_ = [theta(4); theta(5); theta(6)];
    
    %
    %%%%%%%%%%%%%
    % 6) Now the subsequent frame (image i+1) becomes the initial frame for the
    % next subsequent frame (image i+2) and the method continues until camera poses for all
    % images are estimated
    
    cam_in_world_locations(:,:,i+1) = t_;
    cam_in_world_orientations(:,:,i+1) = R_;
end
save('done_optim.mat');
%% Plot camera trajectory in 3D world CS + cameras

figure()
% Predicted trajectory
visualise_trajectory(vertices, edges, cam_in_world_orientations, cam_in_world_locations, 'Color', 'b');
hold on;
% Ground Truth trajectory
visualise_trajectory(vertices, edges, gt_valid.orientations, gt_valid.locations, 'Color', 'g');
hold off;
title('\color{green}Ground Truth trajectory \color{blue}Predicted trajectory')

%% Visualize bounding boxes

figure()
for i=1:num_files
    
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    % Ground Truth Bounding Boxes
    points_gt = project3d2image(vertices',camera_params, gt_valid.orientations(:,:,i), gt_valid.locations(:, :, i));
    % Predicted Bounding Boxes
    points_pred = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points_gt(1, edges(:, j)), points_gt(2, edges(:,j)), 'color', 'g');
        plot(points_pred(1, edges(:, j)), points_pred(2, edges(:,j)), 'color', 'b');
    end
    hold off;
    
    filename = fullfile(test_results_dir, strcat('image', num2str(i), '.png'));
    saveas(gcf, filename)
end

%% =========================================================================================
% $$$$$$$\                                                $$$$$$$\                     $$\     
% $$  __$$\                                               $$  __$$\                    $$ |    
% $$ |  $$ | $$$$$$\  $$$$$$$\  $$\   $$\  $$$$$$$\       $$ |  $$ |$$$$$$\   $$$$$$\$$$$$$\   
% $$$$$$$\ |$$  __$$\ $$  __$$\ $$ |  $$ |$$  _____|      $$$$$$$  |\____$$\ $$  __$$\_$$  _|  
% $$  __$$\ $$ /  $$ |$$ |  $$ |$$ |  $$ |\$$$$$$\        $$  ____/ $$$$$$$ |$$ |  \__|$$ |    
% $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ | \____$$\       $$ |     $$  __$$ |$$ |      $$ |$$\ 
% $$$$$$$  |\$$$$$$  |$$ |  $$ |\$$$$$$  |$$$$$$$  |      $$ |     \$$$$$$$ |$$ |      \$$$$  |
% \_______/  \______/ \__|  \__| \______/ \_______/       \__|      \_______|\__|       \____/ 


    %========================================================================================
    %     ___                __  __                     __                 __    _           
    %    /   |  ____  ____  / /_/ /_  ___  _____       / /___ __________  / /_  (_)___ _____ 
    %   / /| | / __ \/ __ \/ __/ __ \/ _ \/ ___/  __  / / __ `/ ___/ __ \/ __ \/ / __ `/ __ \
    %  / ___ |/ / / / /_/ / /_/ / / /  __/ /     / /_/ / /_/ / /__/ /_/ / /_/ / / /_/ / / / /
    % /_/  |_/_/ /_/\____/\__/_/ /_/\___/_/      \____/\__,_/\___/\____/_.___/_/\__,_/_/ /_/ 
                                                                                       
% Finite differences
J_fd = Jacob_fd(camera_params, rotationVector, t, backProjectedPoints_3Dcoord);
% Symbolic Jacobian
J_Sym = Jacobian_function2(backProjectedPoints_3Dcoord, image_points.',...
             camera_params, t, rotationVector);
error = sum(abs(J_fd)-abs(J_Sym))/size(J_fd,1);
disp(error)
%%
% Save estimated camera poses for the validation sequence using Vision TUM trajectory file
% format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Then estimate Absolute Trajectory Error (ATE) and Relative Pose Error for
% the validation sequence using python tools from: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
% In this task you should implement you own function to convert rotation matrix to quaternion

% Save estimated camera poses for the test sequence using Vision TUM
% trajectory file format

% Attach the file with estimated camera poses for the test sequence to your code submission
% If your code and results are good you will get a bonus for this exercise
% We are expecting the mean absolute translational error (from ATE) to be
% approximately less than 1cm

% TODO: Estimate ATE and RPE for validation and test sequences



%%
%=================================================================
% $$\   $$\           $$\
% $$ |  $$ |          $$ |
% $$ |  $$ | $$$$$$\  $$ | $$$$$$\   $$$$$$\   $$$$$$\  $$$$$$$\
% $$$$$$$$ |$$  __$$\ $$ |$$  __$$\ $$  __$$\ $$  __$$\$$  _____|
% $$  __$$ |$$$$$$$$ |$$ |$$ /  $$ |$$$$$$$$ |$$ |  \__\$$$$$$\
% $$ |  $$ |$$   ____|$$ |$$ |  $$ |$$   ____|$$ |      \____$$\
% $$ |  $$ |\$$$$$$$\ $$ |$$$$$$$  |\$$$$$$$\ $$ |     $$$$$$$  |
% \__|  \__| \_______|\__|$$  ____/  \_______|\__|     \_______/
%                         $$ |
%                         $$ |
%                         \__|


function [e, sigma] = get_residuals(first_2D, second_2D)
%GET_RESIDUALS Summary of this function goes here
%   Detailed explanation goes here
e = abs(first_2D - second_2D);
e = e(:);
sigma = 1.48257968 * mad(e, 1,'all');
end

function rho = Tukey(e)
%TUKEY Summary of this function goes here
%   Detailed explanation goes here
c = 4.685; % Tukey constant

rho = e;
rho(e>c) = c^2/6;
rho(e<=c) = (c^2/6) * (1 - (1 - rho(e<=c).^2/c^2).^3);
end

function W = compute_W(e)
% Calc weighting matrix
%   Detailed explanation goes here
c = 4.685; % Tukey constant
w = e;

w(e>c) = 0;
w(e<c) = (1 - (w(e<c)/c).^2).^2;
W = diag(w);
end

function energy = compute_energy(M_3D, m_2D, camera_params, theta)
%COMPUTE_ENERGY Summary of this function goes here
%   Detailed explanation goes here

[R, t] = get_pose(theta);

reprojected_points = project3d2image(M_3D',camera_params, R, t');

[e, sigma] = get_residuals(reprojected_points, m_2D');

% calc e_init

energy = sum(Tukey(e ./ sigma));

% Should it be like that or without the sigma
%energy = sum(Tukey(e));
end

function [R, t] = get_pose(theta)
    R = rotationVectorToMatrix([theta(1); theta(2); theta(3)]);
    t = [theta(4); theta(5); theta(6)];
end

function skewMatrix = get_sym_matrix(v)

skewMatrix = [0 -v(3) v(2) ; ...
    v(3) 0 -v(1) ; ...
    -v(2) v(1) 0];

end

function J = JacobianChainRule_function(intrinsicsMatrix, rotationVector, m_2D, M_3D)

rotationMatrix = rotationVectorToMatrix(rotationVector);
v = get_sym_matrix(rotationVector);
I = eye(3);

for j = 1:3
    e = I(:,j);
    %Compute derivative of R with respect to each one of the three rotating
    %variables (v1 v2 v3)
    dRdv{j} = ((rotationVector(j)*v + get_sym_matrix(cross(rotationVector, (I-rotationMatrix)*e)))/(norm(rotationVector)^2))*rotationMatrix;
end

dmtildedM = intrinsicsMatrix;

for M = 1: numel(M_3D(1,:))
    % Derivative of m with respect to mtilda.
    dmdmtilde = [1 0 -m_2D(1,M) ; 0 1 -m_2D(2,M)];
    
    % Derivative of Mi with respect to p. p=[R1 R2 R3 t1 t2 t3]
    dMdp = [dRdv{1}*M_3D(:, M) dRdv{2}*M_3D(:, M) dRdv{3}*M_3D(:, M) eye(3)];
    
    % Jacobian by chain rule
    J(2*M-1:2*M,:) = dmdmtilde * dmtildedM * dMdp;
end

end


