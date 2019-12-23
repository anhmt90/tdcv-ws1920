clear
clc
close all
addpath('helper_functions')
% addpath('own_helper_functions')
addpath('nuclear_bomb')
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
% You will need vl_sift() and vl_ubcmatch() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path

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
% You can use this function to label corners of the model on all images
% This function will give an array with image coordinates for all points
% Be careful that some points may not be visible on the image and so this
% will result in NaN values in the output array
% Don't forget to filter NaNs later
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

% Now you need to implement the method of iteratively reweighted least squares (IRLS)
% to optimise reprojection error between consecutive image frames

% Method steps:
% 1) Back-project SIFT keypoints from the initial frame (image i) to the object using the
% initial camera pose and the 3D ray intersection code from the task 1.
% This will give you 3D coordinates (in the world coordinate system) of the
% SIFT keypoints from the initial frame (image i) that correspond to the object

% 2) Find matches between descriptors of back-projected SIFT keypoints from the initial frame (image i) and the
% SIFT keypoints from the subsequent frame (image i+1) using vl_ubcmatch() from VLFeat library
% 3) Project back-projected SIFT keypoints onto the subsequent frame (image i+1) using 3D coordinates from the
% step 1 and the initial camera pose
% 4) Compute the reprojection error between 2D points of SIFT
% matches for the subsequent frame (image i+1) and 2D points of projected matches
% from step 3
% 5) Implement IRLS: for each IRLS iteration compute Jacobian of the reprojection error with respect to the pose
% parameters and update the camera pose for the subsequent frame (image i+1)
% 6) Now the subsequent frame (image i+1) becomes the initial frame for the
% next subsequent frame (image i+2) and the method continues until camera poses for all
% images are estimated

% We suggest you to validate the correctness of the Jacobian implementation
% either using Symbolic toolbox or finite differences approach

% TODO: Implement IRLS method for the reprojection error optimisation
% You can start with these parameters to debug your solution
% but you should also experiment with their different values
threshold_irls = 0.001; % update threshold for IRLS
max_iter = 500; % number of iterations
threshold_ubcmatch = 20; % matching threshold for vl_ubcmatch()


for i=1:(num_files-1)
    fprintf('Optimizing energy function between image %d and image %d\n', i, i+1)
    %   ___          _                 _        _   _
    %  | _ ) __ _ __| |___ __ _ _ ___ (_)___ __| |_(_)___ _ _
    %  | _ \/ _` / _| / / '_ \ '_/ _ \| / -_) _|  _| / _ \ ' \
    %  |___/\__,_\__|_\_\ .__/_| \___// \___\__|\__|_\___/_||_|
    %                   |_|         |__/
    %1) Project all SIFT keypoints from previous image,that have been matched, to 3D object. Use ray
    %intersection.
    
    cam_orient = cam_in_world_orientations(:,:,i);
    cam_loc = cam_in_world_locations(:,:,i);
    K = camera_params.IntrinsicMatrix;
    
    backProjected_point.coord3d = [];
    backProjected_point.descriptors = [];
    keypoint_3Dmodel = [];
    %indx_keypoints = [];
    
    P = K.' * [cam_orient -cam_orient*cam_loc.'];
%     P = [R; t]*camera_params.IntrinsicMatrix;
%     P = P';
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

                backProjected_point.coord3d = [backProjected_point.coord3d; xcoor];
                backProjected_point.descriptors = [backProjected_point.descriptors, descriptors{i}(:, siftpoint_)];
                %disp(xcoor);
            end
        end
        
    end
    
    % Calculate the matches
    sift_matches = vl_ubcmatch(backProjected_point.descriptors, descriptors{i+1}, threshold_ubcmatch);
    
    
    M_3D = backProjected_point.coord3d(sift_matches(1,:),:);
    
    % Project match points from 3D model to the current frame (i+1) (2D image)
    [R, t] = cameraPoseToExtrinsics(cam_orient, cam_loc); 
    reprojected_points = worldToImage(camera_params, R, t, M_3D);
    reprojected_points = reprojected_points.';
    image_points = [keypoints{i+1}(1,sift_matches(2,:));keypoints{i+1}(2,sift_matches(2,:))];
    
    
    
    %====================
    %   ___ ___ _    ___
    %  |_ _| _ \ |  / __|
    %   | ||   / |__\__ \
    %  |___|_|_\____|___/

    iter = 0;
    lambda = 0.001;
    u = threshold_irls + 1;
    
    while iter < max_iter && u > threshold_irls
        disp(iter)
        r = rotationMatrixToVector(R);
        
        [e, sigma] = get_residuals(reprojected_points, image_points);
        energy = compute_energy(M_3D, image_points.', camera_params, R, t);
        W = compute_W(e ./ sigma);
        
        % the Jacobian
        J = Jacobian_function(backProjectedPoints_3Dcoord, image_points.',...
            camera_params, t, r);

%         J = Jacobian_analytic(M_3D, K, R, t);

%         J = Jacobian_FD(camera_params, r, t, M_3D);
        
        delta = -inv(J' * W * J + lambda * eye(6)) * (J' * W * e);

        theta = [r'; t'];
        theta_updated = theta + delta;
        R_new = rotationVectorToMatrix(theta_updated(1:3));
        t_new = theta_updated(4:6).';
        
        energy_new = compute_energy(M_3D, image_points.', camera_params, R_new, t_new);
        
        if energy_new > energy
            lambda = 10 * lambda;
        else
            lambda = lambda / 10;
            R = R_new;
            t = t_new;
        end
        
        u = norm(delta);
        iter = iter + 1;
    end
    
    [cam_in_world_orientations(:, :, i+1), cam_in_world_locations(:, :, i+1)] = ...
        extrinsicsToCameraPose(R, t);
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
                                                                                       

%Finite Diferences
%First we parametrize the rotation using exponential maps
function J_fd = Jacobian_FD(camera_params, rotationVector, translationVector, points3d)
    %Initializing the Jacobian matrix
    J_fd = zeros(size(points3d,1)*2, 6);
    delta = 1e-9;

    for j=1:3
        %We are going to calculate the Jacobian by central differences,
        %which has the form of dF/dx = (F(x+delta)-F(x-delta))/2delta
        diffVect = zeros(1,3);
        diffVect(1,j) = 1;
        diffVect = [rotationVector + delta * diffVect;rotationVector - delta * diffVect];
        %Calculating partial derivatives wrt rotation params as:
        reprojectedDiff_Pos = worldToImage(camera_params,rotationVectorToMatrix(diffVect(1,:)), translationVector, points3d);
        reprojectedDiff_Neg = worldToImage(camera_params,rotationVectorToMatrix(diffVect(2,:)), translationVector, points3d);
        %Partial derivative of the reprojected error (ReprojectedPoints -
        %Matched Points). But since Matched Points do not depend on pose
        %params, we can calculate it only as the difference between the
        %reprojected differences
        partialDev = (reprojectedDiff_Pos - reprojectedDiff_Neg);
        partialDev = partialDev/(2*delta);
        J_fd(1:2:end,j) = partialDev(:,1);
        J_fd(2:2:end,j) = partialDev(:,2);

        %Here, We do the same wrt translation parameters
        diffTrans = zeros(1,3);
        diffTrans(1,j) = 1;
        diffTrans = [translationVector + delta * diffTrans; translationVector - delta * diffTrans];
        % Project match points from 3D model to the current frame (i+1) (2D image)
        reprojectedDiff_Pos = worldToImage(camera_params, rotationVectorToMatrix(rotationVector), diffTrans(1,:), points3d);
        reprojectedDiff_Neg = worldToImage(camera_params, rotationVectorToMatrix(rotationVector), diffTrans(2,:), points3d);
        partialDev = (reprojectedDiff_Pos - reprojectedDiff_Neg);
        partialDev = partialDev/(2*delta);
        J_fd(1:2:end,j+3) = partialDev(:,1);
        J_fd(2:2:end,j+3) = partialDev(:,2);
    end
end
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
% sigma = 1.48257968 * mad(e, 1,'all');
sigma = 1.48257968 * median(abs(e));
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
w(e<c) = (1 - w(e<c).^2/c^2).^2;
W = diag(w);
end

function energy = compute_energy(M_3D, m_2D, camera_params, R, t)
%COMPUTE_ENERGY Summary of this function goes here
%   Detailed explanation goes here

% reprojected_points = project3d2image(M_3D',camera_params, R, t);
reprojected_points = worldToImage(camera_params, R, t, M_3D);

[e, sigma] = get_residuals(reprojected_points', m_2D');

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


