clear
clc
close all
addpath('helper_functions')

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
fx = 2960.37845;
fy = fx;
cx = 1841.68855;
cy = 1235.23369;

intrinsicsMatrix = [fx 0 0;0 fy 0;cx cy 1];
camera_params = cameraParameters('IntrinsicMatrix',intrinsicsMatrix);

% Get all filenames in images folder
FolderInfo = dir(fullfile(valid_img_dir, '*.JPG'));
Filenames = fullfile(valid_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%% Detect SIFT keypoints in all images

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

max_reproj_err = 10;

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
threshold_irls = 0.005; % update threshold for IRLS
N = 100; % number of iterations
threshold_ubcmatch = 4; % matching threshold for vl_ubcmatch()
coord = cell(num_files,1);

for i = 1:num_files-1
    fprintf('Running iteration: %d \n', i);
     % Step 1
    P = camera_params.IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) ...
        -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
    
    %     Randomly select a number of SIFT keypoints
    perm = randperm(size(keypoints{i},2));
    sel = perm(1:30000);
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q;
    descriptors_new = [];
    
    for j = 1:30000
        m = [keypoints{i}(1:2,sel(j)); 1];
        lambda = norm(inv(Q)*m);
        r = orig + lambda*(inv(Q)*m);

        [~, t, u, v, coords] = TriangleRayIntersection(orig', (r-orig)', ...
            vertices(faces(:,1)+1,:), vertices(faces(:,2)+1,:), vertices(faces(:,3)+1,:));
        outliers = find(isnan(coords(:,1)));
        coords(outliers,:)=[];

        if ~isempty(coords)
            t(outliers,:)=[];
            [min_t, index_min] = min(t);
            coords = coords(index_min,:);
            coord{i} = [coord{i}; coords];
            descriptors_new = [descriptors_new, descriptors{i}(:,sel(j))];
        end
    end
     % Step 2
    sift_matches = vl_ubcmatch(descriptors_new, descriptors{i+1}, threshold_ubcmatch);
    
     % Step 3
    world_points = coord{i}(sift_matches(1,:),:);
    image_points = keypoints{i+1}(1:2, sift_matches(2,:));
    
    [init_orientations,init_locations,inlieridx,~] = ...
        estimateWorldCameraPose(image_points', world_points, ...
        camera_params, 'MaxReprojectionError', 20);

     % Step 4
     fprintf('EARL %d \n', i);
     [cam_in_world_orientations(:,:,i+1), cam_in_world_locations(:,:,i+1)] = ...
         earl(init_orientations, init_locations, camera_params, world_points, image_points, threshold_irls, N);
    
end



%% Plot camera trajectory in 3D world CS + cameras

%load('good_rotations.mat')
%load('good_translations.mat')
% load('good_rotations2.mat')
% load('good_translations2.mat')
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

%% Bonus part

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
fileID = fopen('trajectory.txt', 'w');

for quack = 1:num_files
    quat = queeny(cam_in_world_orientations(:,:,quack));
    timestamp = 86400*(datenum(now) - datenum('01-Jan-1970 00:00:00') - 1/24);
    fprintf(fileID, '%f %f %f %f %f %f %f %f\r\n', timestamp, cam_in_world_locations(:,1,quack), cam_in_world_locations(:,2,quack), cam_in_world_locations(:,3,quack),...
        quat(1), quat(2), quat(3), quat(4));
    
end

fclose(fileID);