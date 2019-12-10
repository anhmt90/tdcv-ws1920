clear
clc
close all
addpath('helper_functions')
% addpath('own_helper_functions')

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

% Initialization: Compute camera pose for the first image
%              _      _    _
%   _ __  __ _| |_ __| |_ (_)_ _  __ _
%  | '  \/ _` |  _/ _| ' \| | ' \/ _` |
%  |_|_|_\__,_|\__\__|_||_|_|_||_\__, |
%                                |___/
%   _    _     _                                  _     _
%  / |__| |_  (_)_ __  __ _   ___   _ __  ___  __| |___| |
%  | (_-<  _| | | '  \/ _` | |___| | '  \/ _ \/ _` / -_) |
%  |_/__/\__| |_|_|_|_\__, |       |_|_|_\___/\__,_\___|_|
%                     |___/
% As the initialization step for tracking
% you need to compute the camera pose for the first image
% The first image and it's camera pose will be your initial frame
% and initial camera pose for the tracking process

% You can use estimateWorldCameraPose() function or your own implementation
% of the PnP+RANSAC from previous tasks

% You can get correspondences for PnP+RANSAC either using your SIFT model from the previous tasks
% or by manually annotating corners (e.g. with mark_images() function)


% TODO: Estimate camera position for the first image

load('sift_model.mat', 'model');

threshold_ubcmatch = 3.5;
sift_matches = cell(1,1);
sift_matches{1} = vl_ubcmatch(descriptors{1}, model.descriptors, threshold_ubcmatch);

figure
% Plot the image and the matched points on top.
imshow (char(Filenames(1)));
title('Matched points between model and 1st frame')
hold on;
% Visualize the matched points of the image
vl_plotframe(keypoints{1}(:,sift_matches{1}(1,:)), 'linewidth',2);
%plot (keypoints{h}(1, sift_matches{h}(1,:)), keypoints{h}(2, sift_matches{h}(1,:)), 'r*');
hold off;

%=======================================================
%            _     _      _ _
%   __ _ ___| |_  (_)_ _ (_) |_   _ __  ___ ______
%  / _` / -_)  _| | | ' \| |  _| | '_ \/ _ (_-< -_)
%  \__, \___|\__| |_|_||_|_|\__| | .__/\___/__|___|
%  |___/                         |_|
%   ___      ___     _     ___    _   _  _ ___   _   ___
%  | _ \_ _ | _ \  _| |_  | _ \  /_\ | \| / __| /_\ / __|
%  |  _/ ' \|  _/ |_   _| |   / / _ \| .` \__ \/ _ \ (__
%  |_| |_||_|_|     |_|   |_|_\/_/ \_\_|\_|___/_/ \_\___|

% cam_in_world_orientations = zeros(3,3);
% cam_in_world_locations = zeros(1,3);
best_inliers_set = cell(1);

init_world_orientations = zeros(3,3);
init_world_locations = zeros(1,3);


ransac_iterations = 1000; %input('Please select the number of iterations:','s');
threshold_ransac = 10; %input('Please select the threshold for RANSAC method:','s');

[best_inliers_set, max_num_inliers] = ransac_function(ransac_iterations, threshold_ransac, sift_matches{1}, keypoints{1}, model.coord3d, camera_params);

% Take the indexes of the inliers
inliers_2Dimage = sift_matches{1}(1,best_inliers_set);
inliers_3Dmodel = sift_matches{1}(2,best_inliers_set);


% Get the 2D and 3D coordinates for the inliers
image_points = [keypoints{1}(1,inliers_2Dimage); keypoints{1}(2,inliers_2Dimage)]';
world_points = model.coord3d(inliers_3Dmodel,:);

[init_orientation, init_location] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', 10000);

cam_in_world_orientations(:,:, 1) = init_orientation;
cam_in_world_locations(:,:, 1) = init_location;

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
title(sprintf('Initial Image Camera Pose'));
%   Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
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
threshold_irls = 0.005; % update threshold for IRLS
max_iter = 20; % number of iterations
threshold_ubcmatch = 6; % matching threshold for vl_ubcmatch()


for i=1:(num_files-1)
    fprintf('Optimizing energy function between image %d and image %d\n', i, i+1)
    %   ___          _                 _        _   _
    %  | _ ) __ _ __| |___ __ _ _ ___ (_)___ __| |_(_)___ _ _
    %  | _ \/ _` / _| / / '_ \ '_/ _ \| / -_) _|  _| / _ \ ' \
    %  |___/\__,_\__|_\_\ .__/_| \___// \___\__|\__|_\___/_||_|
    %                   |_|         |__/
    %1) Project all SIFT keypoints from previous image,that have been matched, to 3D object. Use ray
    %intersection.
    
    backProjected_point.coord3d = [];
    backProjected_point.descriptors = [];
    keypoint_3Dmodel = [];
    %indx_keypoints = [];
    
    P = camera_params.IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q; % this corresponds to C
    
    %===========================================
    %      _             _
    %   __| |_  ___  ___| |_   _ _ __ _ _  _ ___
    %  (_-< ' \/ _ \/ _ \  _| | '_/ _` | || (_-<
    %  /__/_||_\___/\___/\__| |_| \__,_|\_, /__/
    %                                   |__/
    % We need to back project all sift points found on image i (the 1st image
    % in this case) to find which one intersects with the 3D model
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
    
    
    
    % Images to check if the points that are said to belong to the model
    % are inside the box in each image
    figure(1)
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit')
    title('SIFT points intersecting with the 3D model when backprojected')
    hold on;
    scatter(keypoint_3Dmodel(1,:),keypoint_3Dmodel(2,:),'bo')
    hold off;
    
    %%%%%%%%%%%%%
    % 2) Find matches between SIFT keypoints from the initial frame (image 1)
    % that have intersected with the 3D model and the keypoints of the subsequent frame (image 2)
    
    threshold_ubcmatch = 2.5;
    
    % Calculate the matches
    sift_matches{i+1} = vl_ubcmatch(backProjected_point.descriptors, descriptors{i+1}, threshold_ubcmatch);
    
    % sift_matches{2}(1,:) --> sift matches from the previous image, out of the ones that were
    % backprojected to the model
    % sift_matches{2}(2,:) ---> sift matches in the subsequent frame
    
    
    
    figure(2)
    % Plot the image and the matched points on top.
    imshow (char(Filenames(i+1)));
    title('Matched SIFT points from current frame (i+1) on top of the current image')
    hold on;
    % Visualize the matched points on the frame we are currently trying to
    % estimate the pose i+1
    vl_plotframe(keypoints{i+1}(:,sift_matches{i+1}(2,:)), 'linewidth',2)
    %plot (keypoints{h}(1, sift_matches{h}(1,:)), keypoints{h}(2, sift_matches{h}(1,:)), 'r*');
    hold off;
    
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
    reprojected_points = project3d2image(backProjectedPoints_3Dcoord',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    
    %========================================================
    %   ___      _ _   _      _   ___
    %  |_ _|_ _ (_) |_(_)__ _| | | __|_ _ _ _ ___ _ _
    %   | || ' \| |  _| / _` | | | _|| '_| '_/ _ \ '_|
    %  |___|_||_|_|\__|_\__,_|_| |___|_| |_| \___/_|
    % 4) Compute the reprojection error between pixels from SIFT
    % matches for the subsequent frame (image i+1) and from reprojected matches
    % from step 3
    image_points = [keypoints{i+1}(1,sift_matches{i+1}(2,:));keypoints{i+1}(2,sift_matches{i+1}(2,:))];
    
    % Calculate the error of reprojected the points
    %     distance_reprojection = pdist2(reprojected_points',image_points');
    
    % We take the diagonal because pdist2 gets the distance between all
    % points to each other, and we only need each point to eachself.
    %     residuals = diag(distance_reprojection);
    %     energy = sum(residuals' * residuals);
    
    % Figure to see the comparison between the points that were matched
    % between the SIFT points of the previous and current frame, vs the
    % reprojected matched points from the 3D model.
    
    
    figure(3)
    imshow(char(Filenames(i+1)))
    title('Reprojected points from 3D model vs Matched points')
    hold on;
    scatter(reprojected_points(1,:)',reprojected_points(2,:)','rx')
    scatter(image_points(1,:)',image_points(2,:)','bo')
    legend('Reprojected points','2D correspondences with previous image')
    hold off;
    
    %=======================================================================
    %    ___                     _          _____           _    _
    %   / __|___ _ __  _ __ _  _| |_ ___   _ | |__ _ __ ___| |__(_)__ _ _ _
    %  | (__/ _ \ '  \| '_ \ || |  _/ -_) | || / _` / _/ _ \ '_ \ / _` | ' \
    %   \___\___/_|_|_| .__/\_,_|\__\___|  \__/\__,_\__\___/_.__/_\__,_|_||_|
    %                 |_|
    % 5) Compute Jacobian of the reprojection error with respect to the pose
    % parameters and apply IRLS to iteratively update the camera pose for the subsequent frame (image i+1)
    
    %%%%%%%%%%%%%
    %Symbolic Method
    % Rotation parameters (given in Exponential Maps)
    rotationVector = rotationMatrixToVector(cam_in_world_orientations(:,:,i));
    
    % Jacobian of the reprojection error with respect to the pose
    %     J = Jacobian_function(backProjectedPoints_3Dcoord, image_points.',...
    %       camera_params, cam_in_world_locations(:,:,i), rotationVector);
    %====================
    %   ___ ___ _    ___
    %  |_ _| _ \ |  / __|
    %   | ||   / |__\__ \
    %  |___|_|_\____|___/
    
    translationVector = cam_in_world_locations(:,:,i);
    theta = [rotationVector, translationVector].';
    
    iter = 0;
    lambda = 0.001;
    u = threshold_irls + 1;
    
    while iter < max_iter && u > threshold_irls
        [e, sigma] = get_residuals(reprojected_points, image_points);
        energy = compute_energy(backProjectedPoints_3Dcoord, image_points.', camera_params, theta);
        W = compute_W(e ./ sigma);
        
        % the Jacobian
        J = Jacobian_function(backProjectedPoints_3Dcoord, image_points.',...
            camera_params, translationVector, rotationVector);
        
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
    R = rotationVectorToMatrix([theta(1); theta(2); theta(3)]);
    t = [theta(4); theta(5); theta(6)];
    
    %
    %%%%%%%%%%%%%
    % 6) Now the subsequent frame (image i+1) becomes the initial frame for the
    % next subsequent frame (image i+2) and the method continues until camera poses for all
    % images are estimated
    
    cam_in_world_locations(:,:,i+1) = t;
    cam_in_world_orientations(:,:,i+1) = R;
end
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
rotationVector = rotationMatrixToVector(cam_in_world_orientations(:,:,i));

%Initializing the Jacobian matrix
J_fd = zeros(size(reprojected_points,2)*2,6);
delta = 1e-9;

for j=1:3
    %We are going to calculate the Jacobian by central differences,
    %which has the form of dF/dx = (F(x+delta)-F(x-delta))/2delta
    diffVect = zeros(1,3);
    diffVect(1,j) = 1;
    diffVect = [rotationVector + delta * diffVect;rotationVector - delta * diffVect];
    %Calculating partial derivatives wrt rotation params as:
    reprojectedDiff_Pos = worldToImage(camera_params,rotationVectorToMatrix(diffVect(1,:)), cam_in_world_locations(:, :, i), backProjectedPoints_3Dcoord);
    reprojectedDiff_Neg = worldToImage(camera_params,rotationVectorToMatrix(diffVect(2,:)), cam_in_world_locations(:, :, i), backProjectedPoints_3Dcoord);
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
    diffTrans = [cam_in_world_locations(:, :, i) + delta * diffTrans;cam_in_world_locations(:, :, i) - delta * diffTrans];
    % Project match points from 3D model to the current frame (i+1) (2D image)
    reprojectedDiff_Pos = worldToImage(camera_params,cam_in_world_orientations(:,:,i), diffTrans(1,:), backProjectedPoints_3Dcoord);
    reprojectedDiff_Neg = worldToImage(camera_params,cam_in_world_orientations(:,:,i), diffTrans(2,:), backProjectedPoints_3Dcoord);
    partialDev = (reprojectedDiff_Pos - reprojectedDiff_Neg);
    partialDev = partialDev/(2*delta);
    J_fd(1:2:end,j+3) = partialDev(:,1);
    J_fd(2:2:end,j+3) = partialDev(:,2);
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
w(e<c) = (1 - w(e<c).^2/c^2).^2;
W = diag(w);
end



function energy = compute_energy(M_3D, m_2D, camera_params, theta)
%COMPUTE_ENERGY Summary of this function goes here
%   Detailed explanation goes here

R = rotationVectorToMatrix([theta(1); theta(2); theta(3)]);
t = [theta(4); theta(5); theta(6)];

reprojected_points = project3d2image(M_3D',camera_params, R, t');

[e, sigma] = get_residuals(reprojected_points, m_2D');

% calc e_init
energy = sum(Tukey(e));
end



