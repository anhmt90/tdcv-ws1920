clear
clc
close all
addpath('helper_functions')

%% Setup
% % path to the images folder
% path_img_dir = 'data/tracking/validation/img';
% % path to object ply file
% object_path = 'data/teabox.ply';
% % path to results folder
% results_path = 'data/tracking/validation/results';

% There are two folders inside the data, I think we have to start with the
% testing. 
path_img_dir = 'data/tracking/test/img';
% path to object ply file
object_path = 'data/teabox.ply';
% % path to results folder
results_path = 'data/tracking/test/results';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);
faces = faces + 1;


% Create directory for results
if ~exist(results_path,'dir') 
    mkdir(results_path); 
end

% Load Ground Truth camera poses for the validation sequence
% Camera orientations and locations in the world coordinate system
load('gt_valid.mat')

% TODO: setup camera parameters (camera_params) using cameraParameters()
focalLength = 2960.37845;
principalPoint_cx = 1841.68855;
principalPoint_cy = 1235.23369;
imageSize = [2456 3680];
camera_params = cameraParameters('IntrinsicMatrix',[focalLength 0 0; 0 focalLength 0; principalPoint_cx principalPoint_cy 1],'ImageSize',imageSize);


%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%% Detect SIFT keypoints in all images

% You will need vl_sift() and vl_ubcmatch() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path
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
%% Initialization: Compute camera pose for the first image

% As the initialization step for the tracking
% we need to compute the camera pose for the first image 
% The first image and it's camera pose will be our initial frame and initial camera pose for the tracking process

% You can use estimateWorldCameraPose() function or your own implementation
% of the PnP+RANSAC from the previous tasks

% You can get correspondences for PnP+RANSAC either using your SIFT model from the previous tasks
% or by manually annotating corners (e.g. with mark_images() function)


% TODO: Estimate camera position for the first image
load('sift_model.mat')
sift_matches=cell(1,1);
sift_scores=cell(1,1);

% Default threshold for SIFT keypoints matching: 1.5
% % When taking higher value, match is only recognized if similarity is very high
threshold_ubcmatch = 2.5;
% Match features between SIFT model and SIFT features from new image
[sift_matches{1} , sift_scores{1}]= vl_ubcmatch(descriptors{1}, model.descriptors, threshold_ubcmatch);

figure
% Plot the image and the matched points on top.
imshow (char(Filenames(1)));
title('Matched points between model and 1st frame')
hold on;
% Visualize the matched points of the image
vl_plotframe(keypoints{1}(:,sift_matches{1}(1,:)), 'linewidth',2);
%plot (keypoints{h}(1, sift_matches{h}(1,:)), keypoints{h}(2, sift_matches{h}(1,:)), 'r*');
hold off;
%%
cam_in_world_orientations = zeros(3,3);
cam_in_world_locations = zeros(1,3);
best_inliers_set = cell(1);

init_world_locations = zeros(1,3);
init_world_orientations = zeros(3,3);

ransac_iterations = 300; %input('Please select the number of iterations:','s');
threshold_ransac = 10; %input('Please select the threshold for RANSAC method:','s');
%     TODO: Implement the RANSAC algorithm here

[best_inliers_set, max_num_inliers] = ransac_function(ransac_iterations, threshold_ransac, sift_matches{1}, keypoints{1}, model.coord3d, camera_params);

% Take the indexes of the inliers
inliers_2Dimage = sift_matches{1}(1,best_inliers_set);
inliers_3Dmodel = sift_matches{1}(2,best_inliers_set);


% Get the 2D and 3D coordinates for the inliers
image_points = [keypoints{1}(1,inliers_2Dimage); keypoints{1}(2,inliers_2Dimage)]';
world_points = model.coord3d(inliers_3Dmodel,:);

[init_orientation, init_location] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', 4);

cam_in_world_orientations(:,:, 1) = init_orientation;
cam_in_world_locations(:,:, 1) = init_location;

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
title(sprintf('Initial Image Camera Pose'));
hold on
%   Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
for j=1:12
    plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
end
hold off;

%% IRLS nonlinear optimisation

% Now you need to implement the method of iteratively reweighted least squares (IRLS)
% to optimise reprojection error between consecutive image frames

% Method steps:
% 1) Project SIFT keypoints from the initial frame (image i) to the object using the
% initial camera pose and the 3D ray intersection code from the task 1. 
% This will give you 3D coordinates (in the world coordinate system) of the
% SIFT keypoints from the initial frame (image i) that correspond to the object
% 2) Find matches between SIFT keypoints from the initial frame (image i) and the
% subsequent frame (image i+1) using vl_ubcmatch() from VLFeat library
% 3) Reproject these matches using corresponding 3D coordinates from the
% step 1 and the initial camera pose back to the subsequent frame (image i+1)
% 4) Compute the reprojection error between pixels from SIFT
% matches for the subsequent frame (image i+1) and from reprojected matches
% from step 3
% 5) Compute Jacobian of the reprojection error with respect to the pose
% parameters and apply IRLS to iteratively update the camera pose for the subsequent frame (image i+1)
% 6) Now the subsequent frame (image i+1) becomes the initial frame for the
% next subsequent frame (image i+2) and the method continues until camera poses for all
% images are estimated

% We suggest you to validate the correctness of the Jacobian implementation
% either using Symbolic toolbox or finite differences approach

% TODO: Implement IRLS method for the reprojection error optimisation

%%%%%%%%% 1st Attempt Implementation %%%%%

% Number of files -1 since we are going to use their scheme:
% i is the initial frame (frame number 1) for which we have already
% calculated the pose.
% i + 1 is the subsequente frame (frame number 2) for which we want to
% estimate the pose.


for i=1 %:(num_files-1)
    
    %%%%%%%%%%%%%
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
    
    % We need to back project all sift points found on image i (first image
    % in this case) to find which one intersect with the 3D model 
    for h= 1:size(keypoints{i},2)
        
        % Create homogeneous coordinate point for one sift point
        point = [keypoints{i}(1, h) keypoints{i}(2, h) 1];
        % calculate ray from camera center through random selected keypoint
        lambda = norm(inv(Q)*point.');
        ray_m = orig + lambda * inv(Q)*point.';
        
        % iterate through faces
        for k=1:size(faces)
            % Skip iteration for not visible faces
            %if(any(ismember(nan_index, faces(k, :))))
            %    continue;
            %end
            
            % calculate intersection ray/face (same as in hw1.1_task1)
            [intersect, t, u, v, xcoor] = TriangleRayIntersection(orig', (ray_m - orig)',...
                vertices(faces(k, 1), :), vertices(faces(k, 2), :), vertices(faces(k, 3), :), ...
                'planeType','one sided');
            
            % save 3d intersection coordinates and descriptors of keypoint
            if intersect ~= 0
                % keypoints that have intersected the 3D model
                keypoint_3Dmodel = [keypoint_3Dmodel, keypoints{i}(:,h)];
                % index of those keypoints with respect to the keypoints
                % matrix
                % indx_keypoints = [indx_keypoints; h];
                backProjected_point.coord3d = [backProjected_point.coord3d; xcoor];
                backProjected_point.descriptors = [backProjected_point.descriptors descriptors{i}(:, h)];
                %disp(xcoor);
            end
        end
        
    end
    
    % Images to check if the points that are said to belong to the model
    % are inside the box in each image
    figure(1)
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit')
    title('SIFT points that have intersected with the 3D model when backprojected')
    hold on;
    scatter(keypoint_3Dmodel(1,:),keypoint_3Dmodel(2,:),'bo')
    hold off;
   
    %%%%%%%%%%%%%
    % 2) Find matches between SIFT keypoints from the initial frame (image 1) 
    % that have intersected with the 3D model and the keypoints of the subsequent frame (image 2)
    
    threshold_ubcmatch = 2.5;
    descriptors_backProjected_points = backProjected_point.descriptors;
    
    % Calculate the matches
    [sift_matches{i+1} , sift_scores{i+1}]= vl_ubcmatch(descriptors_backProjected_points, descriptors{i+1}, threshold_ubcmatch);
    
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
    
    %%%%%%%%%%%%%
    % 3) Reproject these matches using corresponding 3D coordinates from the
    % step 1 and the initial camera pose back to the subsequent frame (image i+1)
    
    % Get the 3D coordinates of the backprojected points coming from the
    % initial frame
    backProjectedPoints_3Dcoord = backProjected_point.coord3d(sift_matches{i+1}(1,:),:);
    
    % Project match points from 3D model to the current frame (i+1) (2D image) 
    reprojected_points = project3d2image(backProjectedPoints_3Dcoord',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
       
    %%%%%%%%%%%%%
    % 4) Compute the reprojection error between pixels from SIFT
    % matches for the subsequent frame (image i+1) and from reprojected matches
    % from step 3
    image_points = [keypoints{i+1}(1,sift_matches{i+1}(2,:));keypoints{i+1}(2,sift_matches{i+1}(2,:))];
    
    % Calculate the error of reprojected the points
    distance_reprojection = pdist2(reprojected_points',image_points');
    
    % We take the diagonal because pdist2 gets the distance between all
    % points to each other, and we only need each point to eachself.
    d = diag(distance_reprojection);
    
    % Figure to see the comparison between the points that were matched
    % between the SIFT points of the previous and current frame, vs the
    % reprojected matched points from the 3D model.
    figure(3)
    imshow(char(Filenames(i+1)))
    title('Reprojected points from 3D model vs Matched points')
    hold on;
    scatter(reprojected_points(1,:)',reprojected_points(2,:)','rx')
    scatter(image_points(1,:)',image_points(2,:)','bo')
    legend('Reprojected points','Matched points from this image')
    hold off;
    %%
    %%%%%%%%%%%%%
    % 5) Compute Jacobian of the reprojection error with respect to the pose
    % parameters and apply IRLS to iteratively update the camera pose for the subsequent frame (image i+1)
    %%%%%%%%%%%%%
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
    
            
    %%%%%%%%%%%%%
    %Symbolic Method
    % Rotation parameters (given in Exponential Maps)
    rotationVector = rotationMatrixToVector(cam_in_world_orientations(:,:,i));
    
    % Jacobian of the reprojection error with respect to the pose
    J = Jacobian_function(backProjectedPoints_3Dcoord, image_points.',...
        camera_params, cam_in_world_locations(:,:,i), rotationVector);
    
    %%
    %%%%%%%%%%%%%
    % 6) Now the subsequent frame (image i+1) becomes the initial frame for the
    % next subsequent frame (image i+2) and the method continues until camera poses for all
    % images are estimated
    
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%



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
    
    filename = fullfile(results_path, strcat('image', num2str(i), '.png'));
    saveas(gcf, filename)
end

%% Bonus part

% Save estimated camera poses for the validation sequence using TUM Ground-truth trajectories file
% format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Then estimate Absolute Trajectory Error (ATE) and Relative Pose Error for
% the validation sequence using python tools from: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools

% Save estimated camera poses for the test sequence using TUM Ground-truth
% trajectories file format

% Send us this file with the estimated camera poses for the evaluation
% If the code and results are good you will get a bonus for this exercise
% We are expecting the mean absolute translational error (from ATE) to be
% approximately less than 1cm

% TODO: Estimate ATE and RPE for validation and test sequences

