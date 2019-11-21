clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/detection';
% path to object ply file
object_path = '../data/teabox.ply';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Load the SIFT model from the previous task
load('sift_model.mat');


% TODO: setup camera intrinsic parameters using cameraParameters()
focalLength = 2960.37845;
principalPoint_cx = 1841.68855;
principalPoint_cy = 1235.23369;
imageSize = [2456 3680];
camera_params = cameraParameters('IntrinsicMatrix',[focalLength 0 0; 0 focalLength 0; principalPoint_cx principalPoint_cy 1],'ImageSize',imageSize);


%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);


%% Match SIFT features of new images to the SIFT model with features computed in the task 1
% You should use VLFeat function vl_ubcmatch()

% Place SIFT keypoints and descriptors of new images here
keypoints=cell(num_files,1);
descriptors=cell(num_files,1);
% Place matches between new SIFT features and SIFT features from the SIFT
% model here
sift_matches=cell(num_files,1);
scores=cell(num_files,1);

% Default threshold for SIFT keypoints matching: 1.5 
% When taking higher value, match is only recognized if similarity is very high
threshold_ubcmatch = 2.5; 

for i=1:num_files
    fprintf('Calculating and matching sift features for image: %d \n', i)
    
    %     TODO: Prepare the image (img) for vl_sift() function
    I = imread(Filenames{i});
    img = single(rgb2gray(I));
    
    % Get sift keypoints and descriptors for each of the images inside the
    % detection folder
    [keypoints{i}, descriptors{i}] = vl_sift(img);
    
    % Match features between SIFT model and SIFT features from new image
    [sift_matches{i} , scores{i}]= vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch);
    
end


% Save sift features, descriptors and matches and load them when you rerun the code to save time%save('sift_matches.mat', 'sift_matches');
save('detection_keypoints.mat', 'keypoints')
save('detection_descriptors.mat', 'descriptors')
save('sift_matches.mat','sift_matches')

%load('sift_matches.mat')
%load('detection_keypoints.mat')
%load('detection_descriptors.mat')

%% TEST MARIA Plot the matched points.
% 
% Select the test image you want to visualize.

for h=1:num_files
    figure
    % Plot the image and the matched points on top.
    imshow (char(Filenames(h)));
    hold on;
    % Visualize the matched points of the image
    vl_plotframe(keypoints{h}(:,sift_matches{h}(1,:)), 'linewidth',2);
    %plot (keypoints{h}(1, sift_matches{h}(1,:)), keypoints{h}(2, sift_matches{h}(1,:)), 'r*');
    hold off;
end


%% PnP and RANSAC 
% Implement the RANSAC algorithm featuring also the following arguments:
% Reprojection error threshold for inlier selection - 'threshold_ransac'  
% Number of RANSAC iterations - 'ransac_iterations'

% Pseudocode
% i Randomly select a sample of 4 data points from S and estimate the pose using PnP.
% ii Determine the set of data points Si from all 2D-3D correspondences 
%   where the reprojection error (Euclidean distance) is below the threshold (threshold_ransac). 
%   The set Si is the consensus set of the sample and defines the inliers of S.
% iii If the number of inliers is greater than we have seen so far,
%   re-estimate the pose using Si and store it with the corresponding number of inliers.
% iv Repeat the above mentioned procedure for N iterations (ransac_iterations).

% For PnP you can use estimateWorldCameraPose() function
% but only use it with 4 points and set the 'MaxReprojectionError' to the
% value of 10000 so that all these 4 points are considered to be inliers

% Place camera orientations, locations and best inliers set for every image here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);
best_inliers_set = cell(num_files, 1);

% to avoid eliminating outlier correspondences through the integrated MSAC-estimation
max_reproj_err = 1000;

ransac_iterations = 1000; %input('Please select the number of iterations:','s');  
threshold_ransac = 4; %input('Please select the threshold for RANSAC method:','s');

for i = 1:num_files
    fprintf('Running PnP+RANSAC for image: %d \n', i)
   
%     TODO: Implement the RANSAC algorithm here
    for j = 1:ransac_iterations
        
        perm = randperm(size(sift_matches{i},2));
        % We want to take 4 random points each time
        sel = perm(1:4);
        
        % Get those 4 random points from the matches between the model and the 2D image. 
        matched_points_image2D = sift_matches{i}(1,sel);
        matched_points_model3D = sift_matches{i}(2,sel);

        % Get the 2D and 3D coordinates for the 4 randomly selected matches
        % and estimate the pose using PnP
        image_points = [keypoints{i}(1,matched_points_image2D); keypoints{i}(2,matched_points_image2D)]';
        world_points = model.coord3d(matched_points_model3D,:);
        
        [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i),inlierIdx,status] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', max_reproj_err);

        %%% Get the indexes of all matches %%%
        all_matchedPoints_2Dimage = sift_matches{i}(1,:);
        all_matchedPoints_3Dmodel = sift_matches{i}(2,:);
        
        % Get 2D coordinates of all matches
        all_image_points = [keypoints{i}(1,all_matchedPoints_2Dimage); keypoints{i}(2,all_matchedPoints_2Dimage)]';
        % Get 3D coordinates of all matches
        all_model_points = model.coord3d(all_matchedPoints_3Dmodel,:);
        
        % Project all of the 3D points from the matches into the 2D image
        % using the obtained orientation and translation transformations
        reprojected_points = project3d2image(all_model_points',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
        reprojected_points = reprojected_points';

        % Calculate the error of reprojecting the points, to determine the
        % number of inliers and the number of outliers. 
        % The error of reprojection is the euclidean distance between the
        % points in the image and the reprojected points. 
        distance_reprojection = pdist2(reprojected_points, all_image_points);
        
        % We take the diagonal because pdist2 gets the distance between all
        % points to each other, and we only need each point to eachself.
        d = diag(distance_reprojection);
         
        % If there is any point where the distance is smaller than the threshold selected
        if any(d < threshold_ransac)
            
           % Determine which matched points are considered inliers
           % (distance smaller than the threshold)
           [pos_inliers,col] = find(d < threshold_ransac);
           num_inliers = size(pos_inliers,1);
           
           % If it is the first iteration of one image, we set the maximum of inliers as
           % the first ones.
           if j == 1
               
               max_num_inliers = num_inliers;
               % save the position of the best inliers with respect to the sift_matches 
               best_inliers_set{i} = pos_inliers;
           
           % If we are not in the first iteration, check if the number of inliers founds is
           % higher than the previous maximum.
           elseif num_inliers > max_num_inliers
               
               % Take the indexes of the inliers
               inliers_2Dimage = sift_matches{i}(1,pos_inliers);
               inliers_3Dmodel = sift_matches{i}(2,pos_inliers);
               
               
               % Get the 2D and 3D coordinates for the inliers
               image_points_new = [keypoints{i}(1,inliers_2Dimage); keypoints{i}(2,inliers_2Dimage)]';
               world_points_new = model.coord3d(inliers_3Dmodel,:);
               
               
               % Reestimate the pose with PnP
               [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i),inlierIdx,status] = estimateWorldCameraPose(image_points_new, world_points_new, camera_params, 'MaxReprojectionError', max_reproj_err);
               % save the position of the best inliers with respect to the sift_matches 
               best_inliers_set{i} = pos_inliers;
               
               % Set the new maximum number of inliers as the last one
                max_num_inliers = num_inliers;
                
           end
           
        end
        
        
        
    end
    
end



%% Visualize inliers and the bounding box

% You can use the visualizations below or create your own one
% But be sure to present the bounding boxes drawn on the image to verify
% the camera pose

edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];

for i=1:num_files
    
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    
%   Plot inliers set
    PlotInlierOutlier(best_inliers_set{i}, camera_params, sift_matches{i}, model.coord3d, keypoints{i}, cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i))
%   Plot bounding box
    points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
    end
    hold off;
end