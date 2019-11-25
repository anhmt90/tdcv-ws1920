function [best_inliers_set, max_num_inliers] = RANSAC(iterations, threshold, sift_matches, keypoints, Coord3D, camera_params)
%RANSAC Summary of this function goes here
%   Detailed explanation goes here

% Initialization of max num of inliers
max_num_inliers = 0;

for j = 1:iterations
    
    perm = randperm(size(sift_matches,2));
    % We want to take 4 random points each time
    sel = perm(1:4);
    
    % Get those 4 random points from the matches between the model and the 2D image.
    matched_points_image2D = sift_matches(1,sel);
    matched_points_model3D = sift_matches(2,sel);
    
    % Get the 2D and 3D coordinates for the 4 randomly selected matches
    % and estimate the pose using PnP
    image_points = [keypoints(1,matched_points_image2D); keypoints(2,matched_points_image2D)]';
    world_points = Coord3D(matched_points_model3D,:);
    
    [init_world_orientations(:,:),init_world_locations(:,:),inlierIdx,status] = estimateWorldCameraPose(image_points, world_points, camera_params,...
        'MaxReprojectionError', 10000,'MaxNumTrials',10,...
        'Confidence',0.0000001);
    
    %%% Get the indexes of all matches %%%
    all_matchedPoints_2Dimage = sift_matches(1,:);
    all_matchedPoints_3Dmodel = sift_matches(2,:);
    
    % Get 2D coordinates of all matches
    all_image_points = [keypoints(1,all_matchedPoints_2Dimage); keypoints(2,all_matchedPoints_2Dimage)]';
    % Get 3D coordinates of all matches
    all_model_points = Coord3D(all_matchedPoints_3Dmodel,:);
    
    % Project all of the 3D points from the matches into the 2D image
    % using the obtained orientation and translation transformations
    reprojected_points = project3d2image(all_model_points',camera_params, init_world_orientations(:,:), init_world_locations(:, :));
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
    if any(d < threshold)
        
        % Determine which matched points are considered inliers
        % (distance smaller than the threshold)
        [pos_inliers,col] = find(d < threshold);
        num_inliers = size(pos_inliers,1);
        
        % If it is the first iteration of one image, we set the maximum of inliers as
        % the first ones.
        if max_num_inliers == 0
            
            max_num_inliers = num_inliers;
            % save the position of the best inliers with respect to the sift_matches
            best_inliers_set = pos_inliers;

            % If we are not in the first iteration, check if the number of inliers founds is
            % higher than the previous maximum.
        elseif num_inliers > max_num_inliers
            
            % save the position of the best inliers with respect to the sift_matches
            best_inliers_set = pos_inliers;
            
            % Set the new maximum number of inliers as the last one
            max_num_inliers = num_inliers;
            
        end
        
    end
    
end
end

