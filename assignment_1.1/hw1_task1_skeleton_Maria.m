clear
clc
close all
addpath('helper_functions')
addpath(genpath('vlfeat-0.9.21/'))

%% Setup
% path to the images folder
path_img_dir = 'data/init_texture';
% path to object ply file
object_path = 'data/teabox.ply';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Coordinate System is right handed and placed at lower left corner of tea
% box. Using coordinates of teabox model, vertex numbering is visualized in
% image vertices.png

imshow('vertices.png')
title('Vertices numbering')

%% Label images
% You can use this function to label corners of the model on all images
% This function will give an array with image coordinates for all points
% Be careful that some points may not be visible on the image and so this
% will result in NaN values in the output array
% Don't forget to filter NaNs later
%num_points = 8;
%labeled_points = mark_image(path_img_dir, num_points);


% Save labeled points and load them when you rerun the code to save time
% save('labeled_points.mat', 'labeled_points')
load('labeled_points.mat')

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);


%% Check corners labeling by plotting labels
for i=1:length(Filenames)
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit')
    title(sprintf('Image: %d', i))
    hold on
    for point_idx = 1:8
        x = labeled_points(point_idx,1,i);
        y = labeled_points(point_idx,2,i); 
        if ~isnan(x)
            plot(x,y,'x', 'LineWidth', 3, 'MarkerSize', 15)
            text(x,y, char(num2str(point_idx)), 'FontSize',12)
        end
    end
end


%% Call estimateWorldCameraPose to perform PnP

% Place estimated camera orientation and location here to use
% visualisations later
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

focal_length = 2960.37845;
cx = 1841.68855;
cy = 1235.23369;
%imageSize = [2456 3680];

camera_params = cameraParameters('IntrinsicMatrix',[focal_length 0 0; 0 focal_length 0; cx cy 1]);%,'ImageSize',imageSize);

max_reproj_err = 5;

% iterate over the images
for i=1:num_files
    
    fprintf('Estimating pose for image: %d \n', i)

%   TODO: Estimate camera pose for every image
%     In order to estimate pose of the camera using the function bellow you need to:
%   - Prepare image_points and corresponding world_points
%   - Setup camera_params using cameraParameters() function
%   - Define max_reproj_err - take a look at the documentation and
%   experiment with different values of this parameter 
        
    
    % First remove all NaNs values for the vertices.
    image_points = labeled_points(:,:,i);
    [rows,cols] = find(isnan(image_points));
    image_points(rows,:) = []; 
   
    % The world points correspond to the 3D coordinates of each vertices.
    world_points = vertices;
    world_points(rows,:) = [];
    
    [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', max_reproj_err);

end

%% Visualize computed camera poses

% Edges of the object bounding box
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
visualise_cameras(vertices, edges, cam_in_world_orientations, cam_in_world_locations);


%% Detect SIFT keypoints in the images

% You will need vl_sift() and vl_plotframe() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path

%Place SIFT keypoints and corresponding descriptors for all images here
 keypoints = cell(num_files,1); 
 descriptors = cell(num_files,1); 

for i=1:length(Filenames)
    fprintf('Calculating sift features for image: %d \n', i)

%    TODO: Prepare the image (img) for vl_sift() function
    I = imread(Filenames{i});
    img = single(rgb2gray(I));
    
    [keypoints{i}, descriptors{i}] = vl_sift(img) ;
end

% When you rerun the code, you can load sift features and descriptors to

% Save sift features and descriptors and load them when you rerun the code to save time
save('sift_descriptors.mat', 'descriptors')
save('sift_keypoints.mat', 'keypoints')
%load('sift_descriptors.mat');
%load('sift_keypoints.mat');


% Visualisation of sift features for the first image
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
vl_plotframe(keypoints{1}(:,:), 'linewidth',2);
title('SIFT features')
hold off;

%% Build SIFT model
% Filter SIFT features that correspond to the features of the object

% Project a 3D ray from camera center through a SIFT keypoint
% Compute where it intersects the object in the 3D space
% You can use TriangleRayIntersection() function here

% Your SIFT model should only consists of SIFT keypoints that correspond to
% SIFT keypoints of the object
% Don't forget to visualise the SIFT model with the respect to the cameras
% positions


% num_samples - number of SIFT points that is randomly sampled for every image
% Leave the value of 1000 to retain reasonable computational time for debugging
% In order to contruct the final SIFT model that will be used later, consider
% increasing this value to get more SIFT points in your model
num_samples = 5000;
size_total_sift_points = num_samples*num_files;

% Visualise cameras and model SIFT keypoints
%visualise_cameras(vertices, edges, cam_in_world_orientations, cam_in_world_locations);
%hold on

% Place model's SIFT keypoints coordinates and descriptors here
model.coord3d = [];
model.descriptors = [];
model_keypoints = [];


for i=1:num_files
    
    %     Randomly select a number of SIFT keypoints
    perm = randperm(size(keypoints{i},2)) ;
    sel = perm(1:num_samples);
    
    %    Section to be deleted starts here
    P = camera_params.IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];    
    Q = P(:,1:3);
    q = P(:,4);
    C = -inv(Q)*q; % this corresponds to C
    %    Section to be deleted ends here
    
    for j=1:num_samples
        
        % TODO: Perform intersection between a ray and the object
        % You can use TriangleRayIntersection to find intersections
        % Pay attention at the visible faces from the given camera position
        
        % SIFT point under examination
        point = [keypoints{i,1}(1,sel(j)), keypoints{i,1}(2,sel(j)), 1]';
        % Ray generated from origin (origin of camera in that view + SIFT
        % point)
        r = C + inv(Q)*point;
        
        for h = 1:size(faces,1)
            
            % Calculate intersection between ray and face/triangle of the
            % box we are looking at. With the added parameters to the
            % function it should take into account if the point intersects
            % in a face that is visible or not
            
            [intersect, t, u, v, xcoord] = TriangleRayIntersection(repmat(C',3,1),repmat(r',3,1),...
                vertices(faces(h,1)+1,:),vertices(faces(h,2)+1,:),vertices(faces(h,3)+1,:),...
                'planeType','one sided', 'border','inclusive');
            
            %%%%%%%%%%%%%%%%%%%%
            %%% This part of the code helps visualizing if the sift point
            %%% that has been taken is said by the algorithm to be inside the 3D model or not. 
            %%% To visualize it better:
            %%% - put a break point on [intersect, t,u, v, xcoord]...  
            %%% - uncomment the lines below.
            %%% - run face by face for a point inside the box
            
            %%% Transforming vertices to homogeneous coordinate
            % vert_test = [vertices ones(1,8)'];
            
            %%% projecting the vertices to the image plane using the
            %%% projection matrix
            % xc = P*vert_test';
            
            %%% Transpose the projected vertices
            % xc = xc';
            
            %%% divide by the 3 coordinate to go back to 2D positions
            % xc = xc(:,1:2)./xc(:,3);
            
            %%% Plot only the vertices that are on that face we are at
            % sel_xc = xc(faces(h,:)+1,:);
            
            %%% Display if the point intersects or not with the face.
            %%% 0 = no intersection
            %%% 1 = intersection
            % disp(intersect)
               
            %imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
            %hold on;
            %scatter(sel_xc(:,1),sel_xc(:,2),'g*')
            %scatter(keypoints{i}(1,sel(j)),keypoints{i}(2,sel(j)),'rx');
            
            %%%%%%%%%%%%%%
            
            % If they intersect, save the intersection point + the
            % descriptors of that SIFT point.
            if intersect == 1
                
                % Added this variable to visualize the result of the
                % sift keypoints that are said to belong to the model on
                % each image
                model_keypoints = [model_keypoints, keypoints{i,1}(:,sel(j))];
                
                model.coord3d = [model.coord3d; xcoord];
                model.descriptors = [model.descriptors, descriptors{i,1}(:,sel(j))];
                
            end
        end
    end
    
    
    % Images to check if the points that are said to belong to the model
    % are inside the box in each image
    figure
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    hold on;
    scatter(model_keypoints(1,:),model_keypoints(2,:),'bo');
    hold off
    
    % Reinitialize the model_keypoints variable for each image
    model_keypoints = [];
end

figure
scatter3(model.coord3d(:,1), model.coord3d(:,2), model.coord3d(:,3), 'o', 'b');

hold off
xlabel('x');
ylabel('y');
zlabel('z');

% Save your sift model for the future tasks
save('sift_model.mat', 'model');

%% Visualise only the SIFT model
figure()
scatter3(model.coord3d(:,1), model.coord3d(:,2), model.coord3d(:,3), 'o', 'b');
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
