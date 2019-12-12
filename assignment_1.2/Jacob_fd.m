function [J] = Jacob_fd(camera_params, rotationVector, translationVector, points3d)
%JACOB_FD Summary of this function goes here
%   Detailed explanation goes here
%Finite Diferences

%Initializing the Jacobian matrix
J = zeros(size(points3d ,1)*2,6);
delta = 1e-9;
for j=1:3
    %We are going to calculate the Jacobian by central differences,
    %which has the form of dF/dx = (F(x+delta)-F(x-delta))/2delta
    diffVect = zeros(1,3);
    diffVect(1,j) = 1;
    diffVect = [rotationVector + delta * diffVect;rotationVector - delta * diffVect];
    %Calculating partial derivatives wrt rotation params as:
    reprojectedDiff_Pos = project3d2image(points3d', camera_params, rotationVectorToMatrix(diffVect(1,:)), translationVector);
    reprojectedDiff_Neg = project3d2image(points3d', camera_params, rotationVectorToMatrix(diffVect(2,:)), translationVector);
    %Partial derivative of the reprojected error (ReprojectedPoints -
    %Matched Points). But since Matched Points do not depend on pose
    %params, we can calculate it only as the difference between the
    %reprojected differences
    partialDev = (reprojectedDiff_Pos - reprojectedDiff_Neg);
    partialDev = partialDev/(2*delta);
    J(1:2:end,j) = partialDev(1,:)';
    J(2:2:end,j) = partialDev(2,:)';
    
    %Here, We do the same wrt translation parameters
    diffTrans = zeros(1,3);
    diffTrans(1,j) = 1;
    diffTrans = [translationVector + delta * diffTrans;translationVector - delta * diffTrans];
    % Project match points from 3D model to the current frame (i+1) (2D image)
    reprojectedDiff_Pos = project3d2image(points3d', camera_params, rotationVectorToMatrix(rotationVector), diffTrans(1,:));
    reprojectedDiff_Neg = project3d2image(points3d', camera_params, rotationVectorToMatrix(rotationVector), diffTrans(2,:));

    partialDev = (reprojectedDiff_Pos - reprojectedDiff_Neg);
    partialDev = partialDev/(2*delta);
    J(1:2:end,j+3) = partialDev(1,:)';
    J(2:2:end,j+3) = partialDev(2,:)';
end

end

