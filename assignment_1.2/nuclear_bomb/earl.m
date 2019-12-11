function [R, t] = earl(rotationMatrix, translationVector, cameraParameters, points3d, points2d, threshIRLS, numiter)
%ENERGY_FUNCTION Summary of this function goes here
%   rotationParameters: rotation matrix
%   translationParameters: translation matrix
%   cameraParameters: cameraParameters type input argument describing
%       intrinsics matrix
%   points3d: Mi, 3D points corresponding to 2d points
%   points2d: mi, 2D points corresponding to 3d points

% Something about rodrigues formula
%rotationParameters = rotationMatrixToVector(rotationMatrix);+
% Don't know if we need transpose or not?
%cameraParameters.IntrinsicMatrix = cameraParameters.IntrinsicMatrix';

rotationVector = rotationMatrixToVector(rotationMatrix);

points3d = points3d';

% Try check results with matlab jacobian
%jackie = Jacobian_function2(points3d, points2d, cameraParameters, translationVector, rotationVector);
%jackie = jack(cameraParameters.IntrinsicMatrix', rotationVector, points2d, points3d);

% est_points2d = project3d2image(points3d, cameraParameters, ...
%         rotationMatrix, translationVector);
% 
% diff = pdist2(points2d', est_points2d');
% energy = trace(diff);

lambda = 0.001;
old_delta = threshIRLS + 1; % ensure step is initialized bigger than thresh

for i = 1:numiter
    if old_delta < threshIRLS % If we move too slowly, we've converged already
        break
    end
      J = Jacob_fd(cameraParameters, rotationVector, translationVector, points3d);
%     J = Jacobian_function2(points3d, points2d, cameraParameters, translationVector, rotationVector);
%     J = jack(cameraParameters.IntrinsicMatrix', rotationVector, points2d, points3d);
    [RHO, W] = E(cameraParameters, rotationMatrix, translationVector, points3d, points2d);
    delta = -inv(J'*W*J + lambda*eye(numel(J(1,:)))) * (J'*W*RHO');
    
    new_error = E(cameraParameters, rotationVectorToMatrix(rotationVector + delta(1:3)'), translationVector + delta(4:6)', points3d, points2d);
    if new_error > RHO
        lambda = lambda * 10;
    else
        lambda = lambda / 10;
        rotationMatrix = rotationVectorToMatrix(rotationVector + delta(1:3)');
        translationVector = translationVector + delta(4:6)';
    end
    old_delta = norm(delta);
end

R = rotationMatrix;
t = translationVector;

end




