% Symbolic representation of the Jacobian computation
% Export to function with:
%   Jacobian_function = matlabFunction(J, 'File', 'Jacobian_function4');

syms u0 v0 f; % Camera intrinsics
syms wx wy wz tx ty tz; % Camera extrinsics
syms X Y Z; % 3D point
syms x y; % 2D point

% Intrinsic matrix
K=[f 0 u0; 0 f v0; 0 0 1];

% Rodrigues' Formula
theta=sqrt(wx^2+wy^2+wz^2);
omega=  [0 -wz wy; wz 0 -wx; -wy wx 0;];
R = eye(3) + (sin(theta)/theta)*omega + ((1-cos(theta))/theta^2)*(omega*omega);
R = R'; % differs from Jacobian_function

% Translation vector
t=[tx ty tz].';
point3D = [X Y Z 1].';
point2D = [x y];

% Reprojected point
uvs=K*[R -R*t]*point3D;
re_point=[uvs(1)/uvs(3) uvs(2)/uvs(3)];

% Euclidean distance
%d = sqrt((re_point(1, 1)-point2D(1, 1))^2+(re_point(1, 2)-point2D(1, 2))^2);
du = abs(point2D(1, 1)-re_point(1, 1));
dv = abs(point2D(1, 2)-re_point(1, 2));

J = [diff(du, wx) diff(du, wy) diff(du, wz)...
    diff(du, tx) diff(du, ty) diff(du, tz);
    diff(dv, wx) diff(dv, wy) diff(dv, wz)...
    diff(dv, tx) diff(dv, ty) diff(dv, tz)];