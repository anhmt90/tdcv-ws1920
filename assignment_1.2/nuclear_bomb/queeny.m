function [q] = queeny(rotationMatrix)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

theta = acos(0.5*(trace(rotationMatrix) - 1));
r = rotationMatrix;

w = (1/2*sin(theta))*[r(3,2) - r(2,3); r(1,3) - r(3,1); r(2,1) - r(1,2)];
a = norm(w)/2;

q = [cos(a), sin(a)*w(1)/2*a, sin(a)*w(2)/2*a, sin(a)*w(3)/2*a];

end

