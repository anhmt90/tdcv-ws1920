function [q] = rot_quaternions(rotationMatrix)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf


r11 = rotationMatrix(1,1);
r12 = rotationMatrix(1,2);
r13 = rotationMatrix(1,3);
r21 = rotationMatrix(2,1);
r22 = rotationMatrix(2,2);
r23 = rotationMatrix(2,3);
r31 = rotationMatrix(3,1);
r32 = rotationMatrix(3,2);
r33 = rotationMatrix(3,3);


if (r33 < 0) 
    if (r11 >r22)
        t = 1 + r11 - r22 - r33;
        q = [ t, r12+r21, r31+r13, r32-r23 ];
        
    else
        t = 1 -r11 + r22 -r33;
        q = [ r12+r21, t, r23+r31, r31-r13 ];
    end
    
else 
        if (r11 < -r22) 
        t = 1 -r11 -r22 + r33;
        q = [ r31+r13, r23+r32, t, r12-r21 ];
        
        else
            t = 1 + r11 + r22 + r33;
            q = [ r23-r31, r31-r13, r12-r21, t ];
            
        end
        
end
n = 0.5 ./ sqrt(t);

q = n.*q;

end

