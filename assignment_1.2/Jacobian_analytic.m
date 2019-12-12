function J = Jacobian_analytic(M_3D, K, R, t)
    
    M_homo = [M_3D, ones(size(M_3D, 1), 1)];
    m_ = K * [R t(:)] * M_homo';

%     m_ = K * (R*M_3D + t);

    grad_m_m_ = get_grad_m_m_(m_');

    grad_m_M = repmat(K, 1, size(m_',1));

    grad_M_p = get_grad_M_p(R, M_3D);

    J = grad_m_m_ * grad_m_M * grad_M_p;
end


%%
function res = get_grad_m_m_(m_)
% m_ is of shape Nx3, where N is the #points
N = size(m_,1);
res = zeros(2*N, 3);
j=1;
for i = 1:N
    U = m_(i,1);
    V = m_(i,2);
    W = m_(i,3);
    
    grad_m_m_i = [1/W     0     -U/W^2 ;
                   0     1/W    -V/W^2];
    
    res(j:j+1,:) = grad_m_m_i;
    j = j + 2;
end
end

function res = get_grad_M_p(R, M)
% M is of shape Nx3, where N is the #points
N = size(M,1);
res = zeros(3*N, 6);
j = 1;

grad_R1 = grad_R(R,1);
grad_R2 = grad_R(R,2);
grad_R3 = grad_R(R,3);

for i = 1:N
    M_i = M(i,:).';
    grad_M_p_i = [grad_R1*M_i, grad_R2*M_i, grad_R3*M_i, eye(3)];
    
    res(j:j+2,:) = grad_M_p_i;
    j = j+3;
end
end


function d_R = grad_R(R, i)
% R : Rotation matrix
% i : component index of v w.r.t which gradient of R should be computed
I = eye(3);
v = rotationMatrixToVector(R);
v = v(:);

d_R = (I - R)*I(:,i);
d_R = cross(v, d_R);
d_R = v(i)*hat(v) + hat(d_R);
d_R = (1/(norm(v)^2)) * d_R * R;
end

function v_hat = hat(v)
% v should be of 3x1
    v_hat = [0 -v(3) v(2); v(3) 0 -v(1); -v(2)  v(1)  0];
end