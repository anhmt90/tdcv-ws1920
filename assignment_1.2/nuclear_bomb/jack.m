function J = jack(intrinsicsMatrix, rotationVector, points2d, points3d)
    %Jacobian
    %   intrinsicsMatrix: matrix describing camera intrinsics
    %   rotationVector: rotation matrix expressed as exponential map
    %   points2d
    %   points3d
    vx = skew_sym_mat(rotationVector);
    I = eye(3);
    rotationMatrix = rotationVectorToMatrix(rotationVector);

    for i = 1:3
        e = I(:,i); % ith base vector of R
        dRdv{i} = ((rotationVector(i)*vx + skew_sym_mat(cross(rotationVector, (I-rotationMatrix)*e)))/(norm(rotationVector)^2))*rotationMatrix;
    end
    dmtildedM = intrinsicsMatrix;
    for M = 1: numel(points3d(1,:))
        dmdmtilde = [1 0 -points2d(1,M) ; 0 1 -points2d(2,M)];
        dMdp = [dRdv{1}*points3d(:, M) dRdv{2}*points3d(:, M) dRdv{3}*points3d(:, M) eye(3)];
        J(2*M-1:2*M,:) = dmdmtilde * dmtildedM * dMdp;
    end
end

