function [rho, W] = E(cameraParameters, rotationMatrix, translationVector, points3d, points2d)
    % Points3d transpose?
    est_points2d = project3d2image(points3d, cameraParameters, ...
        rotationMatrix, translationVector);

    %diff = pdist2(points2d', est_points2d');
    % sum of squared errors
    diff=abs(points2d-est_points2d);
    error=diff(1:1:end)'; % 1 row [x y x y x y...] to match J structure

    % Implement Tukey
    c = 4.685;
    in = find(error<=c);
    out = find(error>c);
    rho(in) = c^2/6*(1-(1-(error(in)/c).^2).^3);
    rho(out) = c^2/6;
    
    MAD = median(abs(error));
    sigma = 1.48257968*MAD;
    err_sig = error/sigma;
    
    in = find(err_sig<c);
    out = find(err_sig>=c);
    w(in) = (1-(err_sig(in)/c).^2).^2;
    w(out) = 0;
    W = diag(w);
end