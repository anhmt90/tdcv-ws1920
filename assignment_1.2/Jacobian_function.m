function J = Jacobian_function(model3D, nextImage2D, camera_params, t, w)
% Compute Jacobian of the reprojection error with respect to the pose
% Arguments:
%   model3D: nx3 matrix of points from 3D model to be projected to the current frame (i+1)
%   nextImage2d: nx3 matrix of SIFT matches for the subsequent frame (image i+1)
%   camera_params: camera parameters containing intrinsic matrix
%   t: translation vector of the camera pose
%   w: orientation vector given in Exponential Maps
% Output:
%   J: Jacobian matrix

J = [];
tx = t(1); ty = t(2); tz = t(3);
wx = w(1); wy = w(2); wz = w(3);
f = camera_params.FocalLength(1);
u0 = camera_params.PrincipalPoint(1);
v0 = camera_params.PrincipalPoint(2);

for i = 1:size(model3D)
    X = model3D(i, 1); Y = model3D(i, 2); Z = model3D(i, 3);
    x = nextImage2D(i, 1); y = nextImage2D(i, 2);

    t2 = wx.^2;
    t3 = wy.^2;
    t4 = wz.^2;
    t5 = t2+t3+t4;
    t6 = sqrt(t5);
    t7 = sin(t6);
    t8 = 1.0./sqrt(t5);
    t9 = cos(t6);
    t10 = t9-1.0;
    t11 = 1.0./t5;
    t12 = t7.*t8.*wx;
    t26 = t10.*t11.*wy.*wz;
    t13 = t12-t26;
    t14 = t7.*t8.*wy;
    t15 = t10.*t11.*wx.*wz;
    t16 = t14+t15;
    t17 = t2+t3;
    t18 = t10.*t11.*t17;
    t19 = t18+1.0;
    t20 = 1.0./t5.^(3.0./2.0);
    t21 = 1.0./t5.^2;
    t22 = t3+t4;
    t23 = t7.*t20.*wx.*wy;
    t24 = t2.*t7.*t20.*wz;
    t25 = t2.*t10.*t21.*wz.*2.0;
    t27 = Y.*t13;
    t28 = Z.*t19;
    t40 = X.*t16;
    t29 = t27+t28-t40+tz;
    t30 = 1.0./t29;
    t31 = t10.*t17.*t21.*wx.*2.0;
    t32 = t7.*t17.*t20.*wx;
    t92 = t10.*t11.*wx.*2.0;
    t33 = t31+t32-t92;
    t34 = t9.*t11.*wx.*wy;
    t35 = t7.*t8;
    t36 = t2.*t9.*t11;
    t37 = t10.*t21.*wx.*wy.*wz.*2.0;
    t38 = t7.*t20.*wx.*wy.*wz;
    t94 = t2.*t7.*t20;
    t39 = t35+t36+t37+t38-t94;
    t41 = f.*tx;
    t42 = t7.*t8.*wz;
    t43 = t10.*t11.*wx.*wy;
    t44 = t42+t43;
    t45 = f.*t44;
    t59 = t13.*u0;
    t46 = t45-t59;
    t47 = tz.*u0;
    t48 = t16.*u0;
    t49 = t10.*t11.*t22;
    t50 = t49+1.0;
    t61 = f.*t50;
    t51 = t48-t61;
    t52 = t14-t15;
    t53 = f.*t52;
    t54 = t19.*u0;
    t55 = t53+t54;
    t56 = Z.*t55;
    t60 = Y.*t46;
    t62 = X.*t51;
    t57 = t41+t47+t56-t60-t62;
    t66 = t10.*t11.*wz;
    t58 = t23+t24+t25-t34-t66;
    t77 = t30.*t57;
    t63 = -t77+x;
    t64 = sign(t63);
    t65 = t3.*t7.*t20;
    t67 = t10.*t17.*t21.*wy.*2.0;
    t68 = t7.*t17.*t20.*wy;
    t75 = t10.*t11.*wy.*2.0;
    t69 = t67+t68-t75;
    t70 = t3.*t7.*t20.*wz;
    t71 = t3.*t10.*t21.*wz.*2.0;
    t72 = -t23+t34-t66+t70+t71;
    t73 = t3.*t9.*t11;
    t74 = 1.0./t29.^2;
    t76 = -t35+t37+t38+t65-t73;
    t78 = t7.*t20.*wx.*wz;
    t79 = t7.*t20.*wy.*wz;
    t80 = t9.*t11.*wy.*wz;
    t81 = t4.*t7.*t20.*wx;
    t82 = t4.*t10.*t21.*wx.*2.0;
    t83 = t10.*t17.*t21.*wz.*2.0;
    t84 = t7.*t17.*t20.*wz;
    t85 = t83+t84;
    t91 = t10.*t11.*wx;
    t86 = t79-t80+t81+t82-t91;
    t87 = t9.*t11.*wx.*wz;
    t88 = t4.*t7.*t20.*wy;
    t89 = t4.*t10.*t21.*wy.*2.0;
    t95 = t10.*t11.*wy;
    t90 = -t78+t87+t88+t89-t95;
    t93 = t2+t4;
    t96 = t2.*t7.*t20.*wy;
    t97 = t2.*t10.*t21.*wy.*2.0;
    t98 = X.*t58;
    t99 = Y.*t39;
    t100 = t98+t99-Z.*t33;
    t101 = f.*ty;
    t102 = t42-t43;
    t103 = f.*t102;
    t117 = t16.*v0;
    t104 = t103-t117;
    t105 = X.*t104;
    t106 = tz.*v0;
    t107 = t13.*v0;
    t108 = t10.*t11.*t93;
    t109 = t108+1.0;
    t110 = f.*t109;
    t111 = t107+t110;
    t112 = Y.*t111;
    t113 = t12+t26;
    t114 = f.*t113;
    t118 = t19.*v0;
    t115 = t114-t118;
    t119 = Z.*t115;
    t116 = t101+t105+t106+t112-t119;
    t127 = t30.*t116;
    t120 = -t127+y;
    t121 = sign(t120);
    t122 = t3.*t7.*t20.*wx;
    t123 = t3.*t10.*t21.*wx.*2.0;
    t124 = Y.*t72;
    t125 = X.*t76;
    t126 = t124+t125-Z.*t69;
    t128 = t4.*t7.*t20;
    t129 = X.*t86;
    t130 = Y.*t90;
    t131 = t129+t130-Z.*t85;
    J = [J; reshape([-t64.*(t30.*(Z.*(f.*(-t23+t24+t25+t34-t10.*t11.*wz)-t33.*u0)+Y.*(t39.*u0+f.*(t78+t96+t97-t10.*t11.*wy-t9.*t11.*wx.*wz))+X.*(t58.*u0-f.*(t7.*t20.*t22.*wx+t10.*t21.*t22.*wx.*2.0)))-t57.*t74.*t100),-t121.*(t30.*(X.*(f.*(-t78+t87-t95+t96+t97)+t58.*v0)+Z.*(f.*(-t35-t36+t37+t38+t94)-t33.*v0)+Y.*(t39.*v0-f.*(-t92+t7.*t20.*t93.*wx+t10.*t21.*t93.*wx.*2.0)))-t74.*t100.*t116),-t64.*(t30.*(-Z.*(t69.*u0-f.*(t35+t37+t38-t65+t73))+X.*(t76.*u0-f.*(-t75+t7.*t20.*t22.*wy+t10.*t21.*t22.*wy.*2.0))+Y.*(t72.*u0+f.*(t79+t122+t123-t10.*t11.*wx-t9.*t11.*wy.*wz)))-t57.*t74.*t126),-t121.*(t30.*(X.*(f.*(-t79+t80-t91+t122+t123)+t76.*v0)+Z.*(f.*(t23-t34-t66+t70+t71)-t69.*v0)+Y.*(t72.*v0-f.*(t7.*t20.*t93.*wy+t10.*t21.*t93.*wy.*2.0)))-t74.*t116.*t126),-t64.*(t30.*(Y.*(f.*(-t35+t37+t38+t128-t4.*t9.*t11)+t90.*u0)+Z.*(f.*(-t79+t80+t81+t82-t91)-t85.*u0)+X.*(t86.*u0-f.*(t10.*t11.*wz.*-2.0+t7.*t20.*t22.*wz+t10.*t21.*t22.*wz.*2.0)))-t57.*t74.*t131),-t121.*(t30.*(X.*(f.*(t35+t37+t38-t128+t4.*t9.*t11)+t86.*v0)+Z.*(f.*(t78-t87+t88+t89-t95)-t85.*v0)+Y.*(t90.*v0-f.*(t10.*t11.*wz.*-2.0+t7.*t20.*t93.*wz+t10.*t21.*t93.*wz.*2.0)))-t74.*t116.*t131),-f.*t30.*t64,0.0,0.0,-f.*t30.*t121,t64.*(t57.*t74-t30.*u0),t121.*(t74.*t116-t30.*v0)],[2,6])];
end