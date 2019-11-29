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
    t7 = cos(t6);
    t8 = 1.0./t5;
    t9 = sin(t6);
    t10 = 1.0./t5.^(3.0./2.0);
    t11 = t7-1.0;
    t12 = 1.0./t5.^2;
    t13 = t3+t4;
    t14 = t2+t3;
    t15 = t9.*t10.*wx.*wy;
    t16 = t2.*t9.*t10.*wz;
    t17 = t2.*t11.*t12.*wz.*2.0;
    t18 = 1.0./sqrt(t5);
    t19 = t11.*t12.*t14.*wx.*2.0;
    t20 = t9.*t10.*t14.*wx;
    t59 = t8.*t11.*wx.*2.0;
    t21 = t19+t20-t59;
    t22 = t7.*t8.*wx.*wy;
    t23 = t9.*t18;
    t24 = t2.*t7.*t8;
    t25 = t11.*t12.*wx.*wy.*wz.*2.0;
    t26 = t9.*t10.*wx.*wy.*wz;
    t61 = t2.*t9.*t10;
    t27 = t23+t24+t25+t26-t61;
    t28 = t9.*t18.*wy;
    t29 = t8.*t11.*wx.*wz;
    t30 = t28+t29;
    t31 = t9.*t18.*wx;
    t38 = t8.*t11.*wy.*wz;
    t32 = t31-t38;
    t33 = Y.*t32;
    t34 = t8.*t11.*t14;
    t35 = t34+1.0;
    t36 = Z.*t35;
    t39 = X.*t30;
    t37 = t33+t36-t39+tz;
    t40 = 1.0./t37;
    t41 = f.*tx;
    t42 = t9.*t18.*wz;
    t43 = t8.*t11.*wx.*wy;
    t44 = t42+t43;
    t45 = f.*t44;
    t86 = t32.*u0;
    t46 = t45-t86;
    t47 = tz.*u0;
    t48 = t30.*u0;
    t49 = t8.*t11.*t13;
    t50 = t49+1.0;
    t88 = f.*t50;
    t51 = t48-t88;
    t52 = t28-t29;
    t53 = f.*t52;
    t54 = t35.*u0;
    t55 = t53+t54;
    t56 = Z.*t55;
    t87 = Y.*t46;
    t89 = X.*t51;
    t57 = t41+t47+t56-t87-t89;
    t65 = t8.*t11.*wz;
    t58 = t15+t16+t17-t22-t65;
    t60 = t2+t4;
    t62 = t9.*t10.*wx.*wz;
    t63 = t2.*t9.*t10.*wy;
    t64 = t2.*t11.*t12.*wy.*2.0;
    t66 = X.*t58;
    t67 = Y.*t27;
    t68 = t66+t67-Z.*t21;
    t69 = 1.0./t37.^2;
    t70 = f.*ty;
    t71 = t42-t43;
    t72 = f.*t71;
    t91 = t30.*v0;
    t73 = t72-t91;
    t74 = X.*t73;
    t75 = tz.*v0;
    t76 = t32.*v0;
    t77 = t8.*t11.*t60;
    t78 = t77+1.0;
    t79 = f.*t78;
    t80 = t76+t79;
    t81 = Y.*t80;
    t82 = t31+t38;
    t83 = f.*t82;
    t92 = t35.*v0;
    t84 = t83-t92;
    t93 = Z.*t84;
    t85 = t70+t74+t75+t81-t93;
    t104 = t40.*t57;
    t90 = -t104+x;
    t103 = t40.*t85;
    t94 = -t103+y;
    t95 = t3.*t9.*t10.*wz;
    t96 = t3.*t11.*t12.*wz.*2.0;
    t97 = t11.*t12.*t14.*wy.*2.0;
    t98 = t9.*t10.*t14.*wy;
    t105 = t8.*t11.*wy.*2.0;
    t99 = t97+t98-t105;
    t100 = -t15+t22-t65+t95+t96;
    t101 = t3.*t9.*t10;
    t106 = t3.*t7.*t8;
    t102 = -t23+t25+t26+t101-t106;
    t107 = t7.*t8.*wy.*wz;
    t108 = t3.*t9.*t10.*wx;
    t109 = t3.*t11.*t12.*wx.*2.0;
    t110 = Y.*t100;
    t111 = X.*t102;
    t112 = t110+t111-Z.*t99;
    t113 = t90.^2;
    t114 = t94.^2;
    t115 = t113+t114;
    t116 = 1.0./sqrt(t115);
    t117 = t9.*t10.*wy.*wz;
    t118 = t7.*t8.*wx.*wz;
    t119 = t4.*t9.*t10.*wx;
    t120 = t4.*t11.*t12.*wx.*2.0;
    t121 = t11.*t12.*t14.*wz.*2.0;
    t122 = t9.*t10.*t14.*wz;
    t123 = t121+t122;
    t128 = t8.*t11.*wx;
    t124 = -t107+t117+t119+t120-t128;
    t125 = t4.*t9.*t10.*wy;
    t126 = t4.*t11.*t12.*wy.*2.0;
    t130 = t8.*t11.*wy;
    t127 = -t62+t118+t125+t126-t130;
    t129 = t4.*t9.*t10;
    t131 = X.*t124;
    t132 = Y.*t127;
    t133 = t131+t132-Z.*t123;
    J = [J; t116.*(t90.*(t40.*(Z.*(f.*(-t15+t16+t17+t22-t65)-t21.*u0)+Y.*(t27.*u0+f.*(t62+t63+t64-t8.*t11.*wy-t7.*t8.*wx.*wz))+X.*(t58.*u0-f.*(t9.*t10.*t13.*wx+t11.*t12.*t13.*wx.*2.0)))-t57.*t68.*t69).*2.0+t94.*(t40.*(X.*(f.*(-t62+t63+t64+t118-t8.*t11.*wy)+t58.*v0)+Z.*(f.*(-t23-t24+t25+t26+t61)-t21.*v0)+Y.*(t27.*v0-f.*(-t59+t9.*t10.*t60.*wx+t11.*t12.*t60.*wx.*2.0)))-t68.*t69.*t85).*2.0).*(-1.0./2.0),t116.*(t94.*(t40.*(X.*(f.*(t107+t108+t109-t117-t8.*t11.*wx)+t102.*v0)+Z.*(f.*(t15-t22-t65+t95+t96)-t99.*v0)+Y.*(t100.*v0-f.*(t9.*t10.*t60.*wy+t11.*t12.*t60.*wy.*2.0)))-t69.*t85.*t112).*2.0+t90.*(t40.*(-Z.*(t99.*u0-f.*(t23+t25+t26-t101+t106))+Y.*(f.*(-t107+t108+t109+t117-t8.*t11.*wx)+t100.*u0)+X.*(t102.*u0-f.*(-t105+t9.*t10.*t13.*wy+t11.*t12.*t13.*wy.*2.0)))-t57.*t69.*t112).*2.0).*(-1.0./2.0),t116.*(t90.*(t40.*(Y.*(f.*(-t23+t25+t26+t129-t4.*t7.*t8)+t127.*u0)+Z.*(f.*(t107-t117+t119+t120-t128)-t123.*u0)+X.*(t124.*u0-f.*(t8.*t11.*wz.*-2.0+t9.*t10.*t13.*wz+t11.*t12.*t13.*wz.*2.0)))-t57.*t69.*t133).*2.0+t94.*(t40.*(X.*(f.*(t23+t25+t26-t129+t4.*t7.*t8)+t124.*v0)+Z.*(f.*(t62-t118+t125+t126-t130)-t123.*v0)+Y.*(t127.*v0-f.*(t8.*t11.*wz.*-2.0+t9.*t10.*t60.*wz+t11.*t12.*t60.*wz.*2.0)))-t69.*t85.*t133).*2.0).*(-1.0./2.0),-f.*t40.*t90.*t116,-f.*t40.*t94.*t116,t116.*(t90.*(t57.*t69-t40.*u0).*2.0+t94.*(t69.*t85-t40.*v0).*2.0).*(1.0./2.0)];
end