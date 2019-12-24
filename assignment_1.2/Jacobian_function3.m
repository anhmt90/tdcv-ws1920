function J = Jacobian_function(model3D, nextImage2D, camera_params, t, w)
    %JACOBIAN_FUNCTION
    %    J = JACOBIAN_FUNCTION(X,Y,Z,F,TX,TY,TZ,U0,V0,WX,WY,WZ,X,Y)
    
    %    This function was generated by the Symbolic Math Toolbox version 8.4.
    %    11-Dec-2019 23:22:20
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
        t5 = t2+t3;
        t6 = t2+t4;
        t7 = t3+t4;
        t8 = t4+t5;
        t9 = 1.0./t8;
        t11 = sqrt(t8);
        t10 = t9.^2;
        t12 = 1.0./t11;
        t14 = cos(t11);
        t15 = sin(t11);
        t13 = t12.^3;
        t16 = t14-1.0;
        t17 = t9.*t14.*wx.*wy;
        t18 = t9.*t14.*wx.*wz;
        t19 = t9.*t14.*wy.*wz;
        t20 = t12.*t15;
        t21 = t2.*t9.*t14;
        t22 = t3.*t9.*t14;
        t23 = t4.*t9.*t14;
        t24 = t20.*wx;
        t25 = t20.*wy;
        t26 = t20.*wz;
        t27 = t9.*t16.*wx;
        t28 = t9.*t16.*wy;
        t29 = t9.*t16.*wz;
        t30 = -t17;
        t31 = -t18;
        t32 = -t19;
        t33 = t13.*t15.*wx.*wy;
        t34 = t13.*t15.*wx.*wz;
        t35 = t13.*t15.*wy.*wz;
        t36 = -t20;
        t39 = -t22;
        t41 = t2.*t13.*t15;
        t42 = t3.*t13.*t15;
        t43 = t4.*t13.*t15;
        t63 = t10.*t16.*wx.*wy.*wz.*2.0;
        t68 = t3.*t10.*t16.*wx.*2.0;
        t69 = t2.*t10.*t16.*wy.*2.0;
        t70 = t4.*t10.*t16.*wx.*2.0;
        t71 = t2.*t10.*t16.*wz.*2.0;
        t72 = t4.*t10.*t16.*wy.*2.0;
        t73 = t3.*t10.*t16.*wz.*2.0;
        t74 = t5.*t9.*t16;
        t75 = t6.*t9.*t16;
        t76 = t7.*t9.*t16;
        t77 = t5.*t13.*t15.*wx;
        t78 = t5.*t13.*t15.*wy;
        t79 = t5.*t13.*t15.*wz;
        t83 = t5.*t10.*t16.*wx.*2.0;
        t84 = t5.*t10.*t16.*wy.*2.0;
        t85 = t5.*t10.*t16.*wz.*2.0;
        t37 = t27.*2.0;
        t38 = t28.*2.0;
        t40 = t29.*2.0;
        t44 = t27.*wy;
        t45 = t27.*wz;
        t46 = t28.*wz;
        t47 = t33.*wz;
        t48 = -t27;
        t50 = -t28;
        t52 = -t29;
        t54 = -t33;
        t55 = -t34;
        t56 = -t35;
        t57 = t42.*wx;
        t58 = t41.*wy;
        t59 = t43.*wx;
        t60 = t41.*wz;
        t61 = t43.*wy;
        t62 = t42.*wz;
        t64 = -t41;
        t80 = t74+1.0;
        t81 = t75+1.0;
        t82 = t76+1.0;
        t104 = t79+t85;
        t49 = -t37;
        t51 = -t38;
        t53 = -t40;
        t65 = -t44;
        t66 = -t45;
        t67 = -t46;
        t86 = X.*t82;
        t87 = Y.*t81;
        t88 = Z.*t80;
        t89 = t24+t46;
        t90 = t25+t45;
        t91 = t26+t44;
        t105 = Z.*t104;
        t113 = t20+t21+t47+t63+t64;
        t115 = t36+t39+t42+t47+t63;
        t117 = t30+t33+t52+t60+t71;
        t118 = t32+t35+t48+t59+t70;
        t119 = t17+t52+t54+t62+t73;
        t120 = t18+t50+t55+t61+t72;
        t92 = X.*t90;
        t93 = Y.*t91;
        t94 = Z.*t89;
        t95 = t24+t67;
        t96 = t25+t66;
        t97 = t26+t65;
        t106 = -t105;
        t107 = t49+t77+t83;
        t108 = t51+t78+t84;
        t114 = Y.*t113;
        t116 = X.*t115;
        t121 = X.*t117;
        t122 = X.*t118;
        t123 = Y.*t119;
        t124 = Y.*t120;
        t98 = X.*t97;
        t99 = -t92;
        t100 = Y.*t95;
        t101 = -t93;
        t102 = Z.*t96;
        t103 = -t94;
        t109 = Z.*t107;
        t110 = Z.*t108;
        t136 = t106+t122+t124;
        t111 = -t109;
        t112 = -t110;
        t125 = t86+t101+t102+tx;
        t126 = t87+t98+t103+ty;
        t127 = t88+t99+t100+tz;
        t128 = f.*t125;
        t129 = f.*t126;
        t130 = t127.*u0;
        t131 = t127.*v0;
        t132 = 1.0./t127;
        t137 = t111+t114+t121;
        t138 = t112+t116+t123;
        t133 = t132.^2;
        t134 = t128+t130;
        t135 = t129+t131;
        t139 = t132.*t134;
        t140 = t132.*t135;
        t141 = -t139;
        t142 = -t140;
        t143 = t141+x;
        t144 = t142+y;
        t145 = sign(t143);
        t146 = sign(t144);
        J_ = reshape([-t145.*(t132.*(t137.*u0+f.*(Y.*(t31+t34+t50+t58+t69)+Z.*(t17+t52+t54+t60+t71)-X.*(t7.*t10.*t16.*wx.*2.0+t7.*t13.*t15.*wx)))-t133.*t134.*t137),-t146.*(t132.*(f.*(X.*(t18+t50+t55+t58+t69)+Z.*(-t21+t36+t41+t47+t63)-Y.*(t49+t6.*t10.*t16.*wx.*2.0+t6.*t13.*t15.*wx))+t137.*v0)-t133.*t135.*t137),-t145.*(t132.*(f.*(Y.*(t32+t35+t48+t57+t68)+Z.*(t20+t22-t42+t47+t63)-X.*(t51+t7.*t10.*t16.*wy.*2.0+t7.*t13.*t15.*wy))+t138.*u0)-t133.*t134.*t138),-t146.*(t132.*(t138.*v0+f.*(X.*(t19+t48+t56+t57+t68)+Z.*(t30+t33+t52+t62+t73)-Y.*(t6.*t10.*t16.*wy.*2.0+t6.*t13.*t15.*wy)))-t133.*t135.*t138),-t145.*(t132.*(f.*(Z.*(t19+t48+t56+t59+t70)+Y.*(-t23+t36+t43+t47+t63)-X.*(t53+t7.*t10.*t16.*wz.*2.0+t7.*t13.*t15.*wz))+t136.*u0)-t133.*t134.*t136),-t146.*(t132.*(f.*(Z.*(t31+t34+t50+t61+t72)+X.*(t20+t23-t43+t47+t63)-Y.*(t53+t6.*t10.*t16.*wz.*2.0+t6.*t13.*t15.*wz))+t136.*v0)-t133.*t135.*t136),-f.*t132.*t145,0.0,0.0,-f.*t132.*t146,t145.*(t133.*t134-t132.*u0),t146.*(t133.*t135-t132.*v0)],[2,6]);
        J = [J; J_];
    end