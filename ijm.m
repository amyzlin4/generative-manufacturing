% HKS-CNN injection molding data generation

clear all
close all
clc

num_ex = 1000;

for i = 1:num_ex
    tic
    num_cut = randi([0,5],1);
    POINT = [];
    NORMAL = [];
    BB = zeros(num_cut,4);
    len = zeros(num_cut+1,1);

    % create internal cuts
    if num_cut > 0
        tran = zeros(num_cut,2);
        for j = 1:num_cut
            [normal, point] = make_shape(1, 2);

            if j == 1
                BB(1,:) = bound_box(point);
            else
                BB(j,:) = bound_box(point);
                index_x = 0;
                index_y = 0;
                while index_x==0 && index_y==0
                    index_x = randi([-1,1],1);
                    index_y = randi([-1,1],1);
                end
                x_tran = index_x*(rand(1) + 1);
                y_tran = index_y*(rand(1) + 1);
                flag_c = 1;
                while flag_c == 1
                    if collision_check(BB, x_tran, y_tran, j)
                        x_tran = 1.2 * x_tran;
                        y_tran = 1.2 * y_tran;
                    else
                        flag_c = 2;
                        point(1,:) = point(1,:) + x_tran;
                        point(2,:) = point(2,:) + y_tran;
                        BB(j,1:2) = BB(j,1:2) + x_tran;
                        BB(j,3:4) = BB(j,3:4) + y_tran;
                        tran(j,:) = [x_tran, y_tran];
                    end
                end
            end

            POINT = [POINT, point];
            NORMAL = [NORMAL, normal];
            len(j,1) = size(point,2);
        end
    end

    % create external boundary
    [normal, point] = make_shape(2, 4);

    center = [0; 0];
    if num_cut ~= 0
        BB_inter = [min(BB(:,1)), max(BB(:,2)), min(BB(:,3)), max(BB(:,4))];
        center = [(BB_inter(1)+BB_inter(2))/2; (BB_inter(3)+BB_inter(4))/2];

        flag_ie_c = 1;
        while flag_ie_c == 1
            if collision_ie_check(BB_inter, point+center)
                point = 1.2 * point;
            else
                flag_ie_c = 2;
            end
        end
    end
    POINT = [POINT, point+center];
    NORMAL = [NORMAL, normal];
    len(end,1) = size(point,2);

    % mesh in 2D plane
    if num_cut == 0
        pgon_new = polyshape(POINT(1,:),POINT(2,:),'SolidBoundaryOrientation','ccw');
    else
        index = zeros(num_cut+1,2);
        for j = 1:num_cut+1
            if j == 1
                index(1,:) = [1, len(1,1)];
            else
                index(j,:) = [sum(len(1:j-1,1))+1, sum(len(1:j,1))];
            end
        end
        pgon_new = polyshape(POINT(1,index(end,1):index(end,2)),POINT(2,index(end,1):index(end,2)),'SolidBoundaryOrientation','ccw');
        for j = 1:num_cut
            pgon_old = polyshape(POINT(1,index(j,1):index(j,2)),POINT(2,index(j,1):index(j,2)),'SolidBoundaryOrientation','ccw');
            pgon_new = subtract(pgon_new, pgon_old);
        end
    end

    T_final = triangulation(pgon_new);
    V = T_final.Points;
    F = T_final.ConnectivityList;
    V1 = pgon_new.Vertices;

    nan_index = find(isnan(V1(:,1))==1);
    n_nan = length(nan_index);
    n_bnd = n_nan + 1;
    if n_bnd ~= num_cut + 1
        warning('Boundary count mismatch (expected %d, got %d). Skipping.', num_cut+1, n_bnd);
        continue;
    end
    len = zeros(n_bnd, 1);
    if n_nan == 0
        len(1,1) = size(V1,1);
    else
        for j = 1:n_nan
            if j == 1
                len(j,1) = nan_index(j,1)-1;
            else
                len(j,1) = nan_index(j,1)-nan_index(j-1,1)-1;
            end
        end
        len(end,1) = size(V1,1)-nan_index(end,1);
    end

    % reorder vertices so external boundary comes last
    POINT = [V(len(1,1)+1:end,:)', V(1:len(1,1),:)'];
    len = [len(2:end,1); len(1,1)];
    index_reorder = [[1:1:len(end,1)]+(size(POINT,2)-len(end,1)), 1:1:size(POINT,2)-len(end,1)];
    T_new = zeros(size(F,1),3);
    for j = 1:size(F,1)
        for k = 1:3
            T_new(j,k) = index_reorder(1,F(j,k));
        end
    end

    % get index of cuts
    index = zeros(num_cut,2);
    for j = 1:num_cut
        if j == 1
            index(j,:) = [1, len(1,1)];
        else
            index(j,:) = [sum(len(1:j-1,1))+1, sum(len(1:j,1))];
        end
    end

    % create extrusion in z axis
    x_dist = max(POINT(1,:)) - min(POINT(1,:));
    y_dist = max(POINT(2,:)) - min(POINT(2,:));
    z_max = min(x_dist, y_dist);
    z = max(rand(1)*0.05, 0.01) * z_max;

    n_pts = size(POINT,2);
    n_tri = size(T_new,1);
    point3D = [];
    normal3D = [];
    face3D = [];

    % bottom plane
    point_new = zeros(3, n_pts);
    point_new(1:2,:) = POINT;
    point3D = [point3D, point_new];
    normal3D = [normal3D, [zeros(2, n_tri); -ones(1, n_tri)]];
    face3D = [face3D; T_new];

    % top plane
    point_new = zeros(3, n_pts);
    point_new(1:2,:) = POINT;
    point_new(3,:) = z;
    point3D = [point3D, point_new];
    normal3D = [normal3D, [zeros(2, n_tri); ones(1, n_tri)]];
    face3D = [face3D; T_new + n_pts];

    % outer surface boundary (lateral wall around external edge)
    index_b = sum(len)-len(end,1)+1:1:sum(len);
    index_t = index_b + sum(len);
    ratio = rand(1)*0.04+1.01;
    point3D(:,index_b) = ratio*point3D(:,index_b);
    z_lateral = (rand(1)*0.4+0.1) * z_max;
    ratio_xy = randsrc(1,1, [1,rand(1)*0.5+1; 1/2,1/2]);
    point_new_b = point3D(:,index_b)*ratio_xy;
    point_new_b(3,:) = z_lateral;
    point_new_t = point3D(:,index_t)*ratio_xy;
    point_new_t(3,:) = z_lateral;
    index_b_new = size(point3D,2)+1:1:size(point3D,2)+len(end,1);
    index_t_new = size(point3D,2)+len(end,1)+1:1:size(point3D,2)+2*len(end,1);
    point3D = [point3D, point_new_b, point_new_t];
    index_lateral = zeros(4, len(end,1)+1);
    index_lateral(1,:) = [index_b, index_b(1)];
    index_lateral(2,:) = [index_b_new, index_b_new(1)];
    index_lateral(3,:) = [index_t_new, index_t_new(1)];
    index_lateral(4,:) = [index_t, index_t(1)];
    for j = 1:len(end,1)
        for k = 1:3
            tri = [index_lateral(k,j), index_lateral(k,j+1), index_lateral(k+1,j);
                   index_lateral(k+1,j), index_lateral(k,j+1), index_lateral(k+1,j+1)];
            face3D = [face3D; tri];
            n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
            n2 = face_normal(point3D, tri(2,1), tri(2,2), tri(2,3));
            normal3D = [normal3D, n1, n2];
        end
    end

    % lateral surface for each cut
    for j = 1:num_cut
        index_b = [index(j,1):1:index(j,2), index(j,1)];
        index_t = index_b + sum(len);
        fill_shape = randi(3);

        if fill_shape == 1
            for k = 1:len(j,1)
                tri = [index_b(k), index_b(k+1), index_t(k);
                       index_t(k), index_b(k+1), index_t(k+1)];
                face3D = [face3D; tri];
                n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
                n2 = face_normal(point3D, tri(2,1), tri(2,2), tri(2,3));
                normal3D = [normal3D, n1, n2];
            end

        elseif fill_shape == 2
            index_b = index(j,1):1:index(j,2);
            index_t = index_b + sum(len);
            ratio = rand(1)*0.5+0.4;
            t_offset = [tran(j,:),0]';
            point3D(:,index_b) = ratio*(point3D(:,index_b)-t_offset)+t_offset;
            z_internal = (rand(1)*0.5+0.4) * z_lateral;
            ratio_xy = randsrc(1,1, [1,rand(1)*1/3+2/3; 1/2,1/2]);
            point_new_b = ratio_xy*(point3D(:,index_b)-t_offset)+t_offset;
            point_new_b(3,:) = z_internal;
            point_new_t = ratio_xy*(point3D(:,index_t)-t_offset)+t_offset;
            point_new_t(3,:) = z_internal;
            index_b = [index(j,1):1:index(j,2), index(j,1)];
            index_t = index_b + sum(len);
            index_b_new = [size(point3D,2)+1:1:size(point3D,2)+len(j,1), size(point3D,2)+1];
            index_t_new = [size(point3D,2)+len(j,1)+1:1:size(point3D,2)+2*len(j,1), size(point3D,2)+len(j,1)+1];
            point3D = [point3D, point_new_b, point_new_t];
            index_internal = [index_b; index_b_new; index_t_new; index_t];
            for ll = 1:len(j,1)
                for k = 1:3
                    tri = [index_internal(k,ll), index_internal(k,ll+1), index_internal(k+1,ll);
                           index_internal(k+1,ll), index_internal(k,ll+1), index_internal(k+1,ll+1)];
                    face3D = [face3D; tri];
                    n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
                    n2 = face_normal(point3D, tri(2,1), tri(2,2), tri(2,3));
                    normal3D = [normal3D, n1, n2];
                end
            end

        else
            index_b = index(j,1):1:index(j,2);
            index_t = index_b + sum(len);
            ratio = rand(1)*0.5+0.4;
            t_offset = [tran(j,:),0]';
            point3D(:,index_b) = ratio*(point3D(:,index_b)-t_offset)+t_offset;
            z_internal = (rand(1)*0.5+0.4) * z_lateral;
            ratio_xy = randsrc(1,1, [1,rand(1)*1/3+2/3; 1/2,1/2]);
            point_new_b = ratio_xy*(point3D(:,index_b)-t_offset)+t_offset;
            point_new_b(3,:) = z_internal;
            point_new_b = [point_new_b, [tran(j,:),z_internal]'];
            point_new_t = ratio_xy*(point3D(:,index_t)-t_offset)+t_offset;
            point_new_t(3,:) = point_new_t(3,:)+z_internal;
            point_new_t = [point_new_t, [tran(j,:),point_new_t(3,1)]'];
            n_b = len(j,1);
            idx_b_ring = [index(j,1):1:index(j,2), index(j,1)];
            idx_b_mid = [size(point3D,2)+1:1:size(point3D,2)+n_b, size(point3D,2)+1];
            idx_b_tip = size(point3D,2)+n_b+1;
            idx_t_ring = idx_b_ring + sum(len);
            idx_t_mid = [size(point3D,2)+n_b+2:1:size(point3D,2)+2*n_b+1, size(point3D,2)+n_b+2];
            idx_t_tip = size(point3D,2)+2*n_b+2;
            point3D = [point3D, point_new_b, point_new_t];

            for ll = 1:n_b
                % bottom ring to mid
                tri = [idx_b_ring(ll), idx_b_ring(ll+1), idx_b_mid(ll);
                       idx_b_mid(ll), idx_b_ring(ll+1), idx_b_mid(ll+1)];
                face3D = [face3D; tri];
                n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
                n2 = face_normal(point3D, tri(2,1), tri(2,2), tri(2,3));
                normal3D = [normal3D, -n1, -n2];

                % top ring to mid
                tri = [idx_t_ring(ll), idx_t_ring(ll+1), idx_t_mid(ll);
                       idx_t_mid(ll), idx_t_ring(ll+1), idx_t_mid(ll+1)];
                face3D = [face3D; tri];
                n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
                n2 = face_normal(point3D, tri(2,1), tri(2,2), tri(2,3));
                normal3D = [normal3D, n1, n2];

                % bottom mid to tip
                tri = [idx_b_mid(ll), idx_b_mid(ll+1), idx_b_tip];
                face3D = [face3D; tri];
                n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
                normal3D = [normal3D, -n1];

                % top mid to tip
                tri = [idx_t_mid(ll), idx_t_mid(ll+1), idx_t_tip];
                face3D = [face3D; tri];
                n1 = face_normal(point3D, tri(1,1), tri(1,2), tri(1,3));
                normal3D = [normal3D, n1];
            end
        end
    end

    % render
    figure()
    hold on;
    body_color = 0.2 + 0.6 * rand(1,3);
    p = patch('Faces', face3D, 'Vertices', point3D');
    p.FaceColor = body_color;
    p.EdgeColor = 'none';
    p.FaceLighting = 'gouraud';
    p.AmbientStrength = 0.4 + 0.2*rand();
    p.DiffuseStrength = 0.6 + 0.2*rand();
    p.SpecularStrength = 0.05 + 0.15*rand();
    p.SpecularExponent = 10 + 20*rand();
    p.BackFaceLighting = 'unlit';
    light('Position', [1 1 1], 'Style', 'infinite');
    light('Position', [-1 0.5 0.5], 'Style', 'infinite');
    light('Position', [0 -1 1], 'Style', 'infinite');
    axis equal;
    grid off;
    axis off;
    set(gcf, 'Color', 'w');
    set(gca, 'Color', 'w');
    view(90 + rand(1)*40 - 20, 25 + rand(1)*15);
    name = strcat(num2str(i));
    saveas(gcf,strcat(name,'.fig'));
    saveas(gcf,strcat(name,'.jpg'));
    close all;
    name = strcat(num2str(i),'.mat');
    save(name,'point3D','face3D','normal3D');
    toc;
end

function [normal, point] = make_shape(r_min, r_max)
    shape = randi([1,10],1);
    r = rand(1)*(r_max-r_min) + r_min;
    if shape == 1
        [normal, point] = circ(r);
    elseif shape == 2
        a = rand(1)*(r_max-r_min) + r_min;
        b = rand(1)*(r_max-r_min) + r_min;
        [normal, point] = elli(a, b);
    else
        [normal, point] = poly(r);
    end
end

function index = collision_ie_check(inter, point)
    index = false;
    if min(point(1,:)) >= inter(1) || max(point(1,:)) <= inter(2) || ...
       min(point(2,:)) >= inter(3) || max(point(2,:)) <= inter(4)
        index = true;
        return;
    end
    n = size(point,2);
    Pt = [point, point(:,1)];
    corners = [inter(1),inter(3); inter(2),inter(3); inter(2),inter(4); inter(1),inter(4); inter(1),inter(3)]';
    for k = 1:n
        for kk = 1:4
            A = [corners(:,kk+1)-corners(:,kk), Pt(:,k)-Pt(:,k+1)];
            b_vec = Pt(:,k)-corners(:,kk);
            t = A\b_vec;
            if t(1)>=0 && t(1)<=1 && t(2)>=0 && t(2)<=1
                index = true;
                return;
            end
        end
    end
end

function index = collision_check(BB, x_tran, y_tran, j)
    BB(j,1:2) = BB(j,1:2) + x_tran;
    BB(j,3:4) = BB(j,3:4) + y_tran;
    index = 0;
    for k = 1:j-1
        if ~(BB(j,1) > BB(k,2) || BB(j,2) < BB(k,1) || BB(j,3) > BB(k,4) || BB(j,4) < BB(k,3))
            index = 1;
            return;
        end
    end
end

function bb = bound_box(point)
    bb = [min(point(1,:)), max(point(1,:)), min(point(2,:)), max(point(2,:))];
end

function [norm, POINT] = poly(r)
    num_v = randi([3,10],1);
    point = zeros(2,num_v+1);
    theta = 0:2*pi/num_v:2*pi*(num_v-1)/num_v;
    for j = 1:num_v
        point(1,j) = r * cos(theta(1,j));
        point(2,j) = r * sin(theta(1,j));
    end
    point(:,num_v+1) = point(:,1);

    Point = zeros(2,3*num_v);
    for j = 1:num_v
        kesi = rand(1)*0.5;
        lamda = rand(1)*0.5+0.5;
        Point(:,3*j-2) = point(:,j);
        Point(:,3*j-1) = point(:,j) + kesi*(point(:,j+1)-point(:,j));
        Point(:,3*j) = point(:,j) + lamda*(point(:,j+1)-point(:,j));
    end
    Point = [Point(:,end), Point(:,1:end-1)];

    POINT = [];
    cham = randsrc(1,1, [1,2; 4/5,1/5]);
    for j = 1:num_v
        if cham == 1
            POINT = [POINT, Point(:,3*j-2:3*j)];
        else
            POINT = [POINT, cham_bezier(Point(:,3*j-2:3*j))];
        end
    end
    norm = norm_cal(POINT, 1);
end

function POINT_new = cham_bezier(Point)
    u = 0:0.05:1;
    POINT_new = Point(:,1)*(1-u).^2 + Point(:,2)*2*(1-u).*u + Point(:,3)*u.^2;
end

function [norm, point] = circ(r)
    num = 36;
    point = zeros(2, num);
    for i = 1:num
        ang = 10*i*(pi/180);
        point(1,i) = r*cos(ang);
        point(2,i) = r*sin(ang);
    end
    norm = norm_cal(point, 1);
end

function [norm, point] = elli(a, b)
    num = 36;
    point = zeros(2, num);
    for i = 1:num
        ang = 10*i*(pi/180);
        point(1,i) = a*cos(ang);
        point(2,i) = b*sin(ang);
    end
    norm = norm_cal(point, 1);
end

function normal = norm_cal(point, FLAG)
    len = size(point,2);
    normal = zeros(2,len);
    for i = 1:len
        if i == len
            dir = (point(:,1)-point(:,len)) / norm(point(:,1)-point(:,len));
        else
            dir = (point(:,i+1)-point(:,i)) / norm(point(:,i+1)-point(:,i));
        end
        normal(:,i) = [0 1 ; -1 0] * dir;
    end
    if FLAG == 2
        normal = -normal;
    end
end

function nm = face_normal(p3D, v1, v2, v3)
    nm = cross(p3D(:,v2)-p3D(:,v1), p3D(:,v3)-p3D(:,v1));
    n = norm(nm);
    if n > eps
        nm = nm / n;
    end
end
