clear all;
close all;
clc;

min_point = 3;
max_point = 13;
x_avg = 0.5;

for i = 1001:2000
    i
    mach = select_machined_faces();

    if ~any(mach)
        [point3D, face3D, normal3D] = generate_block(x_avg);
    else
        [profile_F] = generate_profile(x_avg, min_point, max_point);

        if mach(2)
            [profile_B] = generate_profile(x_avg, min_point, max_point);
        else
            profile_B = profile_F;
        end

        [point3D, face3D, normal3D] = build_mesh(profile_F, profile_B, mach, x_avg);
    end

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
end

function [profile] = generate_profile(x_avg, min_point, max_point)
    point = [];
    point(1,:) = [0  0];
    Point = point(end,:);
    j = randsrc(1,1, [1,2,3; 1/2,1/3,1/6]);
    if j == 1
        point_new = VU(Point, x_avg);
        FLAG = 2;
    elseif j == 2
        point_new = LRU(Point, x_avg);
        FLAG = 4;
    else
        point_new = CRU(Point, x_avg);
        FLAG = 6;
    end
    point = [point ; point_new];
    Point = point(end,:);
    clear point_new;

    num = randi(max_point-min_point)+min_point-2;
    for k = 1:num
        len = length(point);
        j = randsrc(1,1, [1,2,3,4,5,6,7; 2/3,1/12,1/12,1/18,1/18,1/36,1/36]);
        while (FLAG==j || (FLAG==2&&j==3) || (FLAG==3&&j==2))
            j = randi(7);
        end
        if j == 1
            point_new = HR(Point, x_avg);
            FLAG = 1;
        elseif j == 2
            point_new = VU(Point, x_avg);
            FLAG = 2;
        elseif j == 3
            point_new = VD(Point, x_avg);
            FLAG = 3;
        elseif j == 4
            point_new = LRU(Point, x_avg);
            FLAG = 4;
        elseif j == 5
            point_new = LRD(Point, x_avg);
            FLAG = 5;
        elseif j == 6
            point_new = CRU(Point, x_avg);
            FLAG = 6;
        else
            point_new = CRD(Point, x_avg);
            FLAG = 7;
        end
        point = [point ; point_new];
        Point = point(end,:);
        clear point_new
    end

    len = length(point);
    if FLAG == 2 || FLAG == 3
        j = randsrc(1,1, [1,2 ; 2/3,1/3])+1;
    else
        j = randsrc(1,1, [1,2,3 ; 1/2,1/3,1/6]);
    end

    if j == 1
        point(len+1,1) = point(len,1);
        point(len+1,2) = 0;
    elseif j == 2
        point_new = LRD(Point, x_avg);
        point_new(1,2) = 0;
        point = [point ; point_new];
    else
        point_new = CRD(Point, x_avg);
        point_new(end,2) = 0;
        point = [point ; point_new];
    end
    clear Point;

    z_max = max(point(:,2));
    z_m = 2^(rand(1)*4-3)*z_max;
    point(:,2) = point(:,2)+z_m;
    x_cood = flip(unique(point(:,1)));
    point_new = [x_cood , zeros(size(x_cood,1),1)];
    point = [point ; point_new];

    ratio_xz = 0.3*rand(1)+0.1;
    point(:,1) = point(:,1) * (max(point(:,2))/max(point(:,1))) / ratio_xz;

    if ispolycw(point(:,1), point(:,2))
        point = flipud(point);
    end

    profile = point;
end

function [profile_out] = resample_profile(profile, N_target)
    N = size(profile, 1);
    if N == N_target
        profile_out = profile;
        return;
    end

    dx = diff(profile(:,1));
    dz = diff(profile(:,2));
    seg_lengths = sqrt(dx.^2 + dz.^2);
    cum_len = [0; cumsum(seg_lengths)];
    total_len = cum_len(end);

    sample_t = linspace(0, 1, N_target+1);
    sample_t(end) = [];
    sample_len = sample_t * total_len;

    profile_out = zeros(N_target, 2);
    for i = 1:N_target
        idx = find(cum_len >= sample_len(i), 1, 'first');
        if idx == 1
            profile_out(i,:) = profile(1,:);
        else
            t = (sample_len(i) - cum_len(idx-1)) / (cum_len(idx) - cum_len(idx-1));
            profile_out(i,:) = (1-t) * profile(idx-1,:) + t * profile(idx,:);
        end
    end
end

function [point3D, face3D, normal3D] = generate_block(x_avg)
    x_w = 2^(2*rand(1)-1) * x_avg * 5;
    z_h = 2^(2*rand(1)-1) * x_avg * 5;
    y_d = 2^(2*rand(1)-1) * x_w;

    point3D = [0  x_w  x_w  0   0   x_w  x_w  0;
               0  0    y_d  y_d 0   0    y_d  y_d;
               0  0    0    0   z_h z_h  z_h  z_h];

    face3D = [1 2 3; 1 3 4; 5 7 6; 5 8 7;
              1 5 6; 1 6 2; 2 6 7; 2 7 3;
              3 7 8; 3 8 4; 4 8 5; 4 5 1];

    normal3D = compute_normals(point3D, face3D);
end

function mach = select_machined_faces()
    mach = false(1,6);
    r = rand();
    if r < 0.005
        return;
    end
    k = randi(6);
    mach(1) = true;
    if k == 1
        return;
    end
    pool = [2 3 4 5 6];
    chosen = pool(randperm(5, k-1));
    mach(chosen) = true;
end

function edge_labels = classify_edges(profile, x_avg)
    N = size(profile, 1);
    x_max = max(profile(:,1));
    z_max = max(profile(:,2));
    tol_x = 0.08 * max(x_max, x_avg);
    tol_z = 0.08 * max(z_max, x_avg);

    edge_labels = 5 * ones(1, N);
    for k = 1:N
        k_next = mod(k, N) + 1;
        mx = (profile(k,1) + profile(k_next,1)) / 2;
        mz = (profile(k,2) + profile(k_next,2)) / 2;

        if mz < tol_z
            edge_labels(k) = 1;
        elseif mz > z_max - tol_z
            edge_labels(k) = 2;
        elseif mx < tol_x
            edge_labels(k) = 3;
        elseif mx > x_max - tol_x
            edge_labels(k) = 4;
        end
    end
end

function [point3D, face3D, normal3D] = build_mesh(profile_F, profile_B, mach, x_avg)
    N_v = max(size(profile_F,1), size(profile_B,1));
    N_v = max(N_v, 20);
    profile_F = resample_profile(profile_F, N_v);
    profile_B = resample_profile(profile_B, N_v);

    pgon_F = polyshape(profile_F(:,1), profile_F(:,2),'SolidBoundaryOrientation','ccw');
    T_F = triangulation(pgon_F);
    V_F = T_F.Points;
    F_F = T_F.ConnectivityList;
    N_v_F = size(V_F, 1);

    pgon_B = polyshape(profile_B(:,1), profile_B(:,2),'SolidBoundaryOrientation','ccw');
    T_B = triangulation(pgon_B);
    V_B = T_B.Points;
    F_B = T_B.ConnectivityList;
    N_v_B = size(V_B, 1);

    y_dist = 2^(2*rand(1)-1)*max(V_F(:,1));

    N_v_use = N_v_F;

    front3D = zeros(3, N_v_use);
    front3D(1,:) = V_F(:,1)';
    front3D(3,:) = V_F(:,2)';

    back3D = zeros(3, N_v_use);
    if N_v_B == N_v_use
        back3D(1,:) = V_B(:,1)';
        back3D(2,:) = y_dist;
        back3D(3,:) = V_B(:,2)';
    else
        back3D(1,:) = V_F(:,1)';
        back3D(2,:) = y_dist;
        back3D(3,:) = V_F(:,2)';
    end

    point3D = [front3D, back3D];
    face3D = [F_F; F_F + N_v_use];

    edge_labels = classify_edges(profile_F, x_avg);

    point_list1 = [1:1:N_v_use, 1];
    point_list2 = point_list1 + N_v_use;

    for k = 1:N_v_use
        k_next = k + 1;
        f1 = point_list1(k);
        f2 = point_list1(k_next);
        b1 = point_list2(k);
        b2 = point_list2(k_next);

        face_id = edge_labels(k);
        edge_to_mach = [4, 3, 5, 6];
        is_machined = false;
        if face_id >= 1 && face_id <= 4
            is_machined = mach(edge_to_mach(face_id));
        end

        if ~is_machined
            face_new = [f1 f2 b1; f2 b2 b1];
            face3D = [face3D; face_new];
        else
            N_sub = randi([5, 15]);
            n_new = 2 * (N_sub - 1);
            new_verts = zeros(3, n_new);

            for r = 1:N_sub-1
                t = r / N_sub;
                new_verts(:, 2*(r-1)+1) = (1-t)*point3D(:,f1) + t*point3D(:,b1);
                new_verts(:, 2*(r-1)+2) = (1-t)*point3D(:,f2) + t*point3D(:,b2);
            end

            all_x = [point3D(1,f1), point3D(1,f2), point3D(1,b1), point3D(1,b2), new_verts(1,:)];
            all_z = [point3D(3,f1), point3D(3,f2), point3D(3,b1), point3D(3,b2), new_verts(3,:)];
            face_bounds = [min(all_x), max(all_x), min(all_z), max(all_z)];

            edge_dir = point3D(:,f2) - point3D(:,f1);
            extrude_dir = point3D(:,b1) - point3D(:,f1);
            fn = cross(edge_dir, extrude_dir);
            fn_norm = norm(fn);
            if fn_norm > eps
                fn = fn / fn_norm;
            else
                fn = [0;1;0];
            end

            strip_normals = repmat(fn, 1, n_new);

            if face_id == 2
                n_pockets = randi([1, 3]);
                new_verts = apply_pockets(new_verts, strip_normals, face_bounds, n_pockets);
                n_steps = randi([2, 4]);
                new_verts = apply_steps(new_verts, strip_normals, face_bounds, n_steps);
            elseif face_id == 3 || face_id == 4
                n_pockets = randi([0, 2]);
                if n_pockets > 0
                    new_verts = apply_pockets(new_verts, strip_normals, face_bounds, n_pockets);
                end
            elseif face_id == 1
                n_steps = randi([1, 3]);
                new_verts = apply_steps(new_verts, strip_normals, face_bounds, n_steps);
            end

            fw = face_bounds(2) - face_bounds(1);
            fh = face_bounds(4) - face_bounds(3);
            if fw > 0 && fh > 0
                new_verts = apply_tool_marks(new_verts, strip_normals, fw, fh);
            end

            start_idx = size(point3D, 2) + 1;
            point3D = [point3D, new_verts];

            face3D = [face3D; f1, f2, start_idx; f2, start_idx+1, start_idx];

            for r = 1:N_sub-2
                v_rl = start_idx + 2*(r-1);
                v_rr = v_rl + 1;
                v_pl = v_rl + 2;
                v_pr = v_pl + 1;
                face3D = [face3D; v_rl, v_rr, v_pl; v_rr, v_pr, v_pl];
            end

            v_ll = start_idx + 2*(N_sub-2);
            v_lr = v_ll + 1;
            face3D = [face3D; v_ll, v_lr, b1; v_lr, b2, b1];
        end
    end

    face_width_x = max(profile_F(:,1)) - min(profile_F(:,1));
    face_height_z = max(profile_F(:,2)) - min(profile_F(:,2));
    if face_width_x > 0 && face_height_z > 0
        amplitude_fm = (0.003 + 0.01*rand()) * min(face_width_x, face_height_z);
        stepover_fm = (0.05 + 0.10*rand()) * face_width_x;
        theta_fm = 2*pi*rand();
        tool_dir_fm = [cos(theta_fm), 0, sin(theta_fm)];
        for v = 1:N_v_use
            pos = dot(point3D(:,v), tool_dir_fm);
            point3D(2, v) = point3D(2, v) + amplitude_fm * sin(2*pi*pos/stepover_fm);
        end
    end

    normal3D = compute_normals(point3D, face3D);
end

function [verts] = apply_pockets(verts, normals, face_bounds, n_pockets)
    x_range = face_bounds(2) - face_bounds(1);
    z_range = face_bounds(4) - face_bounds(3);
    if x_range <= 0 || z_range <= 0
        return;
    end

    for p = 1:n_pockets
        if rand() < 0.5
            pw = (0.2 + 0.3*rand()) * x_range;
            ph = (0.2 + 0.3*rand()) * z_range;
            pcx = face_bounds(1) + (0.15 + 0.7*rand()) * x_range;
            pcz = face_bounds(3) + (0.15 + 0.7*rand()) * z_range;
            pocket_type = 'rect';
        else
            pr = (0.1 + 0.15*rand()) * min(x_range, z_range);
            pcx = face_bounds(1) + (0.15 + 0.7*rand()) * x_range;
            pcz = face_bounds(3) + (0.15 + 0.7*rand()) * z_range;
            pocket_type = 'circ';
        end
        pd = (0.1 + 0.2*rand()) * min(x_range, z_range);

        for vi = 1:size(verts, 2)
            vx = verts(1, vi);
            vz = verts(3, vi);
            in_pocket = false;
            if strcmp(pocket_type, 'rect')
                if abs(vx - pcx) < pw/2 && abs(vz - pcz) < ph/2
                    in_pocket = true;
                end
            else
                if sqrt((vx - pcx)^2 + (vz - pcz)^2) < pr
                    in_pocket = true;
                end
            end
            if in_pocket
                verts(:, vi) = verts(:, vi) - pd * normals(:, vi);
            end
        end
    end
end

function [verts] = apply_steps(verts, normals, face_bounds, n_steps)
    z_range = face_bounds(4) - face_bounds(3);
    if z_range <= 0
        return;
    end
    band_height = z_range / n_steps;
    band_depths = cumsum(rand(1, n_steps));
    band_depths = band_depths / max(band_depths) * 0.25 * z_range;

    for vi = 1:size(verts, 2)
        vz = verts(3, vi);
        band_idx = floor((vz - face_bounds(3)) / band_height) + 1;
        band_idx = max(1, min(band_idx, n_steps));
        verts(:, vi) = verts(:, vi) - band_depths(band_idx) * normals(:, vi);
    end
end

function [verts] = apply_tool_marks(verts, normals, face_width, face_height)
    amplitude = (0.005 + 0.015*rand()) * min(face_width, face_height);
    stepover = (0.05 + 0.10*rand()) * face_width;
    theta = 2*pi*rand();
    tool_dir = [cos(theta), 0, sin(theta)];

    for vi = 1:size(verts, 2)
        pos = dot(verts(:,vi), tool_dir);
        displacement = amplitude * sin(2*pi*pos/stepover);
        verts(:, vi) = verts(:, vi) + displacement * normals(:, vi);
    end
end

function normals = compute_normals(point3D, face3D)
    n_verts = size(point3D, 2);
    n_faces = size(face3D, 1);
    face_normals = zeros(3, n_faces);

    for i = 1:n_faces
        v1 = point3D(:, face3D(i,1));
        v2 = point3D(:, face3D(i,2));
        v3 = point3D(:, face3D(i,3));
        n = cross(v2-v1, v3-v1);
        nrm = norm(n);
        if nrm > eps
            face_normals(:,i) = n / nrm;
        end
    end

    normals = zeros(3, n_verts);
    for i = 1:n_faces
        for j = 1:3
            normals(:, face3D(i,j)) = normals(:, face3D(i,j)) + face_normals(:,i);
        end
    end
    for i = 1:n_verts
        nrm = norm(normals(:,i));
        if nrm > eps
            normals(:,i) = normals(:,i) / nrm;
        else
            normals(:,i) = [0;1;0];
        end
    end
end

function Point_new = HR(Point, x_avg)
    lamda = 2^(rand(1)*2-1);
    Point_new(:,1) = Point(:,1) + lamda * x_avg;
    Point_new(:,2) = Point(:,2);
end

function Point_new = VU(Point, x_avg)
    lamda = 2^(rand(1)*2-1);
    Point_new(:,1) = Point(:,1);
    Point_new(:,2) = Point(:,2) + lamda * x_avg;
end

function Point_new = VD(Point, x_avg)
    lamda = -2^(rand(1)*2-1);
    Point_new(:,1) = Point(:,1);
    Point_new(:,2) = max(Point(:,2) + lamda * x_avg, Point(:,2)/2);
end

function Point_new = LRU(Point, x_avg)
    lamda1 = 2^(rand(1)*2-1);
    lamda2 = 2^(rand(1)*2-1);
    Point_new(:,1) = Point(:,1) + lamda1 * x_avg;
    Point_new(:,2) = Point(:,2) + lamda2 * x_avg;
end

function Point_new = LRD(Point, x_avg)
    lamda1 = 2^(rand(1)*2-1);
    lamda2 = -2^(rand(1)*2-1);
    Point_new(:,1) = Point(:,1) + lamda1 * x_avg;
    Point_new(:,2) = max(Point(:,2) + lamda2 * x_avg, Point(:,2)/2);
end

function Point_new = CRU(Point, x_avg)
    lamda1 = 2^(rand(1)*2-1);
    lamda2 = 2^(rand(1)*2-1);
    a = lamda1 * x_avg;
    b = lamda2 * x_avg;
    index = randi([1,2],1);
    num_point = 9;
    theta_incr = 10;
    Point_new = zeros(num_point,2);
    if index == 1
        C = Point +  [a, 0];
        for ii = 1:num_point
            theta = (180-theta_incr*ii)/180*pi;
            Point_new(ii,:) = C + [a*cos(theta), b*sin(theta)];
        end
    else
        C = Point + [0, b];
        for ii = 1:num_point
            theta = (270+theta_incr*ii)/180*pi;
            Point_new(ii,:) = C + [a*cos(theta), b*sin(theta)];
        end
    end
end

function Point_new = CRD(Point, x_avg)
    lamda1 = 2^(rand(1)*2-1);
    lamda2 = 2^(rand(1)*2-1);
    a = lamda1 * x_avg;
    b = min(lamda2 * x_avg, Point(1,2)/2);
    index = randi([1,2],1);
    num_point = 9;
    theta_incr = 10;
    Point_new = zeros(num_point,2);
    if index == 1
        C = Point +  [a, 0];
        for ii = 1:num_point
            theta = (180+theta_incr*ii)/180*pi;
            Point_new(ii,:) = C + [a*cos(theta), b*sin(theta)];
        end
    else
        C = Point + [0, -b];
        for ii = 1:num_point
            theta = (90-theta_incr*ii)/180*pi;
            Point_new(ii,:) = C + [a*cos(theta), b*sin(theta)];
        end
    end
end

function [normal] = norm_cal(point, FLAG)
    len = size(point,2);
    normal = zeros(2,len);
    for ii = 1:len
        if ii == len
            dir = (point(:,1)-point(:,len)) / norm(point(:,1)-point(:,len));
            normal(:,ii) = [0 1 ; -1 0] * dir;
        else
            dir = (point(:,ii+1)-point(:,ii)) / norm(point(:,ii+1)-point(:,ii));
            normal(:,ii) = [0 1 ; -1 0] * dir;
        end
    end
    tf = ispolycw(point(:,1), point(:,2));
    if tf == 1
        normal = -normal;
    end
    if FLAG == 0
        normal = -normal;
    end
end
