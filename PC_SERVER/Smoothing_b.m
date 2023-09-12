% 打开 txt 文件
fileID = fopen('E:\File pack\Study file\嵌入式\基于NVIDIA JETSON的人体姿态识别\cursor_pos_data.txt', 'r');

% 初始化一个空的坐标数组
coordinates = [];

% 初始位置
first_line = fgetl(fileID);
parts = strsplit(first_line, ',');  % 使用逗号分割字符串
x_str = strrep(parts{2}, 'x=', '');  % 提取 x 值的字符串部分
y_str = strrep(parts{3}, 'y=', '');  % 提取 y 值的字符串部分    
x0 = str2double(x_str);  % 将 x 值的字符串转换为数值类型
y0 = str2double(y_str);  % 将 y 值的字符串转换为数值类型


% 逐行读取 txt 文件内容
while ~feof(fileID)
    line = fgetl(fileID);  % 读取一行
    parts = strsplit(line, ',');  % 使用逗号分割字符串
    text1 = parts{1};
    text2 = 'pan';
    if strcmp(text1, text2)
        x_str = strrep(parts{2}, 'x=', '');  % 提取 x 值的字符串部分
        y_str = strrep(parts{3}, 'y=', '');  % 提取 y 值的字符串部分
        x = str2double(x_str);  % 将 x 值的字符串转换为数值类型
        y = str2double(y_str);  % 将 y 值的字符串转换为数值类型
        if abs(x)<102 && abs(y)<53
            if abs(x)>10 || abs(y)>5
            coordinates = [coordinates; -x, y];  % 添加坐标到数组中
            end
        end
    end
end
% 关闭文件
fclose(fileID); 

% 初始化数组来存储绝对移动值
absolute_moves = zeros(size(coordinates));

% 逐步累加相对移动值以获得绝对坐标
for i = 1:size(coordinates, 1)
    x_relative = coordinates(i, 1);
    y_relative = coordinates(i, 2);
    
    x_absolute = x0 + x_relative;
    y_absolute = y0 + y_relative;
    
    absolute_moves(i, :) = [x_absolute, y_absolute];
    
    % 更新初始位置
    x0 = x_absolute;
    y0 = y_absolute;
end

% 绘制还原后的绝对移动值轨迹图
figure;
plot(absolute_moves(:, 1), absolute_moves(:, 2), '-.');
set(gca,'xlim',[0,1920]);%y坐标轴范围
set(gca,'ylim',[0,1080]);%x坐标轴范围
set(gca, 'YDir', 'reverse');
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
title('还原后的绝对移动值轨迹图');
xlabel('X 坐标');
ylabel('Y 坐标');
axis equal;
grid on;



