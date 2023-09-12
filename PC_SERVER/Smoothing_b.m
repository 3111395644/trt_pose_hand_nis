clc,clear
% 打开 txt 文件
fileID = fopen('C:\Users\13331\OneDrive\cursor_pos_data.txt', 'r');

% 初始化一个空的坐标数组
coordinates = [];

% 初始位置
first_line = fgetl(fileID);
parts = strsplit(first_line, ',');  % 使用逗号分割字符串
x_str = strrep(parts{2}, 'x=', '');  % 提取 x 值的字符串部分
y_str = strrep(parts{3}, 'y=', '');  % 提取 y 值的字符串部分    
X0 = str2double(x_str);  % 将 x 值的字符串转换为数值类型
Y0 = str2double(y_str);  % 将 y 值的字符串转换为数值类型


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

%%二次指数平滑处理
% for j=1:10:length(coordinates)
%     if j+9<length(coordinates)
%         data_unprocessed=coordinates(j:(j+9),:);
%         lenD=length(data_unprocessed);
%         a=0.9;
%         st1(1, :)=data_unprocessed(1, :);
%         st2(2, :)=data_unprocessed(1, :);
% 
%         for i=2:lenD
%         st1(i, :)=a*data_unprocessed(i, :)+(1-a).*st1(i-1, :);
%         st2(i, :)=a*st1(i, :)+(1-a).*st2(i-1, :);
%         end
% 
%         b1=2*st1-st2;
%         b2=a/(1-a)*(st1-st2);
%         coordinates_processed(j:(j+9),:)=b1+b2;
%     end
% end

absolute_moves = zeros(size(coordinates));
x0 = X0;
y0 = Y0;
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
%二次指数平滑处理
data_unprocessed=absolute_moves;
lenD=length(data_unprocessed);
a=0.2;
st1(1, :)=data_unprocessed(1, :);
st2(1, :)=data_unprocessed(1, :);


for i=2:lenD
st1(i, :)=a*data_unprocessed(i, :)+(1-a).*st1(i-1, :);
st2(i, :)=a*st1(i, :)+(1-a).*st2(i-1, :);
end

b1=2*st1-st2;
b2=a/(1-a)*(st1-st2);
absolute_moves_processed=round(b1+b2);
% % 初始化数组来存储绝对移动值
% absolute_moves_processed = zeros(size(coordinates_processed));
% % 逐步累加相对移动值以获得绝对坐标
% x0 = X0;
% y0 = Y0;
% for i = 1:size(coordinates_processed, 1)
%     x_relative = coordinates_processed(i, 1);
%     y_relative = coordinates_processed(i, 2);
%     
%     x_absolute = x0 + x_relative;
%     y_absolute = y0 + y_relative;
%     
%     absolute_moves_processed(i, :) = [x_absolute, y_absolute];
%     
%     % 更新初始位置
%     x0 = x_absolute;
%     y0 = y_absolute;
% end
% 绘制还原后的绝对移动值轨迹图
figure;
hold on;
plot(absolute_moves(:, 1), absolute_moves(:, 2) ,'-.r');
text(absolute_moves(1, 1),absolute_moves(1, 2),'start','color','r');
text(absolute_moves(end, 1),absolute_moves(end, 2),'end','color','r');
plot(absolute_moves_processed(:, 1), absolute_moves_processed(:, 2) ,'-.g');
text(absolute_moves_processed(1, 1),absolute_moves_processed(1, 2),'start','color','g');
text(absolute_moves_processed(end, 1),absolute_moves_processed(end, 2),'end','color','g');
hold off;


set(gca,'xlim',[0,1920]);%y坐标轴范围
set(gca,'ylim',[0,1080]);%x坐标轴范围
set(gca, 'YDir', 'reverse');
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin');
title('光标移动轨迹图');
xlabel('水平轴');
ylabel('垂直轴');
axis equal;



