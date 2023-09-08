import socket
import pickle
from pynput import keyboard
#------------------------------------
def on_key_release(key):
    if key == keyboard.Key.esc:  # 如果按下的是 "esc" 键
        print('Exiting loop...')
        return False  # 返回 False 来终止监听循环
# 创建监听器
listener = keyboard.Listener(on_release=on_key_release)
# 开始监听
listener.start()
#------------------------------------
# 创建一个TCP/IP Socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定IP地址和端口号
pc_ip = '192.168.1.99'
pc_port = 12345
server.bind((pc_ip, pc_port))
# 监听来自Jetson的连接
server.listen()
# 等待连接并接收数据
conn, addr = server.accept()
print('Connected by', addr)
#------------------------------------
import pyautogui
import time

pyautogui.FAILSAFE = False
ScreenWidth, ScreenHeight = pyautogui.size()
Cur_x, Cur_y = 0,0
Pre_x, Pre_y = 0,0
Delay_L, Delay_R = 0,0
Delay_U, Delay_D = 0,0
MouseSensitivity, ScrollSensitivity = 224, 1#for adjust
Relative_X, Relative_Y = 0,0
Ture_Relative_X, Ture_Relative_Y = 0,0
text = "startpoint"

def control_cursor(data):
    
    global ScreenWidth, ScreenHeight
    global Cur_x, Cur_y
    global Pre_x, Pre_y 
    global Delay_L, Delay_R 
    global Delay_U, Delay_D 
    global MouseSensitivity, ScrollSensitivity 
    global Relative_X, Relative_Y
    global Ture_Relative_X, Ture_Relative_Y
    global text
    
    text = data[0]
    joints_pos_x = data[1]
    joints_pos_y = data[2]
    Cur_x = int((joints_pos_x)*ScreenWidth/MouseSensitivity)
    Cur_y = int((joints_pos_y)*ScreenHeight/MouseSensitivity)
    Relative_X = Cur_x - Pre_x#这里Relative_X是个烦的 --nis
    Relative_Y = Cur_y - Pre_y
    Pre_x = Cur_x
    Pre_y = Cur_y
        
    if text == "pan":
        Delay_L, Delay_R = 0,0
        Delay_U, Delay_D = 0,0

        if abs(Relative_X)<102 and abs(Relative_Y)<53:#hpf
            if abs(Relative_X)>10 or abs(Relative_Y)>5:#lpf
                Ture_Relative_X = -Relative_X 
                Ture_Relative_Y = Relative_Y
                pyautogui.move(Ture_Relative_X, Ture_Relative_Y)#'-' is for revision
    elif text == "fist":
        Delay_L += 1
        if Delay_L > 15:
            pyautogui.click(duration=0.3)
            Delay_L = 0
    elif text == "palm":
        Delay_R += 1
        if Delay_R > 15:
            pyautogui.click(button="right",duration=0.3)
            Delay_R = 0
    elif text == "thumb_up":
        Delay_U += 1
        if Delay_U > 10:
            pyautogui.scroll(200)
            Delay_U = 0
    elif text == "ok":
        Delay_D += 1
        if Delay_D > 10:
            pyautogui.scroll(-200)
            Delay_D = 0
    
                    
print("cursor control has been set --nis\n")
#------------------------------------
output_filename = 'E:\File pack\Study file\嵌入式\基于NVIDIA JETSON的人体姿态识别\cursor_pos_data.txt'  # 输出文件名
data_count = 10000  # 要采集的数据组数
with open(output_filename, 'w') as output_file:
    
    Origin_X,Origin_Y = pyautogui.position()
    output_file.write(f'{text},x={Origin_X},y={Origin_Y}\n')

    while True:
        serialized_data = conn.recv(1024)  # 接收数据,chatgpt这里不准确
        data = pickle.loads(serialized_data)  # 反序列化数据，得到原始的列表

        if not data:
            break
        print('Received:', data)
        
        control_cursor(data)    
        
        if data_count > 0:
            output_file.write(f'{text},x={Relative_X},y={Relative_Y}\n')    
            data_count -= 1
        
        if not listener.running:  # 如果监听器已经停止
            break

# 关闭连接
conn.close()
output_file.close()
print(f'数据已采集并存储在 {output_filename} 中。')




    


