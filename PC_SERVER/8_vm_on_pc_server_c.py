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
import paramiko,time
 
# 创建SSH对象
ssh = paramiko.SSHClient()
 
# 允许连接不在know_hosts文件中的主机
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 
# 连接服务器
ssh.connect(hostname='192.168.1.100', port=22, username='xigong', password='271445')
 
 
# 执行命令
stdin, stdout, stderr = ssh.exec_command('export DISPLAY=192.168.1.99:0.0 && python /home/xigong/trt_pose/tasks/hand_pose_new/8_vm_on_pc_client_c.py')
 
# 获取命令结果
#result = stdout.read().decode('utf8')
#print(result)  # 如果有输出的话
 
#------------------------------------
# 创建一个TCP/IP Socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定IP地址和端口号
pc_ip = '192.168.1.99'
pc_port = 12345
server.bind((pc_ip, pc_port))
# 监听来自Jetson的连接
server.listen()
print('Waiting for connection')
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
CPOS_x, CPOS_y = pyautogui.position()
ST1_x, ST1_y = CPOS_x, CPOS_y
ST2_x, ST2_y = CPOS_x, CPOS_y
CPOS_x_processed, CPOS_y_processed = 0,0

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
    global CPOS_x, CPOS_y
    global ST1_x, ST1_y, ST2_x, ST2_y
    global CPOS_x_processed, CPOS_y_processed
    
    text = data[0]
    joints_pos_x = data[1]
    joints_pos_y = data[2]
    Cur_x = int((joints_pos_x)/MouseSensitivity*ScreenWidth)
    Cur_y = int((joints_pos_y)/MouseSensitivity*ScreenHeight)
    Relative_X = Cur_x - Pre_x#这里Relative_X的方向是反向的 --nis
    Relative_Y = Cur_y - Pre_y
    Pre_x = Cur_x
    Pre_y = Cur_y
        
    if text == "pan":
        Delay_L, Delay_R = 0,0
        Delay_U, Delay_D = 0,0

        if abs(Relative_X)<102 and abs(Relative_Y)<53:#hpf
            if abs(Relative_X)>10 or abs(Relative_Y)>5:#lpf
                Ture_Relative_X = -Relative_X#'-' is for revision --nis 
                Ture_Relative_Y = Relative_Y
                #开始指数平滑
                if 0<(CPOS_x + Ture_Relative_X)<ScreenWidth:
                    CPOS_x += Ture_Relative_X
                if 0<(CPOS_y + Ture_Relative_Y)<ScreenHeight:
                    CPOS_y += Ture_Relative_Y
                a = 0.2#平滑系数
                ST1_x = int(a*CPOS_x + (1-a)*ST1_x)
                ST1_y = int(a*CPOS_y + (1-a)*ST1_y)
                ST2_x = int(a*ST1_x + (1-a)*ST2_x)
                ST2_y = int(a*ST1_y + (1-a)*ST2_y)
                CPOS_x_processed = int(2*ST1_x - ST2_x + a/(1-a)*(ST1_x - ST2_x))
                if CPOS_x_processed > ScreenWidth:
                    CPOS_x_processed = ScreenWidth
                elif CPOS_x_processed < 0:
                    CPOS_x_processed = 0
                CPOS_y_processed = int(2*ST1_y - ST2_y + a/(1-a)*(ST1_y - ST2_y))
                if CPOS_y_processed > ScreenHeight:
                    CPOS_y_processed = ScreenHeight
                elif CPOS_y_processed < 0:
                    CPOS_y_processed = 0#平滑完毕
                pyautogui.moveTo(CPOS_x_processed, CPOS_y_processed)
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
        if Delay_U > 15:
            pyautogui.scroll(200)
            Delay_U = 0
    elif text == "ok":
        Delay_D += 1
        if Delay_D > 15:
            pyautogui.scroll(-200)
            Delay_D = 0
    
                    
print("cursor control has been set --nis\n")
#------------------------------------ 
while True:
    serialized_data = conn.recv(1024)  # 接收数据,chatgpt这里不准确
    data = pickle.loads(serialized_data)  # 反序列化数据，得到原始的列表

    if not data:
        break
    print('Received:', data)
        
    control_cursor(data)    
        
    if not listener.running:  # 如果监听器已经停止
        break

# 关闭各种连接
ssh.close()
conn.close()




    


