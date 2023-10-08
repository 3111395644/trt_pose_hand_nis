#------------------------------------
import cv2
import time

cap = cv2.VideoCapture(0)
#------------------------------------
import pyautogui

pyautogui.FAILSAFE = False
ScreenWidth, ScreenHeight = pyautogui.size()
Cur_x, Cur_y = 0,0
Pre_x, Pre_y = 0,0
Delay_L, Delay_R = 0,0
Delay_U, Delay_D = 0,0
CapSensitivity = 224#for adjust
Relative_X, Relative_Y = 0,0
text = "startpoint"
CPOS_x, CPOS_y = ScreenWidth/2, ScreenHeight/2
ST1_x, ST1_y = CPOS_x, CPOS_y
ST2_x, ST2_y = CPOS_x, CPOS_y
CPOS_x_processed, CPOS_y_processed = 0,0

def control_cursor():
    
    global ScreenWidth, ScreenHeight
    global Cur_x, Cur_y
    global Pre_x, Pre_y 
    global Delay_L, Delay_R 
    global Delay_U, Delay_D 
    global CapSensitivity, MotorSensitivity_PHI, MotorSensitivity_THET 
    global Relative_X, Relative_Y
    global text
    global CPOS_x, CPOS_y
    global ST1_x, ST1_y, ST2_x, ST2_y
    global CPOS_x_processed, CPOS_y_processed
    global listener
    
    pyautogui.moveTo(ScreenWidth/2, ScreenHeight/2)
    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (224, 224))
        img = cv2.flip(img, 1)
        cv2.imshow("capture", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        start = time.perf_counter()
        img_encode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1]# 压缩图像      
        bytedata = img_encode.tostring()# 转换为字节流
        flag_data = (str(len(bytedata))).encode() + ",".encode() + " ".encode()# 标志数据，包括待发送的字节流长度等数据，用‘,’隔开
        conn.send(flag_data)
        data = conn.recv(1024)# 接收服务端的应答
        if ("ok" == data.decode()):
            conn.send(bytedata)# 服务端已经收到标志数据，开始发送图像字节流数据
        data = conn.recv(1024) # 接收服务端的应答
        if ("ok" == data.decode()):
            print("延时：" + str(int((time.perf_counter() - start) * 1000)) + "ms")# 计算发送完成的延时

        serialized_data = conn.recv(1024)  # 接收数据,chatgpt这里不准确
        data = pickle.loads(serialized_data)  # 反序列化数据，得到原始的列表
        if not data:
            break
        print('Received:', data)
        text = data[0]
        joints_pos_x, joints_pos_y = data[1], data[2]

        Cur_x = int((joints_pos_x)/CapSensitivity*ScreenWidth)
        Cur_y = int((joints_pos_y)/CapSensitivity*ScreenHeight)
        Relative_X = Cur_x - Pre_x
        Relative_Y = Cur_y - Pre_y
        Pre_x = Cur_x
        Pre_y = Cur_y
            
        if text == "palm":
            Delay_L, Delay_R = 0,0
            Delay_U, Delay_D = 0,0

            if abs(Relative_X)<200 and abs(Relative_Y)<100:#hpf
                if abs(Relative_X)>10 or abs(Relative_Y)>5:#lpf

                    #开始指数平滑
                    if 0<(CPOS_x + Relative_X)<ScreenWidth:
                        CPOS_x += Relative_X
                    if 0<(CPOS_y + Relative_Y)<ScreenHeight:
                        CPOS_y += Relative_Y
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
        elif text == "pan":
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
import paramiko
# 创建SSH对象
ssh = paramiko.SSHClient()
# 允许连接不在know_hosts文件中的主机
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 连接服务器
ssh.connect(hostname='192.168.43.125', port=22, username='xigong', password='271445')
# 执行命令
stdin, stdout, stderr = ssh.exec_command('python /home/xigong/trt_pose/tasks/hand_pose_new/9_vm_on_pad_client.py') 
#------------------------------------
import socket
import pickle
# 创建一个TCP/IP Socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定IP地址和端口号
pc_ip = '192.168.43.143'
pc_port = 12345
server.bind((pc_ip, pc_port))
# 监听来自Jetson的连接
server.listen()
print('Waiting for connection')
# 等待连接并接收数据
conn, addr = server.accept()
print('Connected by', addr)
#------------------------------------
import pickle
import threading

#camera_cap_thread = threading.Thread(target=camera_cap)
control_cursor_thread = threading.Thread(target=control_cursor)
#control_cursor_thread.daemon = True

#camera_cap_thread.start()
control_cursor_thread.start()    
# 关闭各种连接
#ssh.close()
#conn.close()