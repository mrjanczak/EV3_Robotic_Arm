# Robot Arm with 3 calibration buttons, no gyro
# Prerequisites:
# - libusb0.dll - don't use ver. 1.0

import os
import time
from datetime import datetime
import pickle
import numpy as np
import math
from numbers import Number
import copy

from spatialmath import SE3
from spatialmath.base import *
import matplotlib.pyplot as plt
from thread_task import Task

import pyglet
from pyglet.gl import *
from OpenGL.GL import *
from OpenGL.GLU import *

import roboticstoolbox as rtb
import ev3_dc as ev3
import chess
import chess.engine

# np.set_printoptions(sign=' ', formatter={'int': '{: 4n}'.format})
deg = np.pi / 180. # deg > rad
rad = 180. / np.pi
stud = .008  # m/stud
twoPi = 2. * np.pi
stud_6deg = [1/stud,rad,rad,rad, rad,rad,rad]

def vector(*args):
    return (GLfloat * len(args))(*args)

# ********* Window settings *********
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SCALE = 750
OFFSET = np.array([0,0,0])

def draw_cylinder(X1, X2, RAD, comm = ''):
    z = np.array([0.,0.,1.])            # default cylinder orientation
    p = X2-X1                           # desired cylinder orientation
    r = np.linalg.norm(p)
    t = np.cross(z,p)                   # angle about which to rotate
    a = np.degrees(np.arccos( np.dot(z,p) / r ))   # rotation angle
    glPushMatrix()
    glTranslatef( X1[0], X1[1], X1[2] )
    glRotatef( a, t[0], t[1], t[2] )
    cylinder = gluNewQuadric()
    gluCylinder(cylinder, RAD, RAD, r, 10, 10)  
    glPopMatrix()  

def draw_axes():
    glBegin(GL_LINES)                 # x-axis (traditional-style)
    glColor3f(1, 0, 0)
    glVertex3f(-1000, 0, 0)
    glVertex3f(1000, 0, 0)
    glEnd()

    glBegin(GL_LINES)                 # y-axis (traditional-style)
    glColor3f(0, 0, 1)
    glVertex3f(0,-1000, 0)
    glVertex3f(0,1000, 0)
    glEnd()

    glBegin(GL_LINES)                 # z-axis (traditional-style)
    glColor3f(1, 0, 1)
    glVertex3f(0,0,-1000)
    glVertex3f(0,0,1000)
    glEnd()

# ********* Robot Toolbox settings *********
RTB_ENABLED = True
if RTB_ENABLED:

    # Robot control mode
    USE_SPACE_PILOT = False
    USE_ARROWS      = False
    USE_TRAJECTRY   = True
    USE_NUMS        = True

    print('Init EV3ARM robot toolbox \n--------------------------')
    robot = rtb.models.DH.EV3ARM()
    robot.q = robot.qS
    v = np.zeros(6)                 # End-effector linear speed - 6dof
    dt = 0.05                       # Movement frame time
    
    max_speed_ratio = .1            # Max_speed ratio used for trajectory
    max_speed = np.ones(robot.n)*500 # deg/sec
    max_torque = np.ones(robot.n)*.1      

    G = np.ones(robot.n)            # Gear ratios
    for i,link in enumerate(robot.links):
        if not np.isnan(link.G):
            G[i] = link.G
    G7 = -24/16                     # Gripper gear ratio
    G_ =  np.concatenate((G, [G7]), axis=0)
    A7 = 4*stud                     # Gripper finger length
    robot_q7 = 0                    # Position of gripper
    robot_qd7 = 0                   # Speed of gripper

    Tall = robot.fkine_all().data   # Poses of all robot joints
    Tep = robot.fkine(robot.q)      # End-effector pose
    Pos = Tep.t / stud
    print('Position of end-effector: {}[stud] {}{}[deg]'.format( Pos.astype(int), (robot.q*rad).astype(int), int(robot_q7*rad)) )

    ROLL = robot.n-1
    GRIP = robot.n  
    GRIP_LEN = 2*stud
    GRIP_RAD = .5*stud

    TRANS_INCREMENT = 1*stud
    ANGLE_INCREMENT = 5 #deg

    GAIN_T = 0.0005
    GAIN_R = 0.0001

    # Robot links shape: offset x,z (from midplane of doubled link/gripper) & radius 
    robot.shape = np.array([ (5*stud,0, 2*stud),
                            (0,0,      10*stud), 
                            (0,8*stud, 4*stud), 
                            (0,0,      5*stud), 
                            (0,6*stud, 2*stud), 
                            (0,0,      4*stud), 
                            (0,0,      2*stud) ])

    robot_moves = []                   
    move_completed = True   

    # Append trajectory to robot_moves
    def append_traj(new_robot_moves):
        global robot_moves
        for move in new_robot_moves:
            robot_moves.append(move)

    def move_robot_wrapper(dt):
        global move_completed

        if move_completed:
            move_robot()

    def move_robot():
        """
        Move robot according to end effector speed v or trajectory (sequence of movements) 
        in dt time steps as scheduled in pyglet.clock (Window.__init__)
        """
        global dt, v, robot, Tep, robot_q7, robot_qd7, robot_moves, move_completed
        
        move_completed = False

        # set q,qd from v 
        if USE_SPACE_PILOT or USE_ARROWS:

            # set v directly by _sp_control(SpacePilot)    
            if USE_SPACE_PILOT:  
                robot.qd = np.linalg.pinv(robot.jacob0()) @ v
                robot.q = robot.q + robot.qd * dt

            # move Tep by keyboard (v = p_servo(Tep))
            if USE_ARROWS:   
                v, arrived = rtb.p_servo(robot.fkine(), Tep, 1)
                robot.qd = np.linalg.pinv(robot.jacob0()) @ v
                robot.q = robot.q + robot.qd * dt

            # Check joints position limits
            for i in range(robot.n):            
                if not np.any(np.isnan(robot.qlim[:, i])):
                    if robot.q[i] < robot.qlim[0, i]:
                        robot.q[i] = robot.qlim[0, i]
                        robot.qd[i] = 0
                    if robot.q[i] > robot.qlim[1, i]:
                        robot.q[i] = robot.qlim[1, i]
                        robot.qd[i] = 0 
                
        # set q,qd from trajectory (v = robot.jacobe0() @ robot.qd)
        if USE_TRAJECTRY:
            robot.qd = np.zeros(robot.n)
            robot_qd7 = 0

            # If not empty robot_moves
            if len(robot_moves) > 0:
                move = robot_moves[0]   
                # time.sleep(dt) 

                if len(robot_moves) > 1:
                    robot_moves = robot_moves[1:len(robot_moves)]
                else:
                    robot_moves = []

                robot.qd = np.zeros(robot.n)
                robot_qd7 = 0

                # Trajectory of body
                if len(move) == robot.n:                          
                    robot.q = move

                # Trajectory of gripper              
                if len(move) == 1:
                    robot_q7 = move[0]

                Tep = robot.fkine(robot.q)
                Pos = Tep.t / stud
                print('Position of end-effector {}[stud], {} {} [deg]: '.format(Pos.astype(int), (robot.q*stud_6deg*10).astype(int)/10, int(robot_q7*rad) ) )

        # Movement of EV3 real robot 
        if EV3_ENABLED:
            move_robot_EV3()

        move_completed = True

    def draw_robot():
        Tall = robot.fkine_all().data
        # Links start_time & end point: P1, P2
        for i in range( len(Tall)):
            if i == 0:
                B = robot.base * 1
                P1 = B[:3,3] * SCALE + OFFSET
            else:
                P1 = Tall[i-1][:3,3] * SCALE + OFFSET
            P2 = Tall[i][:3,3] * SCALE + OFFSET 
            P12 = np.linalg.norm(P1 - P2) 

            # For link w/o lenght assume GRIP_LEN
            if P12 < stud * SCALE:
                P2 = P1 + Tall[i][:3,2] * stud * SCALE
            
            # Last link shorter by gripper    
            if i == len(Tall)-1:
                P2 = P2 + Tall[i][:3,2] * GRIP_LEN * SCALE   
            
            # Drow link as 1 or 2 parallel cylinders with offset
            offx, offz, rad = robot.shape[i]
            col = i % 2
            glColor3f(1-col, 0, col)
            if offx + offz == 0:
                draw_cylinder(P1, P2, rad * SCALE, 'link'+str(i) )
            else:
                OFF = (Tall[i][:3,0] * offx + Tall[i][:3,2] * offz) * SCALE 
                draw_cylinder(P1 - OFF, P2 - OFF, rad * SCALE, 'link'+str(i)+'_l' ) 
                draw_cylinder(P1 + OFF, P2 + OFF, rad * SCALE, 'link'+str(i)+'_r' )   

        # Gripper         
        i = len(Tall)-1
        P1 = Tall[i][:3,3] * SCALE  + OFFSET
        P2 = P1 + Tall[i][:3,2] * GRIP_LEN * SCALE
        OFF1 = Tall[i][:3,0] * (math.sin(robot_q7)*A7 + GRIP_RAD / 2) * SCALE 
        OFF2 = Tall[i][:3,1] * GRIP_RAD * SCALE 
        draw_cylinder(P1 - OFF1 - OFF2, P2 - OFF1 - OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_l1' ) 
        draw_cylinder(P1 + OFF1 - OFF2, P2 + OFF1 - OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_r1' )     
        draw_cylinder(P1 - OFF1 + OFF2, P2 - OFF1 + OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_l2' ) 
        draw_cylinder(P1 + OFF1 + OFF2, P2 + OFF1 + OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_r2' ) 

    def plot_data():
        """
        Plot speed_[] vs time_ 
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(robot.n):
            param_dict = {'marker': 'o', 'label': 'm'+str(i)}
            ax.plot(time_, speed_[:,i], **param_dict)
        plt.legend(loc='lower right')
        plt.show()


# ********* Lego EV3 robot settings *********
EV3_ENABLED = True
if EV3_ENABLED:

    # EV3 settings
    EV3_1_MAC = '00:16:53:45:23:9B'
    EV3_2_MAC = '00:16:53:41:3C:72'

    print("\nInit EV3 lego robot \n--------------------------")
    try:
        ev3_1 = ev3.EV3(protocol=ev3.USB, host=EV3_1_MAC)
        print(ev3_1)
    except:
        ev3_1 = None
        print("No device EV3_1 with MAC " + EV3_1_MAC)

    try:
        ev3_2 = ev3.EV3(protocol=ev3.USB, host=EV3_2_MAC)
        print(ev3_2)
    except:
        ev3_2 = None
        print("No device EV3_2 with MAC " + EV3_2_MAC)

    MOTORS_DEF = [
        (ev3.PORT_A, ev3_2), 
        (ev3.PORT_B, ev3_2), 
        (ev3.PORT_C, ev3_2), 
        (ev3.PORT_D, ev3_2), 

        (ev3.PORT_A, ev3_1), 
        (ev3.PORT_B, ev3_1), 
        (ev3.PORT_C, ev3_1), 
        (ev3.PORT_D, ev3_1),                                 
    ]

    TOUCH_SENSORS_DEF = [
        (None, None),
        (None, None),
        (ev3.PORT_1, ev3_1),
        (ev3.PORT_2, ev3_1),

        (None, None),
        (ev3.PORT_3, ev3_1),
        (None, None),
        (None, None),
    ]

    LARGE_MOTOR_MAX_SPEED = 1050 # deg/s
    MEDIUM_MOTOR_MAX_SPEED = 1560 # deg/s
    MAX_SPEED = [LARGE_MOTOR_MAX_SPEED, MEDIUM_MOTOR_MAX_SPEED]
    SHIFT_MOTOR_BY = 31 # deg

    LARGE_MOTOR_MAX_TORQUE  = .4 # Nm
    MEDIUM_MOTOR_MAX_TORQUE = 0.12 # Nm
    MAX_TORQUE = [LARGE_MOTOR_MAX_TORQUE, MEDIUM_MOTOR_MAX_TORQUE]
    RAMP_UP    = 0 #.25
    RAMP_DOWN  = 0 # .25
    MIN_DALF   = 5
    BRAKES    = True

    motors = [] 
    max_speed = []
    max_torque = []
    for motor_def in MOTORS_DEF:
        port, obj = motor_def
        print('connect to motor ',port, obj)
        motor, type_ = (None, 7)
        if port:
            try:
                motor = ev3.Motor(port, ev3_obj = obj )
                motor.verbosity = 0
                motor.delta_time = dt
                motor.stop(brake=BRAKES)   
                motor.position = 0
                type_ = motor.motor_type    # 7: EV3-Large, 8: EV3-Medium
            except:
                print("motor not found in ", port, obj) 

        motors.append(motor)
        max_speed.append(MAX_SPEED[type_ - 7])
        max_torque.append(MAX_TORQUE[type_ - 7])

    max_speed = np.array(max_speed)
    max_torque = np.array(max_torque)

    gyro_sensor = None
    color_sensor = None

    touch_sensors = []
    for sensor_def in TOUCH_SENSORS_DEF:        
        touch_sensor = None
        port, obj = sensor_def
        if port:
            print('connect to touch sensor ',port, obj)
            try:
                touch_sensor = ev3.Touch(port, ev3_obj = obj)
            except:
                touch_sensor = None
                print("touch sensor not found in ",port, obj)
        touch_sensors.append(touch_sensor)
  

    emergency_stop = False
    touch_dt       = 0.1                  # Touch sensor check time
    touch_now      = False

    # Calibration settings
    calibration_sequence  = [2,3,5]
    calibration_direction = [0,0,-1,-1, 0,-1,0,0] # movement to push touch sensor
    calibration_speed     = [0,0,[50,10],[50,10], 0,[50,10],0]
    calibration_zero      = [0,0, 0+4, +15+17, 0, 0+2. ,0,0]
    chess_poses = [
        #1
        [[0.0, 0.0, -1.1693705988362009, -0.2617993877991494, 0.0, -1.5009831567151233, 0.0],
         [0.0, 0.0, -0.8203047484373349, -0.5410520681182421, 0.0, -1.4660765716752366, 0.0]],
        #2 
        [[0.0,  0.0, -0.8726646259971649, -1.0646508437165412, 0.0, -0.9599310885968814, 0.0] ,
         [0.0, 0.0, -0.5585053606381855, -1.1868238913561442, 0.0, -1.0471975511965979, 0.0]], 
        #3
        [[0.0, 0.0, -0.820304748437335, -1.2042771838760875, 0.0, -0.9773843811168246, 0.0] ,
         [0.0, 0.0, -0.5061454830783556, -1.3089969389957472, 0.0, -1.0821041362364843, 0.0]], 
        #4
        [[0.0, 0.0, -0.715584993317675, -1.5707963267948968, 0.0, -0.7330382858376185, 0.0] ,
         [0.0, 0.0, -0.4014257279586956, -1.6755160819145565, 0.0, -0.8377580409572783, 0.0]], 
        #5
        [[0.0, 0.0, -0.6632251157578454, -1.6929693744344991, 0.0, -0.7330382858376185, 0.0] ,
         [0.0, 0.0, -0.34906585039886595, -1.6929693744344994, 0.0, -0.8901179185171083, 0.0]],
        #6
        [[0.0, 0.0, -0.6457718232379018, -1.6755160819145565, 0.0, -0.7853981633974484, 0.0] ,
         [0.0, 0.0, -0.45378560551852576, -1.5882496193148397, 0.0, -0.7853981633974485, 0.0]], 
        #7
        [[0.0, 0.0, -0.5410520681182424, -2.2165681500327983, 0.0, -0.45378560551852565, 0.0] ,
         [0.0, 0.0, -0.174532925199433, -1.6929693744344994, 0.0, -1.1693705988362009, 0.0]], 
        #8
        [[0.0, 0.0, -0.4886921905584124, -2.321287905152458, 0.0, -0.5759586531581287, 0.0] ,
         [0.0, 0.0, -0.06981317007977322, -1.797689129554159, 0.0, -1.2217304763960306, 0.0]],
    ]

    def save_calibration_data():
    
        filename = 'EV3ARM.dat'
        outfile = open(filename,'wb')
        pickle.dump([
            calibration_sequence,
            calibration_direction,
            calibration_speed,
            calibration_zero,
            chess_poses], outfile)
        outfile.close()

        print("Calibration data saved in ", filename)

    def read_calibration_data(filename = 'EV3ARM.dat'):
        global calibration_sequence, calibration_direction, calibration_speed, chess_poses

        if not os.path.isfile(filename):
            print('File EV3ARM.dat not found')
            return False
        else:
            file = open(filename,'rb')
            calibration_sequence, \
            calibration_direction, \
            calibration_speed, \
            calibration_zero, \
            chess_poses = pickle.load(file)
            file.close()
            print('Data imported')

    def shift_robot(id, dalf):       
        motor = motors[id]
        if motor is not None:
            motor.start_move_by(int(dalf), brake = True)

        # Get current positions Alf0 
        alf0 = []
        for motor in motors:
            position = 0
            if motor is not None:
                position = motor.position
            alf0.append(position) # deg
        # alf0 = np.array(alf0)

        print("Shift by {} > {}".format(SHIFT_MOTOR_BY, alf0 ))

    def set_zero_position():
        for motor in motors:
            if motor:
                motor.position = 0
        print('Motors set to zero position')

    def calibrate_robot():

        # Set Motors zero position
        for id in calibration_sequence:
            motor = motors[id]
            gear_ratio = G[id]
            direction = calibration_direction[id]
            zero = calibration_zero[id]
            touch_sensor = touch_sensors[id]
            print("*** CALIBRATE_ZERO", id)

            # Rough touch
            BREAK = True
            if not touch_sensor.touched:                
                speed_perc = calibration_speed[id][0]
                motor.start_move(speed = speed_perc, direction = direction)
                while True:
                    touched = touch_sensor.touched
                    if touched:
                        motor.stop(brake=BREAK) 
                        print('Rough touch ',touched, motor.position)
                        break

            if touch_sensor.touched: 
                speed_perc = calibration_speed[id][0]               
                motor.start_move(speed = speed_perc, direction = -direction)
                while True:
                    touched = touch_sensor.touched
                    if not touched:
                        motor.stop(brake=BREAK) 
                        print('Separate ',touched, motor.position)
                        break        

            if not touch_sensor.touched:                
                speed_perc = calibration_speed[id][1]               
                motor.start_move(speed = speed_perc, direction = direction)
                while True:
                    touched = touch_sensor.touched
                    if touched:
                        motor.stop(brake=BREAK) 
                        pos =motor.position
                        print('Fine touch ',touched, pos)
                        break 

            #  Move to position 0deg measured by external meter
            deg = int(zero*abs(gear_ratio))
            speed_perc = calibration_speed[id][0]    
            t = motor.move_by(deg, speed = speed_perc, ramp_up = 1, ramp_down = 1, brake = True)
            t.start()                
            t.join()  

        set_zero_position()    

    def set_break():
        for motor in motors:
            motor.stop(brake=BRAKES)   
            print('brake = ',BRAKES, motor.port)

    def emergency_button(t_list):
        """
        Stop & continue robot move when button is pressed
        """
        global emergency_stop, touch_now
        touch_sensor = touch_sensors[2]

        touch_prev = touch_now
        touch_now = touch_sensor.touched

        # when just touched
        if touch_now and not touch_prev:
            if_motor_busy = []        
            if not emergency_stop:
                emergency_stop = True
                for t in t_list:
                    if t.state == 'TO_STOP':
                        t.stop()
                for motor in motors:
                    if motor is not None:
                        motor.stop(brake=True)                     
                        if_motor_busy.append(motor.busy)
            else:
                emergency_stop = False
                for t in t_list:
                    if t.state == 'STOPPED':
                        t.cont()            
            print(('count','stop')[int(emergency_stop)], if_motor_busy)

        return move_completed

    def move_robot_EV3():
        global robot, robot_q7, robot_qd7, robot_moves, emergency_stop, move_correction  
        global start_time, end_time, delay_time, time_, speed_      
        
        alf1 = np.concatenate((
            (robot.q - robot.qS ) * G, 
            [robot_q7 * G7 - robot.q[ROLL] ] # Wrist roll compensation
        ), axis=0) * rad # [deg]
            
        # Get current positions Alf0 
        alf0 = []
        # busy = []
        for motor in motors:
            position = 0
            if motor is not None:
                position = motor.position
                # status = motor.busy
            alf0.append(position) # deg
            # busy.append(status)
        alf0 = np.array(alf0).astype(int)

        dalf = alf1 - alf0
        speed_perc = np.array([50,50,50,50, 50,100,50,10])

        alf1 = alf1.astype(int).tolist()
        speed_perc = speed_perc.astype(int).tolist()
        
        start_task = True
        t_list = list()


        for i in range(len(motors)):
            motor = motors[i]
            if motor is not None:
                if abs(dalf[i]) < MIN_DALF :
                    task = motor.stop_as_task(brake = True)
                    # motor.stop(brake = True)
                else:            
                    task = motor.move_to(alf1[i], speed = speed_perc[i], ramp_up = 0, ramp_down = 0, brake = True)  # int(abs(alf1[i])/10)                    
                    t_list.append(task)
            else:
                start_task = False
                print("Some motors are inactive")
        
        # print(busy, alf0, dalf)

        if start_task:      
            for t in t_list:
                t.start()                
            for t in t_list:
                t.join()

        # print("robot.q ", robot.q.tolist() )


# ********* Chess play settings *********
PLAY_CHESS = True
if PLAY_CHESS:
    # Chess board settings
    PLAY_AI_AI = False
    SQUARE_WIDTH = 3*stud
    SQUARE_HEIGHT = 7*stud/2.5
    BOARD_POS = np.array([-4*SQUARE_WIDTH, 17*stud, 6*stud])
    PAWN_HEIGHT = 5*stud
    KING_HEIGHT = 5*stud
    PIECE_HEIGHT = [
        2.2 * KING_HEIGHT, 
        PAWN_HEIGHT, 
        PAWN_HEIGHT, 
        PAWN_HEIGHT, 
        PAWN_HEIGHT, 
        KING_HEIGHT, 
        KING_HEIGHT ]
    PIECE_HANDLE_RAD = 0.5*stud
    PIECE_SHAPE = [ [[] ],  # Piece radius, height, alfa
                    [[1*stud,   2*stud,0],      [.5*stud,2*stud,0] ],                                                               # PAWN
                    [[1.5*stud,1*stud,0],[.5*stud,  1.5*stud,10*deg], [.5*stud,1.5*stud,-120*deg] ],                                  # KNIGHT
                    [[1.5*stud,1*stud,0],[1*stud,   2*stud,0],      [.5*stud,1*stud,0] ],                                           # BISHOP
                    [[1.5*stud,1*stud,0],[1.25*stud,3*stud,0] ],                                                                    # ROOK
                    [[1.5*stud,1*stud,0],[1*stud,   1*stud,0],      [.5*stud,.5*stud,0], [1*stud,.5*stud,0], [.5*stud,1*stud,0]   ],# QUEEN
                    [[1.5*stud,1*stud,0],[1*stud,   2*stud,0],      [.5*stud,.5*stud,0], [1*stud,.5*stud,0] ] ]                     # KING
    PIECE_COLORS = (
        (.3,.3,.3), # gray
        (1.,1.,1.),
        (1.,0.,0.)) # white      
    COLOR_NAMES = ['BLACK', 'WHITE']
    FILE_NAMES = ['A','B','C','D','E','F','G','H','I','J','K','L']
    TIME_FOR_MOVE = 1.0

    print('\nInit chess board \n--------------------------')
    board = chess.Board()
    #print(board.unicode(borders = True,empty_square = ' '))
    path = os.getcwd()
    engine = chess.engine.SimpleEngine.popen_uci(path + '\stockfish.exe')
    margin_file = 8
    margin_rank = -1
    uci = ''

    def find_pieces_on_board(board):
        """
        Assign all EV3 pieces to its squares based on current state of the board
        """
        pieces = []
        for i in range(8): #file - col
            for j in range(8): # rank - row
                piece = board.piece_at(chess.square(i,j))
                if piece:
                    piece_pos = BOARD_POS \
                        + np.array([(7-i) * SQUARE_WIDTH, (7-j) * SQUARE_HEIGHT, 0.]) \
                        + np.array([SQUARE_WIDTH/2., SQUARE_HEIGHT/2., 0.])
                    pieces.append([piece, piece_pos, -1, False])
        return pieces

    def find_chess_poses(chess_poses):
        """
        Define poses of robot over each board square
        """
        q_chess = []
        for i in range(10): # file - col
            x = (3.5-i) * SQUARE_WIDTH 
            col = []
            for j in range(8): # rank - row
                lev = copy.deepcopy(chess_poses[j])
                lev[0][0] = x
                lev[1][0] = x
                col.append(lev)
            q_chess.append(col)
        return q_chess

    def append_uci_move(uci):
        try:
            move = chess.Move.from_uci(uci)
            from_square = move.from_square
            to_square = move.to_square  
            i0, j0 = chess.square_file(from_square), chess.square_rank(from_square)
            i1, j1 = chess.square_file(to_square), chess.square_rank(to_square)            
            append_piece_move(i0, j0, i1, j1)

        except ValueError:
            print("Invalid uci (" + uci + "). Type again.")
            uci = ''

    def append_piece_move(i0, j0, i1, j1, piece_type=0, mode=0, i00=0, j00=0):
        """
        Define sequence of movements of robot to move piece based on 2 or 3 chess board squares 
            i0, j0 : move from square (file, rank - integers in range(0:7))
            i1, j1 : move to square (file, rank as integers in range(0:7))
            mode : 0 - move w/o capturing, 1 - move of captured piece, 2 - move of capturing piece
            i00, j00 : start_time from end of prev. move (file, rank - integers in range(0:7))
        """
        global q_chess

        q7lim = math.asin(2*stud / A7)
        OPEN_GRIPPER  = [q7lim]
        CLOSE_GRIPPER = [0]

        up = 1
        down = 0

        # Move w/o capturing
        if mode == 0:
            new_robot_moves = [
                q_chess[i0][j0][up], 
                OPEN_GRIPPER,
                q_chess[i0][j0][down], 
                CLOSE_GRIPPER,
                q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down],
                OPEN_GRIPPER,
                q_chess[i1][j1][up],
                CLOSE_GRIPPER,
                robot.qS  ]

        # Move Captured piece to margin
        elif mode == 1:
            new_robot_moves = [
                q_chess[i0][j0][up], 
                OPEN_GRIPPER,
                q_chess[i0][j0][down], 
                CLOSE_GRIPPER,
                q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down],
                OPEN_GRIPPER,
                q_chess[i1][j1][up]]  

        # Move Capturing piece to new square
        elif mode == 2:
            new_robot_moves = [
                [q_chess[i0][j0][up], 
                OPEN_GRIPPER,
                q_chess[i0][j0][down]], 
                CLOSE_GRIPPER,
                q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down],
                OPEN_GRIPPER,
                q_chess[i1][j1][up], 
                CLOSE_GRIPPER,
                robot.qS  ]

        append_traj(new_robot_moves)    

    def half_move(move):   
        """
        Make human or AI chess move
        """
        global uci, board, margin_file, margin_rank, pieces_     

        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            print('Invalid move: ' + uci)
        else:
            from_square = move.from_square
            to_square = move.to_square     

            moved_piece = board.piece_at(from_square)  
            captured_piece = board.piece_at(to_square)  
            board.push(move)
            print(COLOR_NAMES[int(moved_piece.color)] + ': ' + move.uci())

            if captured_piece:
                margin_rank += 1
                if margin_rank > 7:
                    margin_rank = 0
                    margin_file += 1

                i0, j0 = chess.square_file(to_square), chess.square_rank(to_square)
                i1, j1 = margin_file, margin_rank            
                append_piece_move(i0, j0, i1, j1, captured_piece.piece_type, mode = 1 )            
                i00, j00 = i1, j1
                i0, j0 = chess.square_file(from_square), chess.square_rank(from_square)
                i1, j1 = chess.square_file(to_square), chess.square_rank(to_square)
                append_piece_move(i0, j0, i1, j1, moved_piece.piece_type, mode = 2, i00 = i00, j00 = j00 )
                #pieces_ = find_pieces_on_board(board)
                print('Captured piece to: ' + FILE_NAMES[i1] + str(j1))

            i0, j0 = chess.square_file(from_square), chess.square_rank(from_square)
            i1, j1 = chess.square_file(to_square), chess.square_rank(to_square)
            append_piece_move(i0, j0, i1, j1, moved_piece.piece_type )
 
    def full_move(move):
        """
        Make full move - human based on typed or visually recognized, AI based on engine move
        """
        # Human move
        half_move(move)

        # AI move
        if not board.is_game_over():
            result = engine.play(board,chess.engine.Limit(time=TIME_FOR_MOVE))
            half_move(result.move)
            
            if board.is_game_over():
                print('You lost!')
        else:
            print('You won!')    

    def AI_AI(dt):
        """
        Half move of AI
        """
        global board
        # Wait for robot to finish its move
        if len(robot_moves) == 0:
            if not board.is_game_over():
                result = engine.play(board,chess.engine.Limit(time=1))
                half_move(move = result.move)         
            else:
                print('The End - ' + COLOR_NAMES[int(board.turn)] + ' won!' )

    def draw_board():      
        """
        Draw board and pieces
        """  
        # Draw board
        glBegin(GL_QUADS)
        k=0 # color index
        corners = np.array([
            [0, 0, 0], 
            [SQUARE_WIDTH, 0, 0], 
            [SQUARE_WIDTH, SQUARE_HEIGHT, 0], 
            [0, SQUARE_HEIGHT, 0] ])

        for i in range(8):
            k = 1-k
            for j in range(8):
                k = 1-k
                glColor3fv(PIECE_COLORS[k])
                for l in range(len(corners)):
                    glVertex3fv(tuple((
                        BOARD_POS \
                        + np.array([(7-i) * SQUARE_WIDTH, (7-j) * SQUARE_HEIGHT, 0]) \
                        + corners[l]) * SCALE + OFFSET))		
        glEnd()

        # Draw pieces
        for piece_ in pieces_:
            piece, piece_pos, dist, hold = tuple(piece_)
            P0 = piece_pos * SCALE + OFFSET
            P1 = P0
            c = int(piece.color)
            c_sign = 2*c-1
            if hold: 
                c = 2
            glColor3fv(PIECE_COLORS[c])

            pt = piece.piece_type
            for shape in PIECE_SHAPE[pt]:
                r, h, a = tuple(shape)
                P2 = P1 + np.array([0, np.sin(a)*h*c_sign, np.cos(a)*h]) * SCALE
                draw_cylinder(P1, P2, r * SCALE, piece.unicode_symbol() )
                P1 = P2

            glPushMatrix()
            P1 = P0 + np.array([0, 0, PIECE_HEIGHT[pt]]) * SCALE
            glTranslatef( P1[0], P1[1], P1[2] )
            sphere = gluNewQuadric()
            gluSphere(sphere,PIECE_HANDLE_RAD * SCALE, 5, 5)
            glPopMatrix()


# ********* Pyglet settings *********

move_key = {
    "a": pyglet.window.key.A,
    "b": pyglet.window.key.B,
    "c": pyglet.window.key.C,
    "d": pyglet.window.key.D,
    "e": pyglet.window.key.E,
    "f": pyglet.window.key.F,
    "g": pyglet.window.key.G,
    "h": pyglet.window.key.H,
    "1": pyglet.window.key._1,
    "2": pyglet.window.key._2,
    "3": pyglet.window.key._3,
    "4": pyglet.window.key._4,
    "5": pyglet.window.key._5,
    "6": pyglet.window.key._6,
    "7": pyglet.window.key._7,
    "8": pyglet.window.key._8,
    }

class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        global start0_time, start_time

        super().__init__(*args, **kwargs)

        self.set_minimum_size(200, 200)
        self.xAngle = -30
        self.yAngle = -0
        self.zAngle = -135
        self._init_gl()
        self.label = pyglet.text.Label( '---',
                          font_name='Arial',
                          font_size=20,
                          x=-75 + OFFSET[0], 
                          y=50 + OFFSET[1],
                          anchor_x='left', anchor_y='top')                

        self.F_keys =    [pyglet.window.key.F1, pyglet.window.key.F2, pyglet.window.key.F3, pyglet.window.key.F4, pyglet.window.key.F5, pyglet.window.key.F6, pyglet.window.key.F7, pyglet.window.key.F8 ]
        self.keys_up =   [pyglet.window.key._1, pyglet.window.key._2, pyglet.window.key._3, pyglet.window.key._4, pyglet.window.key._5, pyglet.window.key._6, pyglet.window.key._7, pyglet.window.key._8, ]
        self.keys_down = [pyglet.window.key.Q,  pyglet.window.key.W,  pyglet.window.key.E,  pyglet.window.key.R,  pyglet.window.key.T,  pyglet.window.key.Y,  pyglet.window.key.U,  pyglet.window.key.I,  ]
        self.i = 0

        pyglet.clock.schedule_interval(move_robot_wrapper, dt) 

        # if USE_SPACE_PILOT:
        #     self._init_sp()

        # if PLAY_CHESS:
        #     if PLAY_AI_AI:
        #         pyglet.clock.schedule_interval(AI_AI, 1)   
        print('Init pyglet') 

    def _init_sp(self):
        """
            Init SpacePilot control
        """
        devices = pyglet.input.get_devices()
        sp = devices[len(devices)-1]
        sp.open() 
        controls = sp.get_controls()
        self.sp_prev_state = self.sp_curr_state = self.diff = self.prev_diff = np.zeros(6)

        self.sp_z  = controls[0]
        self.sp_y  = controls[1]
        self.sp_x  = controls[2]
        self.sp_rz = controls[3]
        self.sp_ry = controls[4]
        self.sp_rx = controls[5]
        self.sp_left_butt  = controls[13] # Left
        self.sp_right_butt = controls[14] # Right

        self.sp_z.on_change         = self._sp_control 
        self.sp_y.on_change         = self._sp_control 
        self.sp_x.on_change         = self._sp_control 
        self.sp_rz.on_change        = self._sp_control
        self.sp_ry.on_change        = self._sp_control
        self.sp_rx.on_change        = self._sp_control
        self.sp_left_butt.on_press  = self._sp_control 
        self.sp_right_butt.on_press = self._sp_control      

    def _sp_control(self = None, value = None):
        global v
                                            # Y                Z                    X                                    
        self.sp_curr_state = np.array([ self.sp_y.value or 0,  self.sp_z.value or 0,  self.sp_x.value or 0, 
                                        self.sp_ry.value or 0, self.sp_rz.value or 0, self.sp_rx.value or 0]) * np.array([1,1,1,0,0,0])   
        self.diff = self.sp_curr_state - self.sp_prev_state
        for i in range(6):
            if abs(self.diff[i]) > 1000 or self.diff[i] == 0:
                self.diff[i] = self.prev_diff[i] 

        self.sp_prev_state = self.sp_curr_state  
        self.prev_diff = self.diff  

        v = self.diff * np.array([GAIN_T, GAIN_T, GAIN_T, GAIN_R, GAIN_R, GAIN_R]) * np.array([-1, +1, -1,   1, -1, 1]) 
 
    def _init_gl(self):
        glClearColor(195/255, 248/255, 248/255, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_COLOR_MATERIAL) # 1
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, vector(0.5, 0.5, 1, 0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, vector(0.5, 0.5, 1, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vector(1, 1, 1, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vector(1, 1, 1, 1))
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) # 2

    def on_draw(self):
        global dt, start_time, max_speed_ratio

        # Translate & Rotate scene
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glTranslatef(0, 0, -1000)
        glRotatef(self.xAngle, 1, 0, 0)
        glRotatef(self.yAngle, 0, 1, 0)
        glRotatef(self.zAngle, 0, 0, 1)
        
        draw_axes()

        if RTB_ENABLED:
            draw_robot()
            self.label.text = 'SR ' + str(np.round(max_speed_ratio,3)) 
            self.label.draw()

        if PLAY_CHESS: 
            draw_board()

        glPopMatrix()     
        
    def on_resize(self, width, height):
        width = width if width else 1
        height = height if height else 1
        aspectRatio = width / height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(35.0, aspectRatio, 1.0, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_text_motion(self, motion): # Rotate about the x or y axis
        global Tep, robot_q7

        if USE_ARROWS:
            # Arrows / Page Up/Down - end effector move
            if motion == pyglet.window.key.MOTION_UP:
                Tep = Tep * SE3.Ty(+TRANS_INCREMENT) 
            elif motion == pyglet.window.key.MOTION_DOWN:
                Tep = Tep * SE3.Ty(-TRANS_INCREMENT) 
            elif motion == pyglet.window.key.MOTION_LEFT:
                Tep = Tep * SE3.Tz(+TRANS_INCREMENT) 
            elif motion == pyglet.window.key.MOTION_RIGHT:
                Tep = Tep * SE3.Tz(-TRANS_INCREMENT) 
            elif motion == pyglet.window.key.MOTION_PREVIOUS_PAGE:
                Tep = Tep * SE3.Tx(+TRANS_INCREMENT) 
            elif motion == pyglet.window.key.MOTION_NEXT_PAGE:
                Tep = Tep * SE3.Tx(-TRANS_INCREMENT)            

            # Home / End - gripper
            elif motion == pyglet.window.key.MOTION_BEGINNING_OF_LINE:
                robot_q7 -= 0.1 * stud
            elif motion == pyglet.window.key.MOTION_END_OF_LINE:
                robot_q7 += 0.1 * stud
            if robot_q7 < GRIP_RAD:
                robot_q7 = GRIP_RAD
            if robot_q7 > 90*deg:
                robot_q7 = 90*deg                

    def on_key_press(self, symbol, modifiers):  
        global uci, PLAY_AI_AI, PLOT_TRESHOLD, PLOT_IMG_YEL, USE_NUMS, SHIFT_MOTOR_BY, BRAKES
        global robot, robot_q7

        if symbol == pyglet.window.key.PAGEUP:
            self.xAngle -= ANGLE_INCREMENT
        elif symbol == pyglet.window.key.PAGEDOWN:
            self.xAngle += ANGLE_INCREMENT
        elif symbol == pyglet.window.key.INSERT:
            self.zAngle -= ANGLE_INCREMENT
        elif symbol == pyglet.window.key.DELETE:
            self.zAngle += ANGLE_INCREMENT


        if symbol == pyglet.window.key.TAB:   
            USE_NUMS = not USE_NUMS 
            print('USE_NUMS',USE_NUMS)            

        if symbol == pyglet.window.key.Z:
            set_zero_position()

        if symbol == pyglet.window.key.C:   
            robot.q = robot.qS
            calibrate_robot()

        if symbol == pyglet.window.key.B:  
            BRAKES = not BRAKES 
            set_break()

        # Chess poses
        for i in range(8):                
            if symbol == self.F_keys[i]:
                self.i = i
                append_traj([chess_poses[i][0]])

        if symbol == pyglet.window.key.F9:
            append_traj([robot.qS])

        if symbol == pyglet.window.key.HOME: 
            chess_poses[self.i][1] = robot.q
            print("Set low pose for row {} | {}".format( i, robot.q))   
            save_calibration_data()

        if symbol == pyglet.window.key.END: 
            chess_poses[self.i][0] = robot.q
            print("Set low pose for row {} | {}".format( i, robot.q))   
            save_calibration_data()

        if USE_NUMS:
            unit = [stud,deg,deg,deg, deg,deg,deg,deg]

            for i in range(7):
                if symbol == self.keys_up[i]:                    
                    q = robot.q
                    q[i] = q[i] + SHIFT_MOTOR_BY*unit[i]
                    append_traj([q])

            for i in range(7):
                if symbol == self.keys_down[i]:                   
                    q = robot.q
                    q[i] = q[i] - SHIFT_MOTOR_BY*unit[i]
                    append_traj([q])

            if symbol == self.keys_up[7]:
                robot_q7 = robot_q7 + SHIFT_MOTOR_BY*unit[7]
                append_traj([q])

            if symbol == self.keys_down[7]:
                robot_q7 = robot_q7 - SHIFT_MOTOR_BY*unit[7]
                append_traj([q])


            if symbol == pyglet.window.key.NUM_SUBTRACT:
                SHIFT_MOTOR_BY = SHIFT_MOTOR_BY - 5 # deg/stud
                print("Shift by ",SHIFT_MOTOR_BY)

            if symbol == pyglet.window.key.NUM_ADD:
                SHIFT_MOTOR_BY = SHIFT_MOTOR_BY + 5 # deg/stud
                print("Shift by ",SHIFT_MOTOR_BY)


        # if PLAY_CHESS:
        #     # Turn on AI vs AI game
        #     if symbol == pyglet.window.key.A:   
        #         PLAY_AI_AI = not PLAY_AI_AI 
                
        #         if PLAY_AI_AI:
        #             pyglet.clock.schedule_interval(AI_AI, 1)
        #         else:
        #             pyglet.clock.unschedule(AI_AI)  

        #     # Submit typed human move
        #     elif symbol == pyglet.window.key.ENTER and len(uci) == 4:
        #         try:
        #             human_move = chess.Move.from_uci(uci)
        #             full_move(human_move)
        #         except ValueError:
        #             print("Invalid uci (" + uci + "). Type again.")
        #             uci = ''

        #     # Type human uci    
        #     else:
        #         for key in list(move_key.keys()):
        #             if symbol == move_key[key]:
        #                 uci += key

        if symbol == pyglet.window.key.ESCAPE:
            pyglet.clock.unschedule(move_robot)

            if PLAY_CHESS and PLAY_AI_AI: 
                pyglet.clock.unschedule(AI_AI) 

            pyglet.app.exit() 
            print('Exit pyglet')

    def on_close(self):
        pyglet.clock.unschedule(move_robot)   
        pyglet.app.exit()      

if __name__ == "__main__":

    read_calibration_data()
    pieces_ = find_pieces_on_board(board)
    q_chess = find_chess_poses(chess_poses)

    # calibrate_robot()

    demo_moves = [
        np.array([5*stud, 0*deg,   0*deg,  -15*deg,   0*deg,  -90*deg, 0*deg]),
        np.array([-5*stud, 0*deg,   0*deg,  -15*deg,   0*deg,  -90*deg, 0*deg]),
        robot.qS,

        np.array([0, 30*deg,   0*deg,  -15*deg,   0*deg,  -90*deg, 0*deg]),
        np.array([0, -30*deg,   0*deg,  -15*deg,   0*deg,  -90*deg, 0*deg]),
        robot.qS,

        np.array([0, 0*deg,   -30*deg,  -15*deg,   0*deg,  -90*deg, 0*deg]),
        robot.qS,

        np.array([0, 0*deg,   0*deg,  -45*deg,   0*deg,  -90*deg, 0*deg]),
        robot.qS,

        np.array([0, 0*deg,   0*deg,  -15*deg,   -15*deg,  -90*deg, 0*deg]),
        np.array([0, 0*deg,   0*deg,  -15*deg,    15*deg,  -90*deg, 0*deg]),
        robot.qS,

        np.array([0, 0*deg,   0*deg,  -15*deg,   0*deg,  -45*deg, 0*deg]),
        robot.qS,

        [30*deg],

        np.array([0, 0*deg,   0*deg,  -15*deg,   0*deg,  -90*deg,  30*deg]),
        np.array([0, 0*deg,   0*deg,  -15*deg,   0*deg,  -90*deg, -30*deg]),
        robot.qS,

        [0*deg],
        [30*deg],
        [0*deg],

        np.array([10*stud,   30*deg,  -30*deg,  -45*deg,   15*deg,  -60*deg,  30*deg]),
        np.array([-10*stud, -30*deg,  -30*deg,  -45*deg,  -15*deg,  -60*deg, -30*deg]),
        robot.qS,
    ]     
    # append_traj(demo_moves)

    # Spanish opening
    # append_uci_move('e1e4')    
    # append_uci_move('e7e5')   
    # append_uci_move('g1f3')    
    # append_uci_move('b8c6')   
    # append_uci_move('f1b5')   

    # while robot_moves:
    #     move_robot() 
    # print('Move completed!')

    Window(WINDOW_WIDTH, WINDOW_HEIGHT, 'EV3ARM simulation')
    pyglet.app.run()
