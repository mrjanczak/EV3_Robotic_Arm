import os.path
import sys
import time
import copy
import pickle
import numpy as np
from collections import namedtuple
from spatialmath import SE3
import matplotlib.pyplot as plt
import struct
from thread_task import Task, Periodic, Repeated, Sleep, concat

import pyglet
from pyglet.gl import *
from OpenGL.GL import *
from OpenGL.GLU import *

import roboticstoolbox as rtb
import ev3_dc as ev3
import chess
import chess.engine
import cv2


np.set_printoptions(sign=' ', formatter={'float': '{: 8.3f}'.format})
deg = np.pi / 180. # deg > rad
rad = 180. / np.pi
stud = .008  # m/stud
twoPi = 2. * np.pi
key_dict = {
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

# Gripper trajectory namedtuple
def gtraj(q0, qf, tv):
    t = np.linspace(0, 1, tv)  # normalized time from 0 -> 1
    q7 = np.linspace(q0, qf, tv)       
    return namedtuple('gtraj', 'q7 t')(q7, t)

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
    USE_KEYBOARD = False
    USE_TRAJECTRY = True

    CORRECT_SPEED = False
    TRANS_INCREMENT = 1*stud
    ANGLE_INCREMENT = 5 #deg
    GAIN_T = 0.0005
    GAIN_R = 0.0001
    ACC_TIME = 1
    SCALE_DT = True

    # Robot gripper size
    GRIP_LEN = 2*stud
    GRIP_RAD = .5*stud
    GRIP_MAX = 1.*stud
    CLOSE_GRIPPER = gtraj(GRIP_MAX, 0, 5)
    OPEN_GRIPPER  = gtraj(0, GRIP_MAX, 5)

    print('Init EV3ARM robot...')
    robot = rtb.models.DH.EV3ARM()
    robot.q = robot.qz
    Tep = robot.fkine(robot.qz)     # Pose controlled by keyboard
    Tall = robot.fkine_all().data   # Poses of all robot joints
    robot_q7 = 0                    # Position of gripper
    v = np.zeros(6)                 # End-effector linear speed - 6dof
    dt = 0.1                        # Frame time
    angle = np.zeros(robot.n+1)     # Position of servos
    speed = np.zeros(robot.n)       # Speed of servos
    max_speed_ratio = 1             # Speed / max_speed ratio
    max_speed = np.ones(robot.n)*1000
    max_torque = np.ones(robot.n)*.1      
    G = np.ones(robot.n)            # Gear ratios
    for i,link in enumerate(robot.links):
        if not np.isnan(link.G):
            G[i] = link.G
    G7 = 32/16                      # Gripper gear ratio
    a7 = 4*stud                     # Gripper finger length
    # Robot links shape: offset x,z (from midplane of doubled link/gripper) & radius 
    robot.shape = np.array([ (5*stud,0, 2*stud),
                            (0,0,      10*stud), 
                            (0,8*stud, 4*stud), 
                            (0,0,      5*stud), 
                            (0,6*stud, 2*stud), 
                            (0,0,      4*stud), 
                            (0,0,      2*stud) ])
    robot_moves = []
    step = 0                            
    print('EV3ARM ready. Position of end-effector [stud]: ',Tep.t / stud)

    # Diagnostic
    speed_ = np.array([speed])      
    time_ = [0]
    q_chess = None                  # Chess board squares poses
    f = open("speed_.dat","w")
    f.write('')
    f.close()

    def move_robot(dt):
        """
        Move robot according to end effector speed v or by 1 step from trajectory (sequence of movements) 
        in dt time steps as scheduled in pyglet.clock (Window.__init__)

            dt : time step
        """
        global v, speed, speed_, max_speed, max_speed_ratio, robot_q7, robot_moves, step, pieces_, start
        start = time.time()

        # Define speed of motors
        speed = np.zeros(robot.n)

        if USE_SPACE_PILOT or USE_KEYBOARD:
            # set v directly by _sp_control(SpacePilot)    
            if USE_SPACE_PILOT:  
                robot.qd = np.linalg.pinv(robot.jacob0()) @ v
                robot.q = robot.q + robot.qd * dt

            # move Tep by keyboard (v = p_servo(Tep))
            if USE_KEYBOARD:   
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

            speed = robot.qd * G 
                
        # set q,qd from trajectory (v = robot.jacobe0() @ robot.qd)
        if USE_TRAJECTRY:
            robot_moves_len = len(robot_moves)
            if robot_moves_len > 0:
                move = robot_moves[0]
                move_len = len(move.t)

                # Trajectory of RTB virtual robot
                if hasattr(move, 'q'):
                    # robot.q = move.q[step]
                    if step < move_len-1:
                        robot.qd = (move.q[step+1] - move.q[step]) / dt
                    else:
                        robot.qd = np.zeros(robot.n)
                    robot.q = move.q[step]

                # Move gripper              
                if hasattr(move, 'q7'):
                    robot_q7 = move.q7[step]
                    robot.qd = np.zeros(robot.n)

                # Trajectory of EV3 real robot - correction of actual speed 
                if EV3_ENABLED and EV3_TRAJECTORY:
                    if CORRECT_SPEED:
                        for i in range(robot.n):
                            angle[i] = m[i].position_sp / m[i].count_per_rot * twoPi
                        speed = robot.qd * G + (robot.q * G - angle[:robot.n]) / dt 
                    else:
                        speed = robot.qd * G

                # Next step or move
                step += 1

                # Last step in move
                if step == len(move.t):
                    robot_moves = robot_moves[1:robot_moves_len]
                    step = 0
                    robot.qd = np.zeros(robot.n)  
                    speed = np.zeros(robot.n)                 
            
                    # Finish the move - go to final position
                    if EV3_ENABLED and EV3_TRAJECTORY:
                        for i in range(robot.n):
                            angle[i] = robot.q * G
                            m[i].on_to_position( 
                                position = angle[i] / twoPi * m[i].count_per_rot, # [rad / 2Pi * deg/1rev = deg]
                                speed=SpeedDPS(max_speed[i]*rad), 
                                brake=True, block=False )

                    # f = open("speed_.dat", "a")
                    # f.write('\n')
                    # f.close

        # Used by EV3 robot and for diagnostic
        max_speed_ratio = max(.5, np.max( np.divide(np.fabs(speed), max_speed[:robot.n]) ) )
        speed = speed / max_speed_ratio

        # Set motor speed
        if EV3_ENABLED and EV3_TRAJECTORY:
            for i in range(robot.n):
                m[i].on(speed=SpeedDPS(speed[i]*rad), brake=True, block=False)

            # Set speed of gripper 
            ROLL = robot.n-1
            GRIP = robot.n
            # Compensate speed of last roll joint if it moves
            if abs(speed[ROLL]) > 0:
                m[GRIP].on(speed=SpeedDPS(speed[ROLL]*rad / G[ROLL]), brake=True, block=False)    

            # Control gripper position
            else:
                m[GRIP].on_to_position(
                    position = (np.arcsin(q7/a7)*G7 + robot.q[ROLL] ) / twoPi * m[GRIP].count_per_rot, 
                    speed=SpeedDPS(max_speed[GRIP]*rad), 
                    brake=True, block=True) 

        # Move pieces by gripper 
        if PLAY_CHESS:
            T = robot.fkine().A
            P1 = T[:3,3] - T[:3,2] * PIECE_HANDLE_RAD/2
            for i in range(len(pieces_)):
                piece, piece_pos, dist, hold = pieces_[i]
                pt = piece.piece_type
                P2 = piece_pos + np.array([0, 0, PIECE_HEIGHT[pt] ])
                dist = np.linalg.norm(P2 - P1)  
                pieces_[i][2] = dist

                if dist <= PIECE_HANDLE_RAD or hold:
                    # Move piece
                    if robot_q7 <= PIECE_HANDLE_RAD:  
                        pieces_[i][3] = True
                        pieces_[i][1] = P1 + np.array([0, 0, -PIECE_HEIGHT[pt] ])
                    # Drop piece
                    else:
                        pieces_[i][3] = False
                        pieces_[i][1][2] = BOARD_POS[2]    

        # Logfile - speed_.dat
        if robot_moves_len > 0:
            timer0 = time.time() - start0
            time_.append(timer0)
            speed_ = np.append(speed_, [speed*rad], axis = 0) # deg/s

            f = open("speed_.dat", "a")
            f.write( \
                ' t '  + str("{:4.2f}".format(timer0)) \
                + ' sr'  + str("{:4.2f}".format(max_speed_ratio)) \
                # + ' P12 ' + str("{:6.3f}".format(pieces_[1][2])) \
                # + ' P1 '  + str(P1) \
                # + ' P2 '  + str(pieces_[1][1] + np.array([0, 0, PIECE_HEIGHT[1] ])) \
                + ' q  ' + str(robot.q*G*rad) \
                + ' qd ' + str(robot.qd*G*rad) \
                + ' q7 ' + str("{:6.3f}".format(robot_q7)) \
                + ' sp ' + str(speed*rad) 
                + '\n')
            f.close()        

        # capture video
        # ret, img = cap.read()
        # cv2.imshow('img', img) 

    def draw_robot():
        Tall = robot.fkine_all().data
        # Links start & end point: P1, P2
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
        OFF1 = Tall[i][:3,0] * (robot_q7 + GRIP_RAD / 2) * SCALE 
        OFF2 = Tall[i][:3,1] * GRIP_RAD * SCALE 
        draw_cylinder(P1 - OFF1 - OFF2, P2 - OFF1 - OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_l1' ) 
        draw_cylinder(P1 + OFF1 - OFF2, P2 + OFF1 - OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_r1' )     
        draw_cylinder(P1 - OFF1 + OFF2, P2 - OFF1 + OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_l2' ) 
        draw_cylinder(P1 + OFF1 + OFF2, P2 + OFF1 + OFF2, GRIP_RAD * SCALE, 'grip'+str(i)+'_r2' ) 

    def plot_speed():
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
EV3_ENABLED = False #True
EV3_TRAJECTORY = False
if EV3_ENABLED:

    # EV3 settings
    EV3_1_MAC = '00:16:53:45:23:9B'
    EV3_2_MAC = '00:16:53:41:3C:72'

    # type of motor (0: EV3-Large, 1: EV3-Medium, )
    motors = [
        (ev3.PORT_A, EV3_2_MAC, 0), 
        (ev3.PORT_B, EV3_2_MAC, 0), 
        (ev3.PORT_C, EV3_2_MAC, 0), 
        (ev3.PORT_D, EV3_2_MAC, 0), 

        (ev3.PORT_A, EV3_1_MAC, 1), 
        (ev3.PORT_B, EV3_1_MAC, 1), 
        (ev3.PORT_C, EV3_1_MAC, 1), 
        (ev3.PORT_D, EV3_1_MAC, 1),   
    ]

    LARGE_MOTOR_MAX_SPEED = 1050 * deg # rad/s
    MEDIUM_MOTOR_MAX_SPEED = 1560 * deg # rad/s
    MAX_SPEED = [LARGE_MOTOR_MAX_SPEED, MEDIUM_MOTOR_MAX_SPEED]

    LARGE_MOTOR_MAX_TORQUE  = .4 # Nm
    MEDIUM_MOTOR_MAX_TORQUE = 0.12 # Nm
    MAX_TORQUE = [LARGE_MOTOR_MAX_TORQUE, MEDIUM_MOTOR_MAX_TORQUE]

    print("Init EV3")
    ev3_1 = ev3.EV3(protocol=ev3.USB, host=EV3_1_MAC)
    ev3_2 = ev3.EV3(protocol=ev3.USB, host=EV3_2_MAC)

    m = []
    max_speed = []
    max_torque = []
    for motor in motors:
        dev, port, type_ = motor
        m.append(ev3.Motor(port, ev3_obj = dev ))
        max_speed.append(MAX_SPEED[type_])
        max_torque.append(MAX_TORQUE[type_])

    for i in range(len(m)):
        m_.verbosity = 0
        m_.delta_time = .01
        m_.stop(brake=False)   

    print("Calibrate EV3")    
    sleep(1) # Wait for robot to drop to initial position
    for i in range(len(m)):  
        m[i].reset_position()

    def check_delay():
        delay_time = np.arraty([])
        N = 10
        for t in range(N):
            angle = [np.zeros(7)]
            start = time.time()
            for i in range(len(m)):
                angle[i] = m[i].position
                m[i].speed_sp = 0          
            stop = time.time()
            delay_time = np.append(delay_time, stop - start)
            print(angle, stop - start)
        print("Avg time to check positions [s]: ", np.average(delay_time))

    check_delay()


# ********* Chess play settings *********
PLAY_CHESS = True
if PLAY_CHESS:
    # Chess board settings
    PLAY_AI_AI = False
    BOARD_POS = np.array([-12*stud, 14*stud, 2*stud])
    SQUARE_WIDTH = 3*stud
    SQUARE_HEIGHT = 7*stud/2.5
    PAWN_HEIGHT = 4.5*stud
    KING_HEIGHT = 5.5*stud
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
                    [[1.5*stud,1*stud,0],[.5*stud,  2.5*stud,10*deg], [.5*stud,1.5*stud,-120*deg] ],                                  # KNIGHT
                    [[1.5*stud,1*stud,0],[1*stud,   2*stud,0],      [.5*stud,1*stud,0] ],                                           # BISHOP
                    [[1.5*stud,1*stud,0],[1.25*stud,3*stud,0] ],                                                                    # ROOK
                    [[1.5*stud,1*stud,0],[1*stud,   2*stud,0],      [.5*stud,.5*stud,0], [1*stud,.5*stud,0], [.5*stud,1*stud,0]   ],# QUEEN
                    [[1.5*stud,1*stud,0],[1*stud,   3*stud,0],      [.5*stud,.5*stud,0], [1*stud,.5*stud,0] ] ]                     # KING
    HANDLE_LEV = [
        2.2 * KING_HEIGHT, 
        PAWN_HEIGHT - PIECE_HANDLE_RAD, 
        KING_HEIGHT - PIECE_HANDLE_RAD ]
    PIECE_TYPE_LEV = [0, 1, 1, 1, 1, 2, 2]
    PIECE_COLORS = (
        (.3,.3,.3), # gray
        (1.,1.,1.),
        (1.,0.,0.)) # white      
    COLOR_NAMES = ['BLACK', 'WHITE']
    FILE_NAMES = ['A','B','C','D','E','F','G','H','I','J','K','L']
    TIME_FOR_MOVE = 1.0

    print('Init chess board...')
    board = chess.Board()
    #print(board.unicode(borders = True,empty_square = ' '))
    path = os.getcwd()
    engine = chess.engine.SimpleEngine.popen_uci(path + '\stockfish.exe')

    def find_pieces_on_board(board):
        """
        Assign all EV3 pieces to its squares based on current state of board
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

    pieces_ = find_pieces_on_board(board)

    def find_chess_poses():
        """
        Define poses of robot over each board square
        """
        q_chess = []
        q0_ = [robot.q0A, robot.q0B, robot.q0C, robot.q0D, robot.q0E, robot.q0F, robot.q0G, robot.q0H, 
                robot.q0I, robot.q0J]
        for i in range(10): #8 file - col
            rowq = []
            for j in range(8): #10 rank - row
                square_cent_pos = BOARD_POS \
                    + np.array([(7-i) * SQUARE_WIDTH, (7-j) * SQUARE_HEIGHT, 0.]) \
                    + np.array([SQUARE_WIDTH/2., SQUARE_HEIGHT/2., 0.])
                levq = []
                for k in range(3):
                    x, y, z = tuple(square_cent_pos + np.array([0., 0., HANDLE_LEV[k]]))
                    T = SE3(x, y, z) # * SE3.Ry(180*deg) * SE3.Rz(90*deg) 
                    q, success, err = robot.ikcon(T, q0=q0_[i])   # solve IK, ignore additional outputs  
                    if not success:
                        print('ikcon failed for square/lev: ' + str(i) + str(j) + str(k))
                        print(err)
                    levq.append(q)
                rowq.append(levq)
            q_chess.append(rowq)
        return q_chess
        
    if not os.path.isfile('chess_poses.p'):
        print('Calculate chess poses...')
        q_chess = find_chess_poses()
        pickle.dump( q_chess, open( "chess_poses.p", "wb" ) )
    else:
        q_chess = pickle.load( open( "chess_poses.p", "rb" ) )

    margin_file = 8
    margin_rank = -1
    uci = ''

    def move_piece_by_robot(i0, j0, i1, j1, piece_type, mode=0, i00=0, j00=0):
        """
        Define sequence of movements of robot to move piece based on 2 or 3 chess board squares 
            i0, j0 : move from square (name, rank - integers in range(0:7))
            i1, j1 : move to square (name, rank as integers in range(0:7))
            mode : 0 - move w/o capturing, 1 - move of captured piece, 2 - move of capturing piece
            i00, j00 : start from end of prev. move (name, rank - integers in range(0:7))
        """
        global robot_moves

        up = PIECE_TYPE_LEV[0]
        down = PIECE_TYPE_LEV[piece_type]
        qz = robot.qz

        # Move w/o capturing
        if mode == 0:
            moves_ = [
                [qz, q_chess[i0][j0][up]], 
                [OPEN_GRIPPER],
                [q_chess[i0][j0][up], q_chess[i0][j0][down]], 
                [CLOSE_GRIPPER],
                [q_chess[i0][j0][down], q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down]],
                [OPEN_GRIPPER],
                [q_chess[i1][j1][down], q_chess[i1][j1][up], qz],
                [CLOSE_GRIPPER]  ]

        # Move Captured piece to margin
        elif mode == 1:
            moves_ = [
                [qz, q_chess[i0][j0][up]], 
                [OPEN_GRIPPER],
                [q_chess[i0][j0][up], q_chess[i0][j0][down]], 
                [CLOSE_GRIPPER],
                [q_chess[i0][j0][down], q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down]],
                [OPEN_GRIPPER],
                [q_chess[i1][j1][down], q_chess[i1][j1][up]]  ]

        # Move Capturing piece to new square
        elif mode == 2:
            moves_ = [
                [q_chess[i00][j00][up], q_chess[i0][j0][up], q_chess[i0][j0][down]], 
                [CLOSE_GRIPPER],
                [q_chess[i0][j0][down], q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down]],
                [OPEN_GRIPPER],
                [q_chess[i1][j1][down], q_chess[i1][j1][up], qz],
                [CLOSE_GRIPPER]  ]

        for move_ in moves_:
            if len(move_) > 1:
                pass
                mstraj = rtb.trajectory.mstraj(
                    np.array(move_), 
                    dt, 
                    ACC_TIME, 
                    qdmax = np.divide(max_speed[:robot.n], G),
                    verbose = False,)
                robot_moves.append(mstraj)     
            else:
                # move of gripper 
                robot_moves.append(move_[0]) # gripper state: close or open    

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
            if USE_CAMERA:
                mask_b0 = board_mask(board)

            if captured_piece:
                margin_rank += 1
                if margin_rank > 7:
                    margin_rank = 0
                    margin_file += 1

                i0, j0 = chess.square_file(to_square), chess.square_rank(to_square)
                i1, j1 = margin_file, margin_rank            
                move_piece_by_robot(i0, j0, i1, j1, captured_piece.piece_type, mode = 1 )            
                i00, j00 = i1, j1
                i0, j0 = chess.square_file(from_square), chess.square_rank(from_square)
                i1, j1 = chess.square_file(to_square), chess.square_rank(to_square)
                move_piece_by_robot(i0, j0, i1, j1, moved_piece.piece_type, mode = 2, i00 = i00, j00 = j00 )
                #pieces_ = find_pieces_on_board(board)
                print('Captured piece to: ' + FILE_NAMES[i1] + str(j1))

            i0, j0 = chess.square_file(from_square), chess.square_rank(from_square)
            i1, j1 = chess.square_file(to_square), chess.square_rank(to_square)
            move_piece_by_robot(i0, j0, i1, j1, moved_piece.piece_type )
            #pieces_ = find_pieces_on_board(board)

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


# ********* Camera settings *********
USE_CAMERA = False
if USE_CAMERA:    
    class Plotter:
        def __init__(self, plot_height, plot_width, ref):
            self.height = plot_height
            self.width = plot_width
            self.height2 = int(plot_height/2)
            self.colors = [(255,0,0),(0,255,0),(0,0,0)]
            self.val = np.array([])
            self.ref = int(ref)

        # Update new values in plot
        def add_point(self, label, val, txt):    
            if self.val.size == 0:
                self.val = np.array([val])
            else:
                self.val = np.append(self.val, np.array([val]), axis = 0)
            rows, cols = self.val.shape

            if rows > self.width:
                self.val = self.val[rows-self.width : self.width - 1, :]
                rows, cols = self.val.shape

            self.plot = np.ones((self.height, self.width, 3))*255       
            cv2.line(self.plot, (0, self.height2 - self.ref), (self.width, self.height2 - self.ref), (0,0,255), 1)
            for j in range(cols):
                for i in range(rows - 1):
                    cv2.line(self.plot, (i, self.height2 - int(self.val[i, j])), (i+1, self.height2 - int(self.val[i+1, j])), self.colors[j], 1)
            cv2.putText(self.plot, txt, (5, self.height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) )
            cv2.imshow(label, self.plot)

    def board_mask(board):
        mask_b = np.zeros([8,8]).astype(bool)
        for f in range(8):
            for r in range(8):
                square = chess.square(f,r)
                piece = board.piece_at(square)
                if piece and piece.color:
                    mask_b[7-r,f] = True
        return mask_b

    def capture_video(data, img_yel, mask_b0 = None, treshold = 0):
        w,h, marg = data   

        yellow_mean = np.zeros([8,8])
        for r in range(8):
            for c in range(8):
                cropped = img_yel[int(r * h/8.) + marg: int((r+1) * h/8.) - marg, int(c * w/8.) + marg: int((c+1) * w/8.) - marg]
                yellow_mean[r][c] = (cropped>0).mean()
        
        if treshold == -1:
            # Calculate treshold based on mask_b0
            yellow_min = np.min(np.ma.masked_array(yellow_mean, mask=np.invert(mask_b0) )) 
            black_max = np.max(np.ma.masked_array(yellow_mean, mask=mask_b0 ))
            diff = (yellow_min - black_max)/2
            return yellow_min, black_max, diff

        else:
            # Calculate change of yellow pieces
            mask_b1 = np.round(yellow_mean > treshold)
            mask_diff = mask_b1.astype(int) - mask_b0.astype(int)
            from_, to_ = [], []
            for f in range(8):
                for r in range(8):
                    if mask_diff[r,f] == -1:
                        from_.append(chess.FILE_NAMES[f] + chess.RANK_NAMES[7-r])
                    if mask_diff[r,f] == +1:
                        to_.append(chess.FILE_NAMES[f] + chess.RANK_NAMES[7-r])
            mask_diff_sum = np.sum(np.abs(mask_diff))
            return from_, to_, mask_diff_sum

        # Setup camera for chess board optical processing settings
        
        CAMERA = 0
        FRAME_WIDTH = 1920
        FRAME_HEIGHT = 1080

        BOARD_HEIGHT = 320
        BOARD_WIDTH = 340 
        SQUARE_MARG = 4
        BOARD_X, BOARD_Y = (int((FRAME_WIDTH - BOARD_WIDTH)/2), int((FRAME_HEIGHT - BOARD_HEIGHT)/2))

        YELLOW_HSV = (10,10,100, 40,255,255)
        COLOR_TRESHOLD = 0.05
        PLOT_TRESHOLD = True
        PLOT_SCALE = 5
        PLOT_IMG_YEL = True

        # Setup camera
        cap = cv2.VideoCapture(CAMERA) # ,cv2.CAP_DSHOW
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Heat up camera :)
        for i in range(30):
            ret, img = cap.read()
        img_yel = img

        # Prepare edges mask - horizontal and vertical lines splitting board into squares
        mask_e = np.zeros([BOARD_HEIGHT, BOARD_WIDTH])
        for r in range(8):
            for c in range(8):
                mask_e[int(r*BOARD_HEIGHT/8.) + SQUARE_MARG : int((r+1)*BOARD_HEIGHT/8.) - SQUARE_MARG, 
                    int(c*BOARD_WIDTH/8.)  + SQUARE_MARG : int((c+1)*BOARD_WIDTH/8.)  - SQUARE_MARG] = 255
        mask_e = mask_e.astype(np.uint8)

    def board_mask(board):
        """
        Create mask of White pieces on board
        """
        mask_b = np.zeros([8,8]).astype(bool)
        for f in range(8):
            for r in range(8):
                square = chess.square(f,r)
                piece = board.piece_at(square)
                if piece and piece.color:
                    mask_b[7-r,f] = True
        return mask_b

    mask_b0 = board_mask(board)

    def capture_video():
        """
        Capture video and split for squares. Check mean level of yellow color
        """
        global img, img_yel
        ret, img = cap.read()
        # flip, crop and remove edges
        img = cv2.flip(img, -1)[BOARD_Y:BOARD_Y + BOARD_HEIGHT, BOARD_X:BOARD_X + BOARD_WIDTH] 
        img = cv2.bitwise_and(img, img, mask=mask_e)

        # increase yellow saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l1,l2,l3, u1,u2,u3 = YELLOW_HSV
        mask_y= cv2.inRange(hsv, np.array([l1,l2,l3]).astype(int), np.array([u1,u2,u3]).astype(int) )
        hsv[:,:,1] = mask_y
        img_yel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_yel = cv2.bitwise_and(img_yel, img_yel, mask=mask_y)   
            
        yellow_mean = np.zeros([8,8])
        for r in range(8):
            for c in range(8):
                cropped = img_yel[int(r * h/8.) + marg: int((r+1) * h/8.) - marg, int(c * w/8.) + marg: int((c+1) * w/8.) - marg]
                yellow_mean[r][c] = (cropped>0).mean()

        return yellow_mean

    def recognize_uci():
        """
        Recognize move by comparison of prev. and current board mask of white pieces
        """
        yellow_mean = capture_video()

        # Calculate change of yellow pieces
        mask_b1 = np.round(yellow_mean > COLOR_TRESHOLD)
        mask_diff = mask_b1.astype(int) - mask_b0.astype(int)
        from_, to_ = [], []
        for f in range(8):
            for r in range(8):
                if mask_diff[r,f] == -1:
                    from_.append(chess.FILE_NAMES[f] + chess.RANK_NAMES[7-r])
                if mask_diff[r,f] == +1:
                    to_.append(chess.FILE_NAMES[f] + chess.RANK_NAMES[7-r])
        mask_diff_sum = np.sum(np.abs(mask_diff))

        if len(from_) == 1 and len(to_) == 1 and mask_diff_sum == 2:
            return str(from_[0]) + str(to_[0]) 
        else: 
            raise ValueError(f"Wrong uci (from: {from_}, to: {to_}, sum of changes: {mask_diff_sum}") 

    def plot_treshold(dt):
        """
        Plot Treshold (call recognize_uci() with default treshold = -1)
        """
        yellow_mean = capture_video()
        
        # Calculate treshold based on mask_b0
        yellow_min = np.min(np.ma.masked_array(yellow_mean, mask=np.invert(mask_b0) )) 
        black_max = np.max(np.ma.masked_array(yellow_mean, mask=mask_b0 ))
        diff = (yellow_min - black_max)/2

        yellow   = round(yellow_min * 100, 1)
        black    = round(black_max * 100, 1)
        diff     = round(diff * 100, 1)
        treshold = round(black + diff, 1)
        data = np.array([yellow, treshold, black]) * PLOT_SCALE
        txt = str(yellow) + '% > '  + str(treshold) + '% > ' + str(black) + '% '
        plot.add_point('treshold', data, txt)

    def plot_yellow(dt):
        cv2.imshow('Yellow', img_yel)

# ---------------------------------------------
# ----------------- MAIN PROGRAM --------------
# ---------------------------------------------
start0 = time.time()
start = 0
# move_piece_by_robot(1, 0, 2, 2,  1)

class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_minimum_size(200, 200)
        self.xAngle = -45
        self.yAngle = -0
        self.zAngle = -135
        self._init_gl()
        self.label = pyglet.text.Label( '---',
                          font_name='Arial',
                          font_size=20,
                          x=-75 + OFFSET[0], 
                          y=50 + OFFSET[1],
                          anchor_x='left', anchor_y='top')                

        if RTB_ENABLED:
            pyglet.clock.schedule_interval(move_robot, dt)          
            if USE_SPACE_PILOT:
                self._init_sp()

        if USE_CAMERA:
            if PLOT_TRESHOLD:
                pyglet.clock.schedule_interval(plot_treshold, dt)

            if PLOT_IMG_YEL:
                pyglet.clock.schedule_interval(plot_yellow, dt)

        if PLAY_CHESS:
            if PLAY_AI_AI:
                pyglet.clock.schedule_interval(AI_AI, 1)     

    def _init_sp(self):
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

        self.sp_z.on_change  = self._sp_control 
        self.sp_y.on_change  = self._sp_control 
        self.sp_x.on_change  = self._sp_control 
        self.sp_rz.on_change = self._sp_control
        self.sp_ry.on_change = self._sp_control
        self.sp_rx.on_change = self._sp_control
        self.sp_left_butt.on_press  = self._sp_control 
        self.sp_right_butt.on_press = self._sp_control      

    def _sp_control(self = None, value = None):
        global v
                                            # Y                Z                    X                                    
        self.sp_curr_state = np.array([ self.sp_y.value or 0, self.sp_z.value or 0, self.sp_x.value or 0, 
                                        self.sp_ry.value or 0, self.sp_rz.value or 0, self.sp_rx.value or 0]) * np.array([1,1,1,0,0,0])   
        self.diff = self.sp_curr_state - self.sp_prev_state
        for i in range(6):
            if abs(self.diff[i]) > 1000 or self.diff[i] == 0:
                self.diff[i] = self.prev_diff[i] 

        self.sp_prev_state = self.sp_curr_state  
        self.prev_diff = self.diff  

        v = self.diff * np.array([GAIN_T, GAIN_T, GAIN_T, GAIN_R, GAIN_R, GAIN_R]) * np.array([-1, +1, -1,   1, -1, 1]) 
        
        # label1.text = 'xyz[%0.3f, %0.3f, %0.3f]' % (tuple(effector_pos))
        # label2.text = 'xyz[%0.3f, %0.3f, %0.3f, %0.3f]' % (tuple(arm_pos)) 

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
        # 1 & 2 mean that we can use glColor*() to color materials

    def on_draw(self):
        global dt, start, min_speed_ratio

        # Translate & Rotate scene
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glTranslatef(0, 0, -600)
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

        if RTB_ENABLED:
            if SCALE_DT:
                timer = time.time() - start
                dt_scaled = dt * max_speed_ratio     
                if timer < dt_scaled:
                    time.sleep(dt_scaled - timer)       
        
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

        if RTB_ENABLED and USE_KEYBOARD:
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

            if robot_q7 > GRIP_MAX:
                robot_q7 = GRIP_MAX                

        else:
            # Num pad - camera rotation
            if motion == pyglet.window.key.MOTION_UP:
                self.xAngle -= ANGLE_INCREMENT
            elif motion == pyglet.window.key.MOTION_DOWN:
                self.xAngle += ANGLE_INCREMENT
            elif motion == pyglet.window.key.MOTION_LEFT:
                self.zAngle -= ANGLE_INCREMENT
            elif motion == pyglet.window.key.MOTION_RIGHT:
                self.zAngle += ANGLE_INCREMENT

    def on_key_press(self, symbol, modifiers):  
        global uci, PLAY_AI_AI, PLOT_TRESHOLD, PLOT_IMG_YEL  

        if symbol == pyglet.window.key.ESCAPE:
            if RTB_ENABLED: 
                pyglet.clock.unschedule(move_robot)
            if USE_CAMERA: 
                if PLOT_TRESHOLD: 
                    pyglet.clock.unschedule(plot_treshold)
            if USE_CAMERA: 
                if PLOT_IMG_YEL: 
                    pyglet.clock.unschedule(plot_yellow)
            if PLAY_CHESS: 
                if PLAY_AI_AI: 
                    pyglet.clock.unschedule(AI_AI) 
            self.close() 
            pyglet.app.exit() 

        if USE_CAMERA:
            # Plot treshold
            if symbol == pyglet.window.key.F2:  
                PLOT_TRESHOLD = not PLOT_TRESHOLD   
                if PLOT_TRESHOLD:
                    pyglet.clock.schedule_interval(plot_treshold, dt)
                else:
                    pyglet.clock.unschedule(plot_treshold)

            if symbol == pyglet.window.key.F3:  
                PLOT_IMG_YEL = not PLOT_IMG_YEL 
                if PLOT_IMG_YEL:
                    pyglet.clock.schedule_interval(plot_yellow, dt)
                else:
                    pyglet.clock.unschedule(plot_yellow)

        if PLAY_CHESS:
            # Turn on AI vs AI game
            if symbol == pyglet.window.key.F4:   
                PLAY_AI_AI = not PLAY_AI_AI 
                if PLAY_AI_AI:
                    pyglet.clock.schedule_interval(AI_AI, 1)
                else:
                    pyglet.clock.unschedule(AI_AI)  

            # Submit recognized human move
            if symbol == pyglet.window.key.SPACE:
                if USE_CAMERA:
                    try:
                        uci = recognize_uci()
                        human_move = chess.Move.from_uci(uci)
                        full_move(human_move)
                    except ValueError:
                        print("Invalid uci (" + uci + "). Move piece again.")
                        uci = ''

            # Submit typed human move
            elif symbol == pyglet.window.key.ENTER and len(uci) == 4:
                try:
                    human_move = chess.Move.from_uci(uci)
                    full_move(human_move)
                except ValueError:
                    print("Invalid uci (" + uci + "). Type again.")
                    uci = ''

            # Type human uci            
            else:
                for key in list(key_dict.keys()):
                    if symbol == key_dict[key]:
                        uci += key

    def on_close(self):
        pyglet.clock.unschedule(move_robot)
        self.close()   
        pyglet.app.exit()
        # plot_speed()        

if __name__ == "__main__":
    Window(WINDOW_WIDTH, WINDOW_HEIGHT, 'EV3ARM simulation')
    pyglet.app.run()
