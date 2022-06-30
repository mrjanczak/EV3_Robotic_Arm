# Robot Arm with 3 calibration buttons, no gyro
# Prerequisites:
# - libusb0.dll - don't use ver. 1.0

import os
import pickle
import numpy as np
import math
import copy

from spatialmath import SE3
from spatialmath.base import *
import matplotlib.pyplot as plt

import pyglet
from pyglet import clock, shapes
import pyglet.window.key

import roboticstoolbox as rtb
import ev3_dc as ev3
import chess
import chess.engine

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# np.set_printoptions(suppress=True)

# np.set_printoptions(sign=' ', formatter={'int': '{: 4n}'.format})
deg = np.pi / 180. # deg > rad
rad = 180. / np.pi
stud = .008  # m/stud
twoPi = 2. * np.pi
stud_6deg = [1/stud,rad,rad,rad, rad,rad,rad]

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
    q7lim = math.asin(2*stud / A7)
    OPEN_GRIPPER  = [q7lim]
    CLOSE_GRIPPER = [0]

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

    LARGE_MOTOR_MAX_TORQUE  = .4 # Nm
    MEDIUM_MOTOR_MAX_TORQUE = 0.12 # Nm
    MAX_TORQUE = [LARGE_MOTOR_MAX_TORQUE, MEDIUM_MOTOR_MAX_TORQUE]
    RAMP_UP    = 0 #.25
    RAMP_DOWN  = 0 # .25
    MIN_DALF   = 5

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

    def set_zero_position(i=0):
        global motors
        if i == 0:
            for motor in motors:
                if motor:
                    motor.position = 0
        else:
            motors[7].position = 0
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
            motor.stop(brake=brakes)   
            print('brake = ',brakes, motor.port)

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
        V1 = 20
        V2 = 100
        speed_perc = np.array([V2,V2,V1,V1, V1,V2,V1,V1])

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

    print('\nInit chess board \n--------------------------')
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(os.getcwd() + '\stockfish.exe')
    margin_file = 8
    margin_rank = -1
    uci = ''

    def save_chess_poses():
    
        filename = 'chess_poses.dat'
        outfile = open(filename,'wb')
        pickle.dump(chess_poses, outfile)
        outfile.close()

        print("chess_poses saved in ", filename)

    def read_chess_poses(filename = 'chess_poses.dat'):
        global chess_poses

        if not os.path.isfile(filename):
            print('File EV3ARM.dat not found')
            return False
        else:
            file = open(filename,'rb')
            chess_poses = pickle.load(file)
            file.close()
            print('Data imported')

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
F_keys =    [pyglet.window.key.F1, pyglet.window.key.F2, pyglet.window.key.F3, pyglet.window.key.F4, pyglet.window.key.F5, pyglet.window.key.F6, pyglet.window.key.F7, pyglet.window.key.F8 ]
keys_up =   [pyglet.window.key._1, pyglet.window.key._2, pyglet.window.key._3, pyglet.window.key._4, pyglet.window.key._5, pyglet.window.key._6, pyglet.window.key._7, pyglet.window.key._8, ]
keys_down = [pyglet.window.key.Q,  pyglet.window.key.W,  pyglet.window.key.E,  pyglet.window.key.R,  pyglet.window.key.T,  pyglet.window.key.Y,  pyglet.window.key.U,  pyglet.window.key.I,  ]

shift_motor_by = 5 # deg
brakes = False
pyglet.clock.schedule_interval(move_robot_wrapper, dt)           

# ********* Window settings *********
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SCALE = 750
OFFSET = np.array([0,WINDOW_WIDTH/2,50]) # 

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, 'EV3 robot')
clock.schedule_interval(move_robot_wrapper, dt) 
batch = pyglet.graphics.Batch()
text1 = text2 = text3 = ''
def draw_robot_to_batch():
    global batch, robot

    Tall = robot.fkine_all().data
    # Links start & end points: P1, P2
    for i in range(len(Tall)):
        if i == 0:
            B = robot.base * 1
            P1 = B[:3,3] * SCALE + OFFSET
        else:
            P1 = Tall[i-1][:3,3] * SCALE + OFFSET
        P2 = Tall[i][:3,3] * SCALE + OFFSET 
       
        # Last link shorter by gripper    
        if i == len(Tall)-1:
            P2 = P2 + Tall[i][:3,2] * GRIP_LEN * SCALE           

        circle = shapes.Circle(P1[0], P1[1], 20, color=(255, 0, 0), batch=batch)
        line = shapes.Line(P1[0], P1[1], P2[0], P2[1], width=10, batch=batch)
        line = shapes.Line(100, 100, 100, 200, width=10, batch=batch)

# on draw event
@window.event
def on_draw():
    global text2,text3
	
    window.clear()

    Tall = robot.fkine_all()
    shp=[]
    # Links start & end points: P1, P2
    for i in range(len(Tall)):
        if i == 0:
            B = robot.base.A[:3,3] * SCALE + OFFSET
            P1 = B 
        else:
            P1 = Tall.A[i-1][:3,3] * SCALE + OFFSET

        P2 = Tall.A[i][:3,3] * SCALE + OFFSET 

        x = 1
        y = 2
        shp.append(shapes.Circle(P1[x], P1[y], 10, color=(255, 0, 0), batch=batch))
        shp.append(shapes.Line(P1[x], P1[y], P2[x], P2[y], width=5, batch=batch))

        for pose in chess_poses:
            T = robot.fkine(pose[0])
            P1 = T.A[:3,3] * SCALE + OFFSET
            shp.append(shapes.Circle(P1[x], P1[y], 10, color=(0, 255, 0), batch=batch))
            T = robot.fkine(pose[1])
            P1 = T.A[:3,3] * SCALE + OFFSET
            shp.append(shapes.Circle(P1[x], P1[y], 10, color=(0, 0, 255), batch=batch))

    batch.draw()  
    text1 = '1-8/Q-I - move robot; F1-F8(F9) - chess pose; shift/ctrl+F.. - save up/down pose; c - callibration; b - breaks, m - move'                   
    label1 = pyglet.text.Label( text1, font_size=10, x=10, y=WINDOW_HEIGHT-20, anchor_x='left', anchor_y='top').draw() 
    label2 = pyglet.text.Label( text2, font_size=10, x=10, y=WINDOW_HEIGHT-40, anchor_x='left', anchor_y='top').draw() 
    label3 = pyglet.text.Label( text3, font_size=10, x=10, y=WINDOW_HEIGHT-60, anchor_x='left', anchor_y='top').draw() 

@window.event
def on_key_press(symbol, modifier): 
    global uci, brakes, engine, shift_motor_by
    global robot, robot_q7, text2, text3
    global pieces_, q_chess, demo_moves

    if symbol == pyglet.window.key.C:   
        robot.q = robot.qS
        text2 = "Clibrate robot"
        calibrate_robot()

    if symbol == pyglet.window.key.B:  
        brakes = not brakes 
        text2 = "Set breaks to {}".format(brakes)
        set_break()

    if symbol == pyglet.window.key.M: 
        pieces_ = find_pieces_on_board(board)
        q_chess = find_chess_poses(chess_poses)
        
        print('Spanish opening')
        append_uci_move('e2e4')    
        # append_uci_move('e7e5')   
        # append_uci_move('g1f3')    
        # append_uci_move('b8c6')   
        # append_uci_move('f1b5')   

        while robot_moves:
            move_robot() 
        print('Move completed!')

    if symbol == pyglet.window.key.D:
        append_traj(demo_moves)
        print('Demo completed!')


    # Chess poses
    for i in range(8):                
        if symbol == F_keys[i]:
            if modifier & pyglet.window.key.MOD_SHIFT: 
                chess_poses[i][1] = copy.deepcopy(robot.q)
                text2 = "Set high pose over row {} ".format(i)
                save_chess_poses()

            elif modifier & pyglet.window.key.MOD_CTRL: 
                chess_poses[i][0] = copy.deepcopy(robot.q)
                text2 = "Set low pose over row {} ".format(i)
                save_chess_poses()
            
            else:
                append_traj([chess_poses[i][0]])
                text2 = "Move to low pose over row {} ".format(i)

    if symbol == pyglet.window.key.F9:
        text2 = "Come back to start pose"
        append_traj([robot.qS])

    if symbol == pyglet.window.key.PAGEUP:
        text2 = "Open gripper"
        append_traj([OPEN_GRIPPER])

    if symbol == pyglet.window.key.PAGEDOWN:
        text2 = "Close gripper"
        append_traj([CLOSE_GRIPPER])

    unit = [stud,deg,deg,deg, deg,deg,deg,deg]
    unit_s = ['stud','deg','deg','deg', 'deg','deg','deg','deg']

    # Move robot
    for i in range(7):
        if symbol == keys_up[i]:                    
            q = copy.deepcopy(robot.q)
            q[i] = q[i] + shift_motor_by*unit[i]
            text2 = "Joint {} shift by -{} {} to {}".format(i, shift_motor_by, unit_s[i], int(q[i]/unit[i]))
            text3 = "Pose {}".format(q/unit[:7])
            append_traj([q])

    for i in range(7):
        if symbol == keys_down[i]:                   
            q = copy.deepcopy(robot.q)
            q[i] = q[i] - shift_motor_by*unit[i]
            text2 = "Joint {} shift by +{} {} to {}".format(i, shift_motor_by, unit_s[i], int(q[i]/unit[i]))
            text3 = "Pose {}".format(q/unit[:7])
            append_traj([q])

    if symbol == keys_up[7]:
        robot_q7 = robot_q7 + shift_motor_by*unit[7]
        text2 = "Gripper shift by -{} {} to {}".format( shift_motor_by, unit_s[i], int(robot_q7/unit[7]))
        append_traj([[robot_q7]])
        set_zero_position(7)

    if symbol == keys_down[7]:
        robot_q7 = robot_q7 - shift_motor_by*unit[7]
        text2 = "Gripper shift by +{} {} to {}".format( shift_motor_by, unit_s[i], int(robot_q7/unit[7]))
        append_traj([[robot_q7]])
        set_zero_position(7)

    if symbol == pyglet.window.key.NUM_SUBTRACT:
        if shift_motor_by >= 10: 
            shift_motor_by = shift_motor_by - 5 # deg/stud
        else: 
            shift_motor_by = shift_motor_by - 1 # deg/stud            
        text2 = "Shift by {}".format(shift_motor_by)

    if symbol == pyglet.window.key.NUM_ADD:
        if shift_motor_by >= 5: 
            shift_motor_by = shift_motor_by + 5 # deg/stud
        else:
            shift_motor_by = shift_motor_by + 1
        text2 = "Shift by {}".format(shift_motor_by)

    if symbol == pyglet.window.key.ESCAPE:
        clock.unschedule(move_robot_wrapper)
        window.close()
        engine.quit()
        print('Exit pyglet')   

@window.event
def on_close(): 
    clock.unschedule(move_robot_wrapper)
    window.close()
    engine.quit()
    print('Exit pyglet')   

# save_chess_poses()
read_chess_poses()

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
   

pyglet.app.run()
