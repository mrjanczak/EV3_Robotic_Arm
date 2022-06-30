# Robot Arm with emergency button to stop any move

import os.path
import sys
import time
from datetime import datetime
import copy
import pickle
import numpy as np
import math
from numbers import Number
from collections import namedtuple

from pyglet.window.key import motion_string
from spatialmath import SE3
from spatialmath.base import *
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

from ev3_dc.functions import LCX
from ev3_dc.constants import (
    opOutput_Speed,
    opOutput_Start)

class ev3_Motor(ev3.Motor):
    
    def start_move_for2(
        self,
        duration: float,
        *,
        speed: int = None,
        direction: int = 1,
        ramp_up_time: float = None,
        ramp_down_time: float = None,
        brake: bool = False,
        _control: bool = False
    ) -> None:
        '''
        start moving the motor for a given duration.

        Mandatory positional arguments

          duration
            duration of the movement [sec.]

        Optional keyword only arguments

          speed
            percentage of maximum speed [1 - 100]
          direction
            direction of movement (-1 or 1)
          brake
            flag if ending with floating motor (False) or active brake (True).
        '''
        assert isinstance(duration, Number), \
            "duration must be a number"
        assert duration >= 0.001, \
            "duration must be at least one millisecond"

        assert speed is None or isinstance(speed, int), \
            'speed must be an int value'
        assert speed is None or 0 < speed and speed <= 100, \
            'speed  must be in range [1 - 100]'

        assert isinstance(direction, int), \
            'direction must be an int value'
        assert direction in (-1, 1), \
            'direction must be 1 (forwards) or -1 (backwards)'

        assert isinstance(brake, bool), \
            'brake must be a boolean'

        assert (
            self._current_movement is None or
            'stopped' in self._current_movement
        ), "concurrent movement in progress"

        if speed is None:
            speed = self._speed

        steady_ms = int(
                1000 *
                (duration )
        )

        ops = b''.join((
            opOutput_Speed,
            LCX(0),  # LAYER
            LCX(self._port),  # NOS
            LCX(direction * speed),  # SPEED

            opOutput_Start,
            LCX(0),  # LAYER
            LCX(self._port)  # NOS
        ))

        self.send_direct_cmd(ops)

        if _control:
            self._current_movement = {
                'op': 'Time_Speed',
                'duration': duration,
                'speed': speed,
                'direction': direction,
                'ramp_up_time': ramp_up_time,
                'ramp_down_time': ramp_down_time,
                'brake': brake,
                'started_at': datetime.now()
            }
        else:
            self._target_position = None
            self._current_movement = None

        
    def move_for2(
        self,
        duration: float,
        *,
        speed: int = None,
        direction: int = 1,
        ramp_up_time: float = None,
        ramp_down_time: float = None,
        brake: bool = False
    ) -> Task:
        '''
        start moving the motor for a given duration.

        Mandatory positional arguments

          duration
            duration of the movement [sec.]

        Optional keyword only arguments

          speed
            percentage of maximum speed [1 - 100]
          direction
            direction of movement (-1 or 1)
          ramp_up_time
            duration time for ramp-up [sec.]
          ramp_down_time
            duration time for ramp-down [sec.]
          brake
            flag if ending with floating motor (False) or active brake (True).

        Returns

          Task object, which can be started, stopped and continued.
        '''
        assert isinstance(duration, Number), \
            "duration must be a number"
        assert duration >= 0.001, \
            "duration must be at least one millisecond"

        assert speed is None or isinstance(speed, int), \
            'speed must be an int value'
        assert speed is None or 0 < speed and speed <= 100, \
            'speed  must be in range [1 - 100]'

        assert isinstance(direction, int), \
            'direction must be an int value'
        assert direction in (-1, 1), \
            'direction must be 1 (forwards) or -1 (backwards)'

        assert ramp_up_time is None or isinstance(ramp_up_time, Number), \
            "ramp_up_time must be a number"
        assert ramp_up_time is None or ramp_up_time >= 0, \
            "ramp_up_time must be positive"

        assert ramp_down_time is None or isinstance(ramp_down_time, Number), \
            "ramp_down_time must be a number"
        assert ramp_down_time is None or ramp_down_time >= 0, \
            "ramp_down_time must be positive"

        assert isinstance(brake, bool), \
            'brake must be a boolean'

        if speed is None:
            speed = self._speed
        if ramp_up_time is None:
            ramp_up_time = self._ramp_up_time
        if ramp_down_time is None:
            ramp_down_time = self._ramp_down_time

        return Task(
                self.start_move_for2,
                args=(duration,),
                kwargs={
                    'speed': speed,
                    'direction': direction,
                    'ramp_up_time': ramp_up_time,
                    'ramp_down_time': ramp_down_time,
                    'brake': brake,
                    '_control': True
                },
                duration=duration,
                action_stop=self.stop,
                kwargs_stop={'brake': False},
                action_cont=self.cont
        ) + Task(
                self._final_move_for
        )

np.set_printoptions(sign=' ', formatter={'int': '{: 4n}'.format})
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
# def gtraj(q0, qf, tv):
#     t = np.linspace(0, 1, tv)  # normalized time from 0 -> 1
#     q7 = np.linspace(q0, qf, tv)       
#     return namedtuple('gtraj', 'q7 t')(q7, t)

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
    USE_KEYBOARD    = False
    USE_TRAJECTRY   = True
    CALIBRATION     = False

    print('Init EV3ARM robot toolbox...')
    robot = rtb.models.DH.EV3ARM()
    robot.q = robot.qz
    Tall = robot.fkine_all().data   # Poses of all robot joints
    Tep = robot.fkine(robot.q)      # End-effector pose
    Pos = Tep.t / stud
    Alf = robot.q[2] + robot.q[3] + robot.q[5]
    print('Position of end-effector [stud]: ',Pos.astype(int), int(Alf*rad))

    v = np.zeros(6)                 # End-effector linear speed - 6dof
    dt = 0.2                        # Movement frame time
    
    max_speed_ratio = .1            # Max_speed ratio used for trajectory
    max_speed = np.ones(robot.n)*1000 # deg/sec
    max_torque = np.ones(robot.n)*.1      
    G = np.ones(robot.n)            # Gear ratios
    for i,link in enumerate(robot.links):
        if not np.isnan(link.G):
            G[i] = link.G
    G7 = -32/16                     # Gripper gear ratio
    G_ =  np.concatenate((G, [G7]), axis=0)
    A7 = 4*stud                     # Gripper finger length
    robot_q7 = 0                    # Position of gripper
    robot_qd7 = 0                   # Speed of gripper
    ROLL = robot.n-1
    GRIP = robot.n  
    GRIP_LEN = 2*stud
    GRIP_RAD = .5*stud

    TRANS_INCREMENT = 1*stud
    ANGLE_INCREMENT = 5 #deg
    ACC_TIME = 2*dt

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
    robot_moves = []                # trajectory of steps
    step = 1      
    move_completed = True   

    # Diagnostic
    speed_ = np.array([np.zeros(robot.n+1)]) 
    time_ = [0]
    q_chess = None                  # Chess board squares poses
    f = open("log.dat","w")
    f.write('--- gyro calib ---')   
    f.close()

    # Append trajectory to robot_moves
    def append_traj(robot_moves_def):
        global robot_moves
        for move_def in robot_moves_def:
            # Move of robot or gripper
            if len(move_def[0]) == robot.n:
                max_speed_rad = max_speed[:robot.n] * rad
                qdmax = np.divide(max_speed_rad, G)
            else:
                qdmax = np.array([max_speed[robot.n] * rad / G7])

            move = rtb.trajectory.mstraj(
                np.array(move_def), 
                dt, 
                ACC_TIME, 
                qdmax = qdmax,
                verbose = False,)
            robot_moves.append(move) 

    def move_robot_wrapper(dt):
        """
        Execute move_robot() if previous move is completed
            dt : time step
        """
        if move_completed and not CALIBRATION:
            move_robot()

    def move_robot():
        """
        Move robot according to end effector speed v or trajectory (sequence of movements) 
        in dt time steps as scheduled in pyglet.clock (Window.__init__)
        """
        global v, robot, Tep, robot_q7, robot_qd7, robot_moves, step, start_time, end_time, delay_time
        global time_, speed_, pieces_, emergency_stop, move_completed
        
        move_completed = False
        finish_move = False

        # set q,qd from v 
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
                
        # set q,qd from trajectory (v = robot.jacobe0() @ robot.qd)
        if USE_TRAJECTRY:
            robot.qd = np.zeros(robot.n)
            robot_qd7 = 0

            # If not empty robot_moves
            robot_moves_len = len(robot_moves)
            if robot_moves_len > 0:
                move = robot_moves[0]
                move_steps_len = len(move.t)     

                # Trajectory of body
                if len(move.q[step]) == robot.n:                          
                    robot.q = move.q[step]
                    robot.qd = (move.q[step] - move.q[step-1]) / dt

                # Trajectory of gripper              
                if len(move.q[step]) == 1:
                    robot_q7 = move.q[step].item()
                    robot_qd7 = (move.q[step].item() - move.q[step-1].item()) / dt

                step += 1                
                if step == move_steps_len:  
                    finish_move = True 
                    # Remove finished move from begining of the list
                    robot_moves = robot_moves[1:robot_moves_len]
                    step = 1
                    robot.qd = np.zeros(robot.n)
                    robot_qd7 = 0

                    Tep = robot.fkine(robot.q)
                    Pos = Tep.t / stud
                    Alf = robot.q[2] + robot.q[3] + robot.q[5]
                    print('Position of end-effector [stud]: ',Pos.astype(int), int(Alf*rad))

        # Movement of EV3 real robot 
        if EV3_ENABLED:
            end_time = time.time()
            delay_time = end_time - start_time - dt       
            start_time = end_time
            dt_corr =  dt # + delay_time

            # RTB robot position including initial offset (corresponding to EV3 motors positions)
            robot_q_off  = (robot.q - robot.qz)
            alf   = np.concatenate((robot_q_off * G, [robot_q7 * G7 + robot_q_off[ROLL]]), axis=0) * rad # deg         
            # speed = np.concatenate((robot.qd * G, [robot_qd7 * G7 + robot.qd[ROLL]]), axis=0) * rad # deg/sec
                
            # Get current positions Alf0 
            alf0 = []
            for motor in motors:
                position = 0
                if motor is not None:
                    position = motor.position
                alf0.append(position) # deg
            alf0 = np.array(alf0).astype(int)
            dalf = alf - alf0
            speed = dalf / dt_corr # deg/sec

            ramp_down = np.around(np.absolute(dalf) * RAMP_DOWN).astype(int)
            ramp_down = ramp_down + (ramp_down == 0).astype(int)
            ramp_down = ramp_down.tolist()

            direction = np.sign(speed).astype(int).tolist()
            speed = np.abs(speed)

            # Normalize percent speed
            speed_perc = speed / max_speed * 100 # [%]
            max_speed_perc = np.max(speed_perc)
            # If needed, reduce to 100%
            if max_speed_perc > 100 * max_speed_ratio:
                speed_perc = speed_perc / max_speed_perc * 100 * max_speed_ratio

            t_list = list()
            # Lists of integer inputs
            alf = alf.astype(int).tolist()
            speed_perc = speed_perc.astype(int).tolist()
            
            start_task = True
            for i in range(robot.n + 1):
                motor = motors[i]
                if motor is not None:
                    if speed_perc[i] == 0:
                        task = motor.stop_as_task(brake = True)
                    else:            
                        if not finish_move:
                            task = motor.move_for2(dt_corr, speed = speed_perc[i], direction = direction[i], brake = False) 
                        else:
                            task = motor.move_to(alf[i], speed = speed_perc[i], ramp_up = 0, ramp_down = ramp_down[i], brake = True)                 
                    t_list.append(task)
                else:
                    start_task = False
                    print("Some motors are inactive")

            if start_task:                    
                Periodic(touch_dt, emergency_button, args=(t_list,)).start()
                for t in t_list:
                    t.start()                
                for t in t_list:
                    t.join()
                while emergency_stop:
                    time.sleep(touch_dt)  
            else:
                time.sleep(dt)

            # Print move
            if robot_moves_len > 0:
                timer0 = time.time() - start0_time
                # Verify true rotation of arm
                q2 = robot_q_off[2]*rad
                q3 = robot_q_off[3]*rad
                q_true = 0
                if gyro_sensor  is not None:
                    q_true = gyro_sensor.angle 
                print("t{:4.2f} {} {}   {}".format(timer0, int(q2), int(q3), q_true)) 

                time_.append(timer0)
                speed_ = np.append(speed_, [speed], axis = 0) 
                f = open("log.dat", "a")
                f.write("t{:4.2f} {} {}   {} \n".format(timer0, int(q2), int(q3), q_true)) 
                # f.write("t{:4.2f} {:4.2f} a0{} a{} sp{} d{} \n".format(timer0, delay_time, alf0, alf, speed_perc, direction) )
                # print("t{:4.2f} {:4.2f} a0{} a{} sp{} d{}".format(timer0, delay_time, alf0, alf, speed_perc, direction) )
                f.close()  
                
        # Move virtual pieces by gripper 
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

        # capture video
        # ret, img = cap.read()
        # cv2.imshow('img', img) 

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
    except:
        ev3_1 = None
        print("No device EV3_1 with MAC " + EV3_1_MAC)

    try:
        ev3_2 = ev3.EV3(protocol=ev3.USB, host=EV3_2_MAC)
    except:
        ev3_2 = None
        print("No device EV3_2 with MAC " + EV3_2_MAC)

    # type of motor (0: EV3-Large, 1: EV3-Medium, )
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
    LARGE_MOTOR_MAX_SPEED = 1050 # deg/s
    MEDIUM_MOTOR_MAX_SPEED = 1560 # deg/s
    MAX_SPEED = [LARGE_MOTOR_MAX_SPEED, MEDIUM_MOTOR_MAX_SPEED]

    LARGE_MOTOR_MAX_TORQUE  = .4 # Nm
    MEDIUM_MOTOR_MAX_TORQUE = 0.12 # Nm
    MAX_TORQUE = [LARGE_MOTOR_MAX_TORQUE, MEDIUM_MOTOR_MAX_TORQUE]
    RAMP_UP = .25
    RAMP_DOWN = .25

    TOUCH_SENSOR = (ev3.PORT_1, ev3_1)
    GYRO_SENSOR = (ev3.PORT_4, ev3_2)    
    # COLOR_SENSOR = (ev3.PORT_4, ev3_2)

    motors = [] 
    max_speed = []
    max_torque = []
    for motor_def in MOTORS_DEF:
        port, obj = motor_def
        print('connect to port ',port, obj)
        motor, type_ = (None, 7)
        if obj is not None:
            try:
                motor = ev3.Motor(port, ev3_obj = obj )
                motor.verbosity = 0
                motor.delta_time = dt/2
                motor.stop(brake=True)   
                motor.position = 0
                type_ = motor.motor_type    # 7: EV3-Large, 8: EV3-Medium
            except:
                print("No motor in port/dev " + port + obj) 

        motors.append(motor)
        max_speed.append(MAX_SPEED[type_ - 7])
        max_torque.append(MAX_TORQUE[type_ - 7])

    max_speed = np.array(max_speed)
    max_torque = np.array(max_torque)

    port, obj = GYRO_SENSOR
    qd_true, q_true = (0, 0)
    try:
        gyro_sensor = ev3.Gyro(port, ev3_obj = obj)
        # Reset
        qd_true = gyro_sensor.rate
        q_true = gyro_sensor.angle
    except:
        gyro_sensor = None
        print("No Gyro sensor")

    f = open("log.dat","a")
    f.write("t{} {} {}   {} {} \n".format(0, 0, 0, q_true, qd_true))     
    f.close()    

    port, obj = TOUCH_SENSOR
    try:
        touch_sensor = ev3.Touch(port, ev3_obj = obj)
    except:
        touch_sensor = None
        print("No Touch sensor")

    touch_now = False
    touch_prev = False
    emergency_stop = False
    touch_dt = 0.1                  # Touch sensor check time

    def emergency_button(t_list):
        """
        Stop & continue robot move when button pressed
        """
        global emergency_stop, touch_prev, touch_now

        touch_prev = touch_now
        if touch_sensor:
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

    def calibrate_robot(i, dalf):
        dalf = dalf*rad # deg
        # print('Calibrate motor {} by {}deg'.format(i,dalf))
        motor = motors[i]
        if motor is not None:
            motor.start_move_by(int(dalf), brake = True)
            motor.position = 0

# ********* Chess play settings *********
PLAY_CHESS = False
if PLAY_CHESS:
    # Chess board settings
    PLAY_AI_AI = False
    BOARD_POS = np.array([-12*stud, 16*stud, 2*stud])
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

    print('\nInit chess board \n--------------------------')
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

    def append_piece_move(i0, j0, i1, j1, piece_type, mode=0, i00=0, j00=0):
        """
        Define sequence of movements of robot to move piece based on 2 or 3 chess board squares 
            i0, j0 : move from square (file, rank - integers in range(0:7))
            i1, j1 : move to square (file, rank as integers in range(0:7))
            mode : 0 - move w/o capturing, 1 - move of captured piece, 2 - move of capturing piece
            i00, j00 : start_time from end of prev. move (file, rank - integers in range(0:7))
        """
        global robot_moves

        q7lim = math.asin(2*stud / A7)
        OPEN_GRIPPER  = [[0], [q7lim]]
        CLOSE_GRIPPER = [[q7lim], [0]]

        up = PIECE_TYPE_LEV[0]
        down = PIECE_TYPE_LEV[piece_type]
        qz = robot.qz

        # Move w/o capturing
        if mode == 0:
            robot_moves_def = [
                [qz, q_chess[i0][j0][up]], 
                OPEN_GRIPPER,
                [q_chess[i0][j0][up], q_chess[i0][j0][down]], 
                CLOSE_GRIPPER,
                [q_chess[i0][j0][down], q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down]],
                OPEN_GRIPPER,
                [q_chess[i1][j1][down], q_chess[i1][j1][up], qz],
                CLOSE_GRIPPER  ]

        # Move Captured piece to margin
        elif mode == 1:
            robot_moves_def = [
                [qz, q_chess[i0][j0][up]], 
                OPEN_GRIPPER,
                [q_chess[i0][j0][up], q_chess[i0][j0][down]], 
                CLOSE_GRIPPER,
                [q_chess[i0][j0][down], q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down]],
                OPEN_GRIPPER,
                [q_chess[i1][j1][down], q_chess[i1][j1][up]]  ]

        # Move Capturing piece to new square
        elif mode == 2:
            robot_moves_def = [
                [q_chess[i00][j00][up], q_chess[i0][j0][up], q_chess[i0][j0][down]], 
                CLOSE_GRIPPER,
                [q_chess[i0][j0][down], q_chess[i0][j0][up], q_chess[i1][j1][up], q_chess[i1][j1][down]],
                OPEN_GRIPPER,
                [q_chess[i1][j1][down], q_chess[i1][j1][up], qz],
                CLOSE_GRIPPER  ]

        append_traj(robot_moves_def)    

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
PLOT_TRESHOLD = False
PLOT_IMG_YEL = False
if USE_CAMERA:    

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

    # Setup camera
    cap = cv2.VideoCapture(CAMERA) # ,cv2.CAP_DSHOW
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Board ROI
    fw, fh, w, h, m = FRAME_WIDTH, FRAME_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT, SQUARE_MARG
    x1,y1,x2,y2 = (int((fw-w)/2),int((fh-h)/2),int((fw+w)/2),int((h+h)/2))
    l1,l2,l3, u1,u2,u3 = YELLOW_HSV

    # Prepare edges mask
    mask_e = np.zeros([h,w])
    for r in range(8):
        for c in range(8):
            mask_e[int(r*h/8.) + m : int((r+1)*h/8.) - m, int(c*w/8.) + m : int((c+1)*w/8.) - m] = 255
    mask_e = mask_e.astype(np.uint8)

    # Heat up camera :)
    for i in range(30):
        ret, img = cap.read()

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

    # Create a plotter class object
    plot = Plotter(200, 400, COLOR_TRESHOLD * 100 * PLOT_SCALE)  

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
                cropped = img_yel[int(r * h/8.) + SQUARE_MARG: int((r+1) * h/8.) - SQUARE_MARG, int(c * w/8.) + SQUARE_MARG: int((c+1) * w/8.) - SQUARE_MARG]
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


# ********* Main program *********
start0_time = time.time()
start_time = time.time()

# append_piece_move(3, 1, 3, 2,  1)

qz = robot.qz
q0 = robot.q0
qv = robot.qv
qz2 = qz + np.array([0, 0, -30*deg, 0, 0, 0, 0])
qz3 = qz + np.array([0, 0, 0, -30*deg, 0, 0, 0])

dT = SE3( 0, 5*stud, 0 )
T = robot.fkine(robot.q) * dT     # End-effector pose
qz2, success, err = robot.ikcon(T, q0=qz)   # solve IK, ignore additional outputs  

robot_moves_def = [
    [qz, qz2], 
    [qz2, qz],
    # [qz, qz3], 
    # [qz3, qz],    
]
append_traj(robot_moves_def)
   

class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        global start0_time, start_time

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

        pyglet.clock.schedule_interval(move_robot_wrapper, dt/10)          
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
        global dt, start_time, max_speed_ratio

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

        # if RTB_ENABLED and SCALE_DT:
        #     timer = time.time() - start_time
        #     dt_scaled = dt * max_speed_ratio     
        #     if timer < dt_scaled:
        #         time.sleep(dt_scaled - timer)       
        
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

        print('Pressed ', motion)

        if USE_KEYBOARD:
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
        global uci, PLAY_AI_AI, PLOT_TRESHOLD, PLOT_IMG_YEL, CALIBRATION

        if symbol == pyglet.window.key.ESCAPE:
            pyglet.clock.unschedule(move_robot)
            if USE_CAMERA: 
                if PLOT_TRESHOLD: 
                    pyglet.clock.unschedule(plot_treshold)
                if PLOT_IMG_YEL: 
                    pyglet.clock.unschedule(plot_yellow)
            if PLAY_CHESS and PLAY_AI_AI: 
                pyglet.clock.unschedule(AI_AI) 
            self.close() 
            pyglet.app.exit() 
            print('Exit pyglet')

        if symbol == pyglet.window.key.F1:   
            CALIBRATION = not CALIBRATION             

        if CALIBRATION:
            DALF = [.5*stud, 5*deg, 5*deg, 5*deg,   5*deg, 5*deg, 5*deg, 5*deg]
            keys_up = [pyglet.window.key._1, pyglet.window.key._2, pyglet.window.key._3, pyglet.window.key._4, 
                       pyglet.window.key._5, pyglet.window.key._6, pyglet.window.key._7, pyglet.window.key._8]
            keys_down = [pyglet.window.key.Q, pyglet.window.key.W, pyglet.window.key.E, pyglet.window.key.R, 
                         pyglet.window.key.T, pyglet.window.key.Y, pyglet.window.key.U, pyglet.window.key.I]

            for i in range(8):
                if symbol == keys_up[i]:
                    calibrate_robot(i, DALF[i] * G_[i])
            for i in range(8):
                if symbol == keys_down[i]:
                    calibrate_robot(i, -DALF[i] * G_[i])

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
        plot_data()        

if __name__ == "__main__":
    Window(WINDOW_WIDTH, WINDOW_HEIGHT, 'EV3ARM simulation')
    pyglet.app.run()
