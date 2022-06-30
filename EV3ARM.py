#!/usr/bin/env python

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3

class EV3ARM(DHRobot):
    """
    Create model of EV3ARM manipulator

    Defined joint configurations are:
        . qz, zero joint angle configuration

    :notes:
        . LEGO units used (1 stud = 8mm)
        . Gear ratios included
    :references:
        . 7 axis robot made of LEGO Technic drived by EV3 servos, controlled by Linux based ststem [ev3dev](https://www.ev3dev.org/docs/getting-started/)

    """

    def __init__(self):

        twoPi = 2. * np.pi
        deg = np.pi/180
        stud = .008 # [m] - length of 1 stud
        m = 1
        inch = 0.0254
        SQUARE_WIDTH = 3*stud

        B = SE3(0*stud, 0*stud, 8*stud) * SE3.Ry(np.pi/2) * SE3.Rz(np.pi/2)
        
        # RotZ(fi) TrZ(d) TrX(a) RotX(alfa) 
        L0 = PrismaticDH(
            offset=0*stud,
            a=0,
            alpha=-90*deg,
            G=twoPi/(16/10*4*stud), # rad/studs 
            qlim=[-18*stud, 18*stud])

        L1 = RevoluteDH(
            offset=0*deg,
            d=22*stud,          # link length (Dennavit-Hartenberg notation)
            a=0,
            alpha=90*deg,         # link twist (Dennavit-Hartenberg notation)
            G=(140/12),            
            qlim=[-90*deg, 90*deg])    # minimum and maximum joint angle    

        L2 = RevoluteDH(
            offset=90*deg,
            d=0,
            a=22*stud,            # link offset (Dennavit-Hartenberg notation)
            alpha=0,
            G=-(60/12*36/12*20/12),      # gear ratio 60/20*40/12
            qlim=[-90*deg,0*deg]) # minimum and maximum joint angle    

        L3 = RevoluteDH(
            offset=-90*deg,   
            d=0*stud,
            a=0,
            alpha=-90*deg,
            G=-(60/12*36/12),            # gear ratio 60/20
            qlim=[-135*deg,0*deg]) # minimum and maximum joint angle    

        L4 = RevoluteDH(
            offset=0*deg,
            d=17*stud,          # link length (Dennavit-Hartenberg notation)
            a=0,
            alpha=90*deg,         # link twist (Dennavit-Hartenberg notation)
            G=(60/12),            
            qlim=[-90*deg, 90*deg])    # minimum and maximum joint angle  

        L5 = RevoluteDH(
            offset=0*deg,
            d=0*stud,
            a=0,
            alpha=-90*deg,         # link twist (Dennavit-Hartenberg notation)
            G=(60/12*40/24*20/12*20/12),           # gear ratio 60/20*20/12
            qlim=[-90*deg,0*deg])    # minimum and maximum joint angle    

        L6 = RevoluteDH(
            offset=-90*deg,
            d=22*stud,            # link length (Dennavit-Hartenberg notation)
            a=0,
            alpha=180*deg,
            G=(60/12),            # gear ratio 60/20
            qlim=[-90*deg, 90*deg])    # minimum and maximum joint angle   

        super().__init__(
            [L0, L1, L2, L3, L4, L5, L6],
            name="EV3ARM",
            manufacturer="mr_majczel, MOC Design Studio",
            base = B
            )        

        # zero angles, L shaped pose
        self.addconfiguration("qrL", np.array([0*m,        0*deg, 0*deg, 0*deg,    0*deg,  -90*deg, 0*deg, ])) # initial pose (all touch sensors pressed)
        self.addconfiguration("qv",  np.array([0*m,        0*deg, -30*deg, -30*deg,0*deg,  -75*deg, 0*deg, ])) # slightly sloped pose (all touch sensors not pressed)
        self.addconfiguration("qz",  np.array([0*m,        0*deg, 0*deg, -135*deg, 0*deg,  -45*deg, 0*deg]))   # embrional pose
        self.addconfiguration("q0",  np.array([0*m,        0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))   # horizontal pose with gripper down

        self.addconfiguration("q0A", np.array([+3.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0B", np.array([+2.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0C", np.array([+1.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0D", np.array([+0.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0E", np.array([-0.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0F", np.array([-1.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0G", np.array([-2.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0H", np.array([-3.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        # For removed pieces
        self.addconfiguration("q0I", np.array([-4.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))
        self.addconfiguration("q0J", np.array([-5.5*SQUARE_WIDTH,  0*deg, -90*deg, 0*deg,   0*deg, -90*deg, 0*deg]))

if __name__ == '__main__':

    robot = EV3ARM()
    print(robot)        