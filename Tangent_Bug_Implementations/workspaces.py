# import os 
# import os.path as osp
import polytope as pc 
import numpy as np

def config():

    WO1_1 = pc.box2poly([[1,2],[1,5]])
    WO1_2 = pc.box2poly([[3,4],[4,12]])
    WO1_3 = pc.box2poly([[3,12],[12,13]])
    WO1_4 = pc.box2poly([[12,13],[5,13]])
    WO1_5 = pc.box2poly([[6,12],[5,6]])

    WO2_1 = pc.box2poly([[-6,25],[-6,-5]])
    WO2_2 = pc.box2poly([[-6,30],[5,6]])
    WO2_3 = pc.box2poly([[-6,-5],[-5,5]])
    WO2_4 = pc.box2poly([[4,5],[-5,1]])
    WO2_5 = pc.box2poly([[9,10],[0,5]])
    WO2_6 = pc.box2poly([[14,15],[-5,1]])
    WO2_7 = pc.box2poly([[19,20],[0,5]])
    WO2_8 = pc.box2poly([[24,25],[-5,1]])
    WO2_9 = pc.box2poly([[29,30],[0,5]])  

    DEFAULT_TEST_CONFIG = {
            'WO1' : [WO1_1, WO1_2, WO1_3, WO1_4, WO1_5],
            'WO2': [WO2_1, WO2_2, WO2_3, WO2_4, WO2_5, WO2_6, WO2_7,WO2_8, WO2_9],
            'start_pos': np.array([[0], [0]]),
            'WO1_goal': np.array([[10], [10]]),
            'WO2_goal': np.array([[35], [0]])
    }
    return DEFAULT_TEST_CONFIG
