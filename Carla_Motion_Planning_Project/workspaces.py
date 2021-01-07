# import os 
# import os.path as osp
import polytope as pc 
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np

# def config():

#     WO1_1 = pc.box2poly([[1,2],[1,5]])
#     WO1_2 = pc.box2poly([[3,4],[4,12]])
#     WO1_3 = pc.box2poly([[3,12],[12,13]])
#     WO1_4 = pc.box2poly([[12,13],[5,13]])
#     WO1_5 = pc.box2poly([[6,12],[5,6]])

#     WO2_1 = pc.box2poly([[-6,25],[-6,-5]])
#     WO2_2 = pc.box2poly([[-6,30],[5,6]])
#     WO2_3 = pc.box2poly([[-6,-5],[-5,5]])
#     WO2_4 = pc.box2poly([[4,5],[-5,1]])
#     WO2_5 = pc.box2poly([[9,10],[0,5]])
#     WO2_6 = pc.box2poly([[14,15],[-5,1]])
#     WO2_7 = pc.box2poly([[19,20],[0,5]])
#     WO2_8 = pc.box2poly([[24,25],[-5,1]])
#     WO2_9 = pc.box2poly([[29,30],[0,5]])  

#     WO3_1 = pc.box2poly([[3.5,4.5],[0.5,1.5]])
#     WO3_2 = pc.box2poly([[6.5,7.5],[-1.5,-0.5]])

#     DEFAULT_TEST_CONFIG = {
#             'WO1' : [WO1_1, WO1_2, WO1_3, WO1_4, WO1_5],
#             'WO2': [WO2_1, WO2_2, WO2_3, WO2_4, WO2_5, WO2_6, WO2_7,WO2_8, WO2_9],
#             'WO3' : [WO3_1, WO3_2],
#             'start_pos': np.array([[0], [0]]),
#             'WO1_goal': np.array([[10], [10]]),
#             'WO2_goal': np.array([[35], [0]]),
#             'WO3_goal': np.array([[10], [0]])
#     }
#     return DEFAULT_TEST_CONFIG


def config():

    WO1_1 = Polygon([[-1, 1], [1, 1], [1, 5], [-1, 5]])
    WO1_2 = Polygon([[2, 4], [3, 4], [3, 12], [2, 12]])
    WO1_3 = Polygon([[3, 12], [12, 12], [12, 13], [3, 13], [3, 12]])
    WO1_4 = Polygon([[12, 5], [13, 5], [13, 13], [12, 13], [12, 5]])
    WO1_5 = Polygon([[6, 5], [12, 5], [12, 6], [6, 6], [6, 5]])

    WO2_1 = Polygon([[-6, -6], [25, -6], [25, -5], [-6, -5]])
    WO2_2 = Polygon([[-6, 5], [30, 5], [30, 6], [-6, 6]])
    WO2_3 = Polygon([[-6, -5], [-5, -5], [-5, 5], [-6, 5]])
    # WO2_10 = Polygon([[-6, -5], [5, -5], [5, -6], [-6, -6]])
    WO2_4 = Polygon([[4, -5], [5, -5], [5, -0.75], [4, -0.75]])
    WO2_5 = Polygon([[9, 1.5], [10, 1.5], [10, 4], [9, 4]])
    WO2_6 = Polygon([[14, -5], [15, -5], [15, -0.75], [14, -0.75]])
    WO2_7 = Polygon([[19, 1.5], [20, 1.5], [20, 4], [19, 4]])
    WO2_8 = Polygon([[24, -5], [25, -5], [25, -0.75], [24, -0.75]])
    WO2_9 = Polygon([[29, 1.5], [30, 1.5], [30, 4], [29, 4]])  
    

    WO3_1 = Polygon([[3.5, 1.5], [4.5, 1.5], [4.5, 0.5], [3.5, 0.5], [3.5, 1.5, [3.5, 1.5]]])
    WO3_2 = Polygon([[6.5, -0.5], [7.5, -0.5], [7.5, -1.5], [6.5, -1.5], [6.5, -1.5], [6.5, -0.5]])

    WO4_1 = Polygon([[-0.25, 1.1], [-0.25, 2], [0.25, 2], [0.25, 1.1]])
    WO4_2 = Polygon([[-2, -0.5], [-2, -0.3], [2, -0.3], [2, -0.5]])
    

    DEFAULT_TEST_CONFIG = {
            'WO1' : MultiPolygon([WO1_1, WO1_2, WO1_3, WO1_4, WO1_5]),
            'WO2': MultiPolygon([WO2_1, WO2_2, WO2_3, WO2_4, WO2_5, WO2_6, WO2_7,WO2_8, WO2_9]),
            'WO3' : MultiPolygon([WO3_1, WO3_2]),
            'WO4' : MultiPolygon([WO4_1, WO4_2]),
        #     'WO5' : MultiPolygon([WO5_1, WO5_2, WO5_3, WO5_4, WO5_5, WO5_6, WO5_7, WO5_8, WO5_9,\
        #              WO5_10, WO5_11, WO5_12, WO5_13, WO5_14, WO5_15, WO5_16, WO5_17, WO5_18, WO5_19, WO5_20]),
            'start_pos': np.array([[0], [0]]),
            'manip_start_pos': np.array([[2], [0]]),
            'WO1_goal': np.array([[10], [10]]),
            'WO2_goal': np.array([[35], [0]]),
            'WO3_goal': np.array([[10], [0]]),
            'manip_goal_pos': np.array([[-2.0], [0]]),

    }
    return DEFAULT_TEST_CONFIG


