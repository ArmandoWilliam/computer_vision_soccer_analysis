import time
import numpy as np

def check_possession(player_coordinates, ball_center, distance_threshold):
    result = 0
    
    x_min, y_min = np.min(player_coordinates, axis=0)
    x_max, y_max = np.max(player_coordinates, axis=0)
    bottom_left = (x_min, y_min)
    bottom_right = (x_max, y_min)
    dist_to_left_corner = np.linalg.norm(np.array(ball_center) - np.array(bottom_left))
    dist_to_right_corner = np.linalg.norm(np.array(ball_center) - np.array(bottom_right))
    if dist_to_left_corner <= distance_threshold or dist_to_right_corner <= distance_threshold:
        result = 1
    else:
        result = 0
            
    return result

def possession_timer_team_A(found_possessor_team_A, start_time_possession_team_A, total_possession_time_team_A):
    print("found possessor of the team A")
    found_possessor_team_A = 1
    if start_time_possession_team_A is None:
        start_time_possession_team_A = time.time()
        print("possession of the team A started: ", total_possession_time_team_A)
    else:
        total_possession_time_team_A += time.time() - start_time_possession_team_A
        print("Total time possession team A: ", total_possession_time_team_A)
    
    return total_possession_time_team_A, start_time_possession_team_A, found_possessor_team_A

def possession_timer_team_B(found_possessor_team_B, start_time_possession_team_B, total_possession_time_team_B):
    # if (found_possessor_team_A == 0) and (found_possessor_team_B == 0):
    print("found possessor of the team B")
    found_possessor_team_B = 1
    if start_time_possession_team_B is None:
        start_time_possession_team_B = time.time()
        print("possession of the team B started: ", total_possession_time_team_B)
    else:
        total_possession_time_team_B += time.time() - start_time_possession_team_B
        print("Total time possession team A: ", total_possession_time_team_B)
    
    return total_possession_time_team_B, start_time_possession_team_B, found_possessor_team_B

def possession_change_team_A(start_time_possession_team_A, total_possession_time_team_A):
    stop_time_possession_team_A = time.time()
    total_possession_time_team_A += stop_time_possession_team_A - start_time_possession_team_A
    print("Team A lost possession, total possession time team A: ", total_possession_time_team_A)
    start_time_possession_team_A = None
    
    return start_time_possession_team_A, total_possession_time_team_A

def possession_change_team_B(start_time_possession_team_B, total_possession_time_team_B):
    stop_time_possession_team_B = time.time()
    total_possession_time_team_B += stop_time_possession_team_B - start_time_possession_team_B
    print("Team B lost possession, total possession time team B: ", total_possession_time_team_B)
    start_time_possession_team_B = None
    
    return start_time_possession_team_B, total_possession_time_team_B