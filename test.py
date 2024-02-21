from math import *


path = [(2, 2), (1, 2), (1, 3), (2, 3), (3, 3), (2, 3), (2, 4), (3, 5)]
cena = 0

for pos_idx in range(len(path)-2):
    angle = (round(atan2(path[pos_idx][0] - path[pos_idx+1][0], path[pos_idx][1] - path[pos_idx+1][1]) * 360 / pi / 45, 0) * 22.5) % 360 - 180
    angle1 = (round(atan2(path[pos_idx+1][0] - path[pos_idx+2][0], path[pos_idx+1][1] - path[pos_idx+2][1]) * 360 / pi / 45, 0) * 22.5) % 360 - 180
    
    if abs(angle - angle1) not in [180, 0]:
        cena+=1

print(cena)