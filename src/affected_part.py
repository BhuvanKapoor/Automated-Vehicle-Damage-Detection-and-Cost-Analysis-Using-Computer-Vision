import numpy as np

def get_affected_parts(damage_centers, parts_centers):
    affected_parts = []
    
    for damage_x, damage_y, _, damage_type in damage_centers:
        closest_part = None
        min_distance = float("inf")
        
        for part_x, part_y, part_name in parts_centers:
            distance = np.sqrt((damage_x - part_x) ** 2 + (damage_y - part_y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_part = part_name
        
        if closest_part:
            affected_parts.append((damage_x, damage_y, damage_type, closest_part))
    
    return affected_parts
