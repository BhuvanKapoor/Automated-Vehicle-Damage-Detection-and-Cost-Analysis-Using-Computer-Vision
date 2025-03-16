import pandas as pd

def estimate_repair_costs(car_brand, car_model, affected_parts):

    # Load repair cost data from the existing CSV file
    cost_data = pd.read_csv('Data\CarPart price.csv')
    
    repair_costs = []
    
    # Process each affected part
    processed_damages = set()  # To avoid duplicate entries
    
    for _, _, damage_type, part_name in affected_parts:          # x, y, damage_type, part_name
        # Create a unique key for this damage type + part combination
        damage_key = f"{damage_type}_{part_name}"
        
        if damage_key in processed_damages:
            continue
            
        processed_damages.add(damage_key)
        
        # Look up repair cost in the CSV
        matching_costs = cost_data[
            (cost_data['Car Brand'].str.lower() == car_brand.lower()) & 
            (cost_data['Car Model'].str.lower() == car_model.lower()) & 
            (cost_data['Damage Type'].str.lower() == damage_type.lower()) & 
            (cost_data['Damage Car Part'].str.lower() == part_name.lower())
        ]
        # If we have a match, use it
        if not matching_costs.empty:
            # Use the first matching entry's cost
            repair_cost = matching_costs.iloc[0]['Cost']
            
            # Add to repair costs list
            repair_costs.append({
                "Part": part_name.capitalize(),
                "Damage Type": damage_type.replace('_', ' ').capitalize(),
                "Repair Cost": f"{repair_cost:.2f}"
            })
        else:
            # If no match in the database at all, skip this part
            print(f"No cost data for {car_brand} {car_model}, {damage_type} on {part_name}")
    
    return repair_costs