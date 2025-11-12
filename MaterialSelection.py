import pandas as pd
import numpy as np

def find_best_material(csv_file_path, target_DampingRatio, target_spring_constant):
    
    # Read material database
    materials_df = pd.read_csv("Materials.csv")
    
    best_match = None
    best_score = -1
    
    for _, material in materials_df.iterrows():
        # Calculate damping match score
        damping_diff = abs(material['DampingRatio'] - target_DampingRatio)
        damping_score = 1 / (1 + damping_diff)
        
        # Calculate spring constant match score (using Young's modulus as proxy)
        material_stiffness = material['youngs_modulus'] * 1e9  # Convert GPa to N/m
        spring_diff = abs(material_stiffness - target_spring_constant) / target_spring_constant
        spring_score = 1 / (1 + spring_diff)
        
        # Combined score (equal weighting)
        total_score = 0.5 * damping_score + 0.5 * spring_score
        
        if total_score > best_score:
            best_score = total_score
            best_match = {
                'name': material['name'],
                'youngs_modulus': material['youngs_modulus'],
                'DampingRatio': material['DampingRatio'],
                'equivalent_spring_constant': material_stiffness,
                'damping_score': damping_score,
                'spring_score': spring_score,
                'total_score': total_score
            }
    
    return best_match

# Example usage:
if __name__ == "__main__":
    

    # Find best match
    TargetDampingRatio = 0.3
    TargetSpringCoEff = 5  # 100 kN/m
    
    best_material = find_best_material('Materials.csv', TargetDampingRatio, TargetSpringCoEff)
    
    print(f"   BEST MATERIAL MATCH: {best_material['name']}")
    print(f"   Damping Ratio: {best_material['DampingRatio']:.3f} (target: {TargetDampingRatio})")
    print(f"   Young's Modulus: {best_material['youngs_modulus']} GPa")
    print(f"   Equivalent Spring Constant: {best_material['equivalent_spring_constant']:.2e} N/m")
    print(f"   Match Score: {best_material['total_score']:.3f}")