import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# from damage_detection import generate_damage_mask
from src.damage_detection import generate_damage_mask
from src.car_part_detection import detect_car_parts
from src.affected_part import get_affected_parts
from src.repair_cost_estimator import estimate_repair_costs

# Set page configuration
st.set_page_config(page_title="Car Damage Assessment Tool", layout="wide")

def main():
    st.title("Car Damage Assessment & Repair Cost Estimator")
    
    # Sidebar with form inputs
    with st.sidebar:
        st.header("Vehicle Information")
        
        # Create a form
        with st.form("car_info_form"):
            # Car brand dropdown (with some popular brands)
            car_brands = ["Toyota", "Honda", "Ford", "Hyundai", "Volkswagen", 
                         "BMW", "Mercedes", "Audi", "Maruti Suzuki", "Chevrolet", "Nissan", "Other"]
            car_brand = st.selectbox("Car Brand", car_brands)
            
            # If "Other" is selected, allow custom input
            if car_brand == "Other":
                car_brand = st.text_input("Enter Car Brand")
                
            # Car model text input
            car_model = st.text_input("Car Model")
            
            # Image upload
            uploaded_image = st.file_uploader("Upload Image of Car Damage", type=["jpg", "jpeg", "png"])
            
            # Submit button
            submit_button = st.form_submit_button("Analyze Damage")
    
    # Main content area
    if submit_button and uploaded_image is not None and car_brand and car_model:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        
        # Create two columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        
        with st.spinner("Analyzing damage..."):
            # Generate damage mask (using YOLO model)
            masked_image, damage_centers = generate_damage_mask(image)
            _, parts_centers = detect_car_parts(image)
            affected_parts = get_affected_parts(damage_centers, parts_centers)
            
            # Create a visualization of the damage
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(np.array(image))
            
            ax.imshow(masked_image)
            
            ax.set_title("Damage Detection")
            ax.axis('off')
            
            # Display the damage visualization
            with col2:
                st.subheader("Damage Detection")
                st.pyplot(fig)
        
        # Calculate repair costs
        repair_costs = estimate_repair_costs(car_brand, car_model, affected_parts)
        # Display the repair cost table
        st.subheader(f"Repair Cost Estimate for {car_brand} {car_model}")
        
        if repair_costs:
            # Create DataFrame for table
            df = pd.DataFrame(repair_costs)
            
            # Convert cost strings to numeric values for summing
            total_cost = sum([float(cost["Repair Cost"]) for cost in repair_costs])
            
            # Display table
            st.dataframe(df, hide_index=True)
            
            # Display total cost
            st.subheader(f"Total Estimated Repair Cost: ₹{total_cost:.2f}")
            
            # Add disclaimer
            st.caption("Note: This is an automated estimate. Actual repair costs may vary based on " +
                      "inspection by a qualified mechanic and availability of parts.")
        else:
            st.info("No significant damage detected that requires repair.")
    
    elif submit_button:
        if not uploaded_image:
            st.warning("Please upload an image of the car damage.")
        if not car_brand or car_brand == "":
            st.warning("Please select or enter a car brand.")
        if not car_model:
            st.warning("Please enter a car model.")

if __name__ == "__main__":
    main()
    