import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

def calculate_medically_consistent_iss(gcs, sbp, resp_rate, age, injury_type, injury_text):
    """Calculate ISS score based on medical logic"""
    # Base ISS calculation based on physiological parameters
    base_iss = 25  # Start with moderate injury
    
    # Adjust based on GCS (Glasgow Coma Scale)
    if gcs <= 8:
        base_iss += 20  # Severe brain injury
    elif gcs <= 12:
        base_iss += 10  # Moderate brain injury
    elif gcs <= 14:
        base_iss += 5   # Mild brain injury
    
    # Adjust based on systolic blood pressure
    if sbp < 90:
        base_iss += 15  # Hypotension indicates severe injury
    elif sbp < 110:
        base_iss += 8   # Low blood pressure
    elif sbp > 160:
        base_iss += 3   # Hypertension
    
    # Adjust based on respiratory rate
    if resp_rate > 30:
        base_iss += 12  # Tachypnea indicates respiratory distress
    elif resp_rate > 25:
        base_iss += 8
    elif resp_rate < 10:
        base_iss += 10  # Bradypnea indicates brain injury
    
    # Adjust based on age
    if age > 65:
        base_iss += 8   # Elderly patients have higher mortality
    elif age > 50:
        base_iss += 4
    
    # Adjust based on injury type
    if injury_type == 'Penetrating':
        base_iss += 5   # Penetrating injuries often more severe
    
    # Adjust based on specific injury text
    injury_lower = str(injury_text).lower()
    if any(severe in injury_lower for severe in ['brain injury', 'spinal cord', 'aortic dissection']):
        base_iss += 15
    elif any(severe in injury_lower for severe in ['liver laceration', 'pelvic fracture', 'pneumothorax']):
        base_iss += 10
    elif any(severe in injury_lower for severe in ['multiple fractures', 'hemothorax']):
        base_iss += 8
    elif any(severe in injury_lower for severe in ['chest trauma', 'abdominal trauma']):
        base_iss += 6
    
    # Ensure ISS is within valid range (1-75)
    return max(1, min(75, base_iss))

def generate_medically_consistent_physiology(age, injury_type, injury_text):
    """Generate medically consistent physiological parameters"""
    # Base physiological parameters based on age
    if age < 30:
        base_sbp = random.randint(110, 140)
        base_resp_rate = random.randint(12, 18)
        base_gcs = random.randint(13, 15)
    elif age < 50:
        base_sbp = random.randint(100, 135)
        base_resp_rate = random.randint(12, 20)
        base_gcs = random.randint(12, 15)
    else:
        base_sbp = random.randint(95, 130)
        base_resp_rate = random.randint(14, 22)
        base_gcs = random.randint(11, 15)
    
    # Adjust based on injury severity (from injury text)
    injury_lower = str(injury_text).lower()
    severity_multiplier = 1.0
    
    if any(severe in injury_lower for severe in ['brain injury', 'spinal cord', 'aortic dissection']):
        severity_multiplier = 2.5
    elif any(severe in injury_lower for severe in ['liver laceration', 'pelvic fracture', 'pneumothorax']):
        severity_multiplier = 2.0
    elif any(severe in injury_lower for severe in ['multiple fractures', 'hemothorax']):
        severity_multiplier = 1.8
    elif any(severe in injury_lower for severe in ['chest trauma', 'abdominal trauma']):
        severity_multiplier = 1.5
    elif any(severe in injury_lower for severe in ['fracture', 'laceration']):
        severity_multiplier = 1.2
    
    # Apply severity adjustments
    sbp = max(60, int(base_sbp / severity_multiplier))
    resp_rate = min(40, int(base_resp_rate * severity_multiplier))
    gcs = max(3, int(base_gcs / severity_multiplier))
    
    # Add some realistic variation
    sbp += random.randint(-10, 10)
    resp_rate += random.randint(-3, 3)
    gcs += random.randint(-1, 1)
    
    # Ensure values are within medical ranges
    sbp = max(60, min(200, sbp))
    resp_rate = max(8, min(40, resp_rate))
    gcs = max(3, min(15, gcs))
    
    return sbp, resp_rate, gcs

def generate_medically_consistent_data(n_entries=200):
    """Generate medically consistent trauma registry data"""
    # Define possible values for categorical variables
    genders = ['Male', 'Female']
    injury_types = ['Blunt', 'Penetrating', 'Burn', 'Other']
    transfer_in_options = ['Yes', 'No']
    discharge_statuses = ['Home', 'Transfer', 'Death', 'Rehabilitation']
    
    # Define injury texts by severity
    severe_injuries = [
        'Severe traumatic brain injury with intracranial hemorrhage',
        'Spinal cord injury with paralysis',
        'Aortic dissection with hemodynamic instability',
        'Multiple organ injuries with hemorrhagic shock',
        'Severe chest trauma with cardiac contusion',
        'Complex pelvic fracture with vascular injury'
    ]
    
    moderate_injuries = [
        'Liver laceration with moderate bleeding',
        'Multiple rib fractures with pneumothorax',
        'Spleen injury requiring intervention',
        'Pelvic fracture with stable hemodynamics',
        'Moderate head injury with brief loss of consciousness',
        'Chest trauma with pulmonary contusion'
    ]
    
    mild_injuries = [
        'Simple fracture of upper extremity',
        'Superficial laceration requiring sutures',
        'Minor head injury without loss of consciousness',
        'Soft tissue injury with bruising',
        'Simple rib fracture without complications',
        'Minor burn injury'
    ]
    
    # Define injury cause memos
    injury_causes = [
        'Motor vehicle accident - driver',
        'Motor vehicle accident - passenger',
        'Motor vehicle accident - pedestrian',
        'Fall from height',
        'Fall on same level',
        'Assault with blunt object',
        'Assault with sharp object',
        'Gunshot wound',
        'Industrial accident',
        'Sports injury',
        'Domestic accident'
    ]
    
    data = []
    
    for i in range(n_entries):
        # Generate basic demographics
        age = random.randint(18, 85)
        gender = random.choice(genders)
        injury_type = random.choice(injury_types)
        
        # Generate injury description based on type and severity
        if injury_type == 'Penetrating':
            injury_text = random.choice(severe_injuries + moderate_injuries)
        elif injury_type == 'Burn':
            injury_text = 'Burn injury with varying degrees of severity'
        else:
            injury_text = random.choice(severe_injuries + moderate_injuries + mild_injuries)
        
        injury_memo = random.choice(injury_causes)
        
        # Generate medically consistent physiology
        sbp, resp_rate, gcs = generate_medically_consistent_physiology(age, injury_type, injury_text)
        
        # Calculate ISS based on medical logic
        iss = calculate_medically_consistent_iss(gcs, sbp, resp_rate, age, injury_type, injury_text)
        
        # Calculate RTS (Revised Trauma Score)
        rts = 0
        if gcs >= 13: rts += 4
        elif gcs >= 9: rts += 3
        elif gcs >= 6: rts += 2
        elif gcs >= 4: rts += 1
        
        if sbp >= 89: rts += 4
        elif sbp >= 76: rts += 3
        elif sbp >= 50: rts += 2
        elif sbp >= 1: rts += 1
        
        if resp_rate >= 10 and resp_rate <= 29: rts += 4
        elif resp_rate > 29: rts += 3
        elif resp_rate >= 6: rts += 2
        elif resp_rate >= 1: rts += 1
        
        # Calculate TRISS (Trauma and Injury Severity Score)
        triss = 0.9378 * (0.7326 if gender == 'Male' else 0.7708) * (0.9965 ** age) * (0.9930 ** iss) * (0.9205 ** (16 - rts))
        
        # Generate dates
        admission_date = datetime.now() - timedelta(days=random.randint(1, 365))
        discharge_date = admission_date + timedelta(days=random.randint(1, 30))
        
        # Generate other vital signs
        heart_rate = random.randint(60, 120)
        temperature = round(random.uniform(36.0, 39.0), 1)
        oxygen_saturation = random.randint(85, 100)
        
        # Generate LLM extracted data
        llm_extracted_data = f"BP: {sbp}/{random.randint(60, 90)} mmHg, HR: {heart_rate} bpm, RR: {resp_rate} /min, Temp: {temperature}°C, SpO2: {oxygen_saturation}%"
        
        # Generate filename
        filename = f"trauma_case_{i+1:03d}_{admission_date.strftime('%Y%m%d')}.csv"
        
        # Generate discharge status
        if iss > 25:
            discharge_status = random.choice(['Transfer', 'Death', 'Rehabilitation'])
        elif iss > 15:
            discharge_status = random.choice(['Home', 'Transfer', 'Rehabilitation'])
        else:
            discharge_status = random.choice(['Home', 'Transfer'])
        
        # Generate transfer in status
        transfer_in = random.choice(transfer_in_options)
        
        row = {
            'Age': age,
            'Gender': gender,
            'Injury Type': injury_type,
            'Injury Text': injury_text,
            'INJ_CAU_MEMO': injury_memo,
            'SBP on Admission': sbp,
            'Unassisted Resp Rate on Admission': resp_rate,
            'GCS on Admission': gcs,
            'ISS': iss,
            'RTS on Admission': rts,
            'TRISS': round(triss, 4),
            'Admission Date': admission_date.strftime('%Y-%m-%d'),
            'Discharge Date': discharge_date.strftime('%Y-%m-%d'),
            'Heart Rate': heart_rate,
            'Temperature': temperature,
            'Oxygen Saturation': oxygen_saturation,
            'llm_extracted_data_columns': llm_extracted_data,
            'Filename': filename,
            'Discharge Status': discharge_status,
            'Transfer In': transfer_in
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    """Main function"""
    print("Generating medically consistent trauma registry data...")
    synthetic_df = generate_medically_consistent_data(200)
    output_filename = "medically_consistent_trauma_reg_200.csv"
    synthetic_df.to_csv(output_filename, index=False)
    print(f"Generated {len(synthetic_df)} medically consistent trauma registry entries")
    print(f"Saved to: {output_filename}")
    
    # Display basic statistics
    print(f"\nBasic Statistics:")
    print(f"Age range: {synthetic_df['Age'].min()} - {synthetic_df['Age'].max()}")
    print(f"ISS range: {synthetic_df['ISS'].min()} - {synthetic_df['ISS'].max()}")
    print(f"GCS range: {synthetic_df['GCS on Admission'].min()} - {synthetic_df['GCS on Admission'].max()}")
    print(f"SBP range: {synthetic_df['SBP on Admission'].min()} - {synthetic_df['SBP on Admission'].max()}")
    
    # Display first few rows
    print(f"\nFirst 3 entries:")
    print(synthetic_df.head(3))

if __name__ == "__main__":
    main()
