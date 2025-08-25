#!/usr/bin/env python3
"""
Create mock OULAD dataset for development and testing purposes.
This generates synthetic data that matches the structure of the real OULAD dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def create_mock_oulad_data():
    """Create mock OULAD dataset with realistic structure."""
    np.random.seed(42)
    
    # Create directories
    raw_dir = Path("data/oulad/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Student Info
    n_students = 5000
    student_ids = range(1, n_students + 1)
    
    student_info = pd.DataFrame({
        'id_student': student_ids,
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'], n_students),
        'code_presentation': np.random.choice(['2013B', '2013J', '2014B', '2014J'], n_students),
        'gender': np.random.choice(['M', 'F'], n_students),
        'region': np.random.choice(['Region1', 'Region2', 'Region3', 'Region4'], n_students),
        'highest_education': np.random.choice(['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification'], n_students),
        'imd_band': np.random.choice(['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'], n_students),
        'age_band': np.random.choice(['0-35', '35-55', '55<='], n_students),
        'num_of_prev_attempts': np.random.choice([0, 1, 2, 3], n_students, p=[0.6, 0.25, 0.1, 0.05]),
        'studied_credits': np.random.choice([30, 60, 90, 120], n_students, p=[0.3, 0.4, 0.2, 0.1]),
        'disability': np.random.choice(['Y', 'N'], n_students, p=[0.1, 0.9])
    })
    
    # Student Registration
    student_registration = pd.DataFrame({
        'code_module': student_info['code_module'],
        'code_presentation': student_info['code_presentation'],
        'id_student': student_info['id_student'],
        'date_registration': np.random.randint(-50, 30, n_students),
        'date_unregistration': [None if np.random.random() > 0.2 else np.random.randint(50, 200) for _ in range(n_students)],
        'final_result': np.random.choice(['Pass', 'Fail', 'Withdrawn', 'Distinction'], n_students, p=[0.4, 0.3, 0.2, 0.1])
    })
    
    # VLE data
    n_vle_records = 50000
    vle_df = pd.DataFrame({
        'id_site': range(1, 201),  # 200 VLE sites
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'], 200),
        'code_presentation': np.random.choice(['2013B', '2013J', '2014B', '2014J'], 200),
        'activity_type': np.random.choice(['oucontent', 'resource', 'url', 'forumng', 'subpage', 'quiz'], 200),
        'week_from': np.random.randint(1, 40, 200),
        'week_to': [None if np.random.random() > 0.7 else np.random.randint(w+1, 40) for w in np.random.randint(1, 39, 200)]
    })
    
    # Student VLE interactions
    student_vle = []
    for student_id in np.random.choice(student_ids, 20000):  # Not all students have VLE data
        n_interactions = np.random.poisson(50)  # Average 50 interactions per student
        for _ in range(n_interactions):
            student_vle.append({
                'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF']),
                'code_presentation': np.random.choice(['2013B', '2013J', '2014B', '2014J']),
                'id_student': student_id,
                'id_site': np.random.choice(range(1, 201)),
                'date': np.random.randint(1, 250),
                'sum_click': np.random.randint(1, 100)
            })
    
    student_vle_df = pd.DataFrame(student_vle)
    
    # Assessments
    assessments = pd.DataFrame({
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'], 100),
        'code_presentation': np.random.choice(['2013B', '2013J', '2014B', '2014J'], 100),
        'id_assessment': range(1, 101),
        'assessment_type': np.random.choice(['TMA', 'CMA', 'Exam'], 100, p=[0.6, 0.3, 0.1]),
        'date': np.random.randint(50, 200, 100),
        'weight': np.random.randint(5, 50, 100)
    })
    
    # Student Assessments
    student_assessments = []
    for student_id in np.random.choice(student_ids, 15000):  # Not all students submit assessments
        n_assessments = np.random.poisson(4)  # Average 4 assessments per student
        for _ in range(n_assessments):
            student_assessments.append({
                'id_assessment': np.random.choice(range(1, 101)),
                'id_student': student_id,
                'date_submitted': np.random.randint(50, 200),
                'is_banked': np.random.choice([0, 1], p=[0.9, 0.1]),
                'score': np.random.randint(0, 100)
            })
    
    student_assessments_df = pd.DataFrame(student_assessments)
    
    # Save all files
    student_info.to_csv(raw_dir / "studentInfo.csv", index=False)
    student_registration.to_csv(raw_dir / "studentRegistration.csv", index=False)
    vle_df.to_csv(raw_dir / "vle.csv", index=False)
    student_vle_df.to_csv(raw_dir / "studentVle.csv", index=False)
    assessments.to_csv(raw_dir / "assessments.csv", index=False)
    student_assessments_df.to_csv(raw_dir / "studentAssessment.csv", index=False)
    
    print(f"Created mock OULAD dataset with {n_students} students")
    print(f"Files saved to {raw_dir}")
    print(f"Student Info: {student_info.shape}")
    print(f"Student Registration: {student_registration.shape}")
    print(f"VLE: {vle_df.shape}")
    print(f"Student VLE: {student_vle_df.shape}")
    print(f"Assessments: {assessments.shape}")
    print(f"Student Assessments: {student_assessments_df.shape}")

if __name__ == "__main__":
    create_mock_oulad_data()