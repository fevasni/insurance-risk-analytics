"""
Run hypothesis tests for Task 3: A/B Hypothesis Testing
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.abspath(''), 'src'))

from hypothesis_tests import (
    calculate_kpis,
    segment_data,
    run_hypothesis_test,
    format_results,
    check_statistical_equivalence,
    generate_business_interpretation
)

import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("Task 3: A/B Hypothesis Testing")
print("="*100)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)
print(f"Dataset shape: {df.shape}")

# Calculate KPIs
print("\nCalculating KPIs...")
df = calculate_kpis(df)
print(f"Claim Frequency: {(df['claim_frequency'] == 1).sum()} claims out of {len(df)} policies ({(df['claim_frequency'] == 1).sum()/len(df)*100:.2f}%)")
print(f"Claim Severity: Mean = R{df[df['TotalClaims'] > 0]['claim_severity'].mean():.2f}")
print(f"Margin: Mean = R{df['margin'].mean():.2f}")

# Results storage
all_results = []
all_interpretations = []

# ============================================
# Hypothesis Test 1: Risk Differences Across Provinces
# ============================================
print("\n" + "="*100)
print("Hypothesis Test 1: Risk Differences Across Provinces")
print("H₀: There are no risk differences across provinces")
print("KPI: Claim Frequency")
print("="*100)

# Check statistical equivalence
covariates = ['Gender', 'VehicleType', 'CoverType', 'SumInsured']
equivalence = check_statistical_equivalence(
    df, 'Province', 'Gauteng', 'Western Cape', covariates
)

print("\nStatistical Equivalence Check (Gauteng vs Western Cape):")
for cov, is_equiv in equivalence.items():
    status = "✓ Equivalent" if is_equiv else "✗ Not equivalent"
    print(f"  {cov}: {status}")

# Segment data
gauteng_df, western_cape_df = segment_data(df, 'Province', 'Gauteng', 'Western Cape')
print(f"\nGauteng: {len(gauteng_df)} policies")
print(f"Western Cape: {len(western_cape_df)} policies")

# Run hypothesis test for Claim Frequency
results_province = run_hypothesis_test(
    gauteng_df, western_cape_df, 'claim_frequency', test_type='z_test'
)

print("\n" + format_results(results_province))

# Calculate effect size
gauteng_freq = gauteng_df['claim_frequency'].mean()
western_cape_freq = western_cape_df['claim_frequency'].mean()
effect_size = (gauteng_freq - western_cape_freq) / western_cape_freq

print(f"\nGauteng Claim Frequency: {gauteng_freq:.4f} ({gauteng_freq*100:.2f}%)")
print(f"Western Cape Claim Frequency: {western_cape_freq:.4f} ({western_cape_freq*100:.2f}%)")
print(f"Relative Difference: {effect_size:.2%}")

# Generate business interpretation
interpretation_province = generate_business_interpretation(
    hypothesis="No risk differences across provinces",
    results=results_province,
    feature_name="Province",
    control_name="Western Cape",
    test_name="Gauteng",
    kpi_name="Claim Frequency",
    effect_size=effect_size
)

print(f"\nBusiness Interpretation:\n{interpretation_province}")

all_results.append({
    'Hypothesis': 'H₀: No risk differences across provinces',
    'KPI': 'Claim Frequency',
    'Groups Compared': 'Gauteng vs Western Cape',
    'Test Used': results_province['test_name'],
    'P-value': results_province['p_value'],
    'Decision': results_province['decision'],
    'Effect Size': effect_size
})
all_interpretations.append(("Province-based Risk Adjustment", interpretation_province))

# ============================================
# Hypothesis Test 2: Risk Differences Between Zip Codes
# ============================================
print("\n" + "="*100)
print("Hypothesis Test 2: Risk Differences Between Zip Codes")
print("H₀: There are no risk differences between zip codes")
print("KPI: Claim Severity")
print("="*100)

# Find postal codes with sufficient data
postal_counts = df['PostalCode'].value_counts()
large_postals = postal_counts[postal_counts >= 1000].index.tolist()

print(f"\nPostal codes with >= 1000 policies: {len(large_postals)}")
print(f"Top 10: {large_postals[:10]}")

# Select two postal codes for comparison
postal_a = large_postals[0]
postal_b = large_postals[1]

print(f"\nComparing Postal Code {postal_a} vs {postal_b}")
print(f"Data types: PostalCode column is {df['PostalCode'].dtype}")
print(f"Selected postal codes: {postal_a} (type: {type(postal_a)}), {postal_b} (type: {type(postal_b)})")

# Check statistical equivalence
equivalence_postal = check_statistical_equivalence(
    df, 'PostalCode', postal_a, postal_b, covariates
)

print(f"\nStatistical Equivalence Check (Postal {postal_a} vs {postal_b}):")
for cov, is_equiv in equivalence_postal.items():
    status = "✓ Equivalent" if is_equiv else "✗ Not equivalent"
    print(f"  {cov}: {status}")

# Segment data
postal_a_df, postal_b_df = segment_data(df, 'PostalCode', postal_a, postal_b)

# Filter to only policies with claims for severity analysis
postal_a_claims = postal_a_df[postal_a_df['TotalClaims'] > 0]
postal_b_claims = postal_b_df[postal_b_df['TotalClaims'] > 0]

print(f"\nPostal {postal_a}: {len(postal_a_df)} policies, {len(postal_a_claims)} with claims")
print(f"Postal {postal_b}: {len(postal_b_df)} policies, {len(postal_b_claims)} with claims")

# Run hypothesis test for Claim Severity
results_postal_severity = run_hypothesis_test(
    postal_a_claims, postal_b_claims, 'claim_severity', test_type='t_test'
)

print("\n" + format_results(results_postal_severity))

# Calculate effect size
severity_a = postal_a_claims['claim_severity'].mean()
severity_b = postal_b_claims['claim_severity'].mean()
effect_size_severity = (severity_a - severity_b) / severity_b if severity_b > 0 else 0

print(f"\nPostal {postal_a} Claim Severity: R{severity_a:.2f}")
print(f"Postal {postal_b} Claim Severity: R{severity_b:.2f}")
print(f"Relative Difference: {effect_size_severity:.2%}")

# Generate business interpretation
interpretation_postal_severity = generate_business_interpretation(
    hypothesis="No risk differences between zip codes",
    results=results_postal_severity,
    feature_name="Postal Code",
    control_name=f"Postal {postal_b}",
    test_name=f"Postal {postal_a}",
    kpi_name="Claim Severity",
    effect_size=effect_size_severity
)

print(f"\nBusiness Interpretation:\n{interpretation_postal_severity}")

all_results.append({
    'Hypothesis': 'H₀: No risk differences between zip codes',
    'KPI': 'Claim Severity',
    'Groups Compared': f'Postal {postal_a} vs {postal_b}',
    'Test Used': results_postal_severity['test_name'],
    'P-value': results_postal_severity['p_value'],
    'Decision': results_postal_severity['decision'],
    'Effect Size': effect_size_severity
})
all_interpretations.append(("Postal Code-based Risk Adjustment (Claim Severity)", interpretation_postal_severity))

# ============================================
# Hypothesis Test 3: Margin Differences Between Zip Codes
# ============================================
print("\n" + "="*100)
print("Hypothesis Test 3: Margin Differences Between Zip Codes")
print("H₀: There is no significant margin (profit) difference between zip codes")
print("KPI: Margin")
print("="*100)

# Run hypothesis test for Margin
results_postal_margin = run_hypothesis_test(
    postal_a_df, postal_b_df, 'margin', test_type='t_test'
)

print("\n" + format_results(results_postal_margin))

# Calculate effect size
margin_a = postal_a_df['margin'].mean()
margin_b = postal_b_df['margin'].mean()
effect_size_margin = (margin_a - margin_b) / abs(margin_b) if margin_b != 0 else 0

print(f"\nPostal {postal_a} Margin: R{margin_a:.2f}")
print(f"Postal {postal_b} Margin: R{margin_b:.2f}")
print(f"Relative Difference: {effect_size_margin:.2%}")

# Generate business interpretation
interpretation_postal_margin = generate_business_interpretation(
    hypothesis="No significant margin difference between zip codes",
    results=results_postal_margin,
    feature_name="Postal Code",
    control_name=f"Postal {postal_b}",
    test_name=f"Postal {postal_a}",
    kpi_name="Margin",
    effect_size=effect_size_margin
)

print(f"\nBusiness Interpretation:\n{interpretation_postal_margin}")

all_results.append({
    'Hypothesis': 'H₀: No significant margin difference between zip codes',
    'KPI': 'Margin',
    'Groups Compared': f'Postal {postal_a} vs {postal_b}',
    'Test Used': results_postal_margin['test_name'],
    'P-value': results_postal_margin['p_value'],
    'Decision': results_postal_margin['decision'],
    'Effect Size': effect_size_margin
})
all_interpretations.append(("Postal Code-based Margin Adjustment", interpretation_postal_margin))

# ============================================
# Hypothesis Test 4: Risk Differences Between Women and Men
# ============================================
print("\n" + "="*100)
print("Hypothesis Test 4: Risk Differences Between Women and Men")
print("H₀: There is no significant risk difference between Women and Men")
print("KPI: Claim Frequency")
print("="*100)

# Filter to only Male and Female
gender_df = df[df['Gender'].isin(['Male', 'Female'])].copy()

print(f"\nPolicies with Gender specified: {len(gender_df)}")
print(gender_df['Gender'].value_counts())

# Check statistical equivalence
equivalence_gender = check_statistical_equivalence(
    gender_df, 'Gender', 'Male', 'Female', ['Province', 'VehicleType', 'CoverType', 'SumInsured']
)

print("\nStatistical Equivalence Check (Male vs Female):")
for cov, is_equiv in equivalence_gender.items():
    status = "✓ Equivalent" if is_equiv else "✗ Not equivalent"
    print(f"  {cov}: {status}")

# Segment data
male_df, female_df = segment_data(gender_df, 'Gender', 'Male', 'Female')

print(f"\nMale: {len(male_df)} policies")
print(f"Female: {len(female_df)} policies")

# Run hypothesis test for Claim Frequency
results_gender = run_hypothesis_test(
    male_df, female_df, 'claim_frequency', test_type='z_test'
)

print("\n" + format_results(results_gender))

# Calculate effect size
male_freq = male_df['claim_frequency'].mean()
female_freq = female_df['claim_frequency'].mean()
effect_size_gender = (male_freq - female_freq) / female_freq if female_freq > 0 else 0

print(f"\nMale Claim Frequency: {male_freq:.4f} ({male_freq*100:.2f}%)")
print(f"Female Claim Frequency: {female_freq:.4f} ({female_freq*100:.2f}%)")
print(f"Relative Difference: {effect_size_gender:.2%}")

# Generate business interpretation
interpretation_gender = generate_business_interpretation(
    hypothesis="No significant risk difference between Women and Men",
    results=results_gender,
    feature_name="Gender",
    control_name="Female",
    test_name="Male",
    kpi_name="Claim Frequency",
    effect_size=effect_size_gender
)

print(f"\nBusiness Interpretation:\n{interpretation_gender}")

all_results.append({
    'Hypothesis': 'H₀: No risk difference between Women and Men',
    'KPI': 'Claim Frequency',
    'Groups Compared': 'Male vs Female',
    'Test Used': results_gender['test_name'],
    'P-value': results_gender['p_value'],
    'Decision': results_gender['decision'],
    'Effect Size': effect_size_gender
})
all_interpretations.append(("Gender-based Risk Adjustment", interpretation_gender))

# ============================================
# Results Summary Table
# ============================================
print("\n" + "="*100)
print("HYPOTHESIS TESTING RESULTS SUMMARY")
print("="*100)

results_summary = pd.DataFrame(all_results)
results_summary['P-value'] = results_summary['P-value'].apply(lambda x: f"{x:.6f}")
results_summary['Effect Size'] = results_summary['Effect Size'].apply(lambda x: f"{x:.2%}")
print(results_summary.to_string(index=False))

# Save results summary to CSV
results_summary.to_csv('hypothesis_test_results.csv', index=False)
print("\nResults saved to hypothesis_test_results.csv")

# ============================================
# Business Recommendations
# ============================================
print("\n" + "="*100)
print("BUSINESS RECOMMENDATIONS")
print("="*100)

for i, (title, interpretation) in enumerate(all_interpretations, 1):
    print(f"\n{i}. {title}:")
    print(f"   {interpretation}")

# Save business recommendations to file
with open('business_recommendations.txt', 'w', encoding='utf-8') as f:
    f.write("BUSINESS RECOMMENDATIONS\n")
    f.write("="*100 + "\n\n")
    for i, (title, interpretation) in enumerate(all_interpretations, 1):
        f.write(f"{i}. {title}:\n")
        f.write(f"   {interpretation}\n\n")

print("\nBusiness recommendations saved to business_recommendations.txt")

print("\n" + "="*100)
print("Hypothesis testing complete!")
print("="*100)
