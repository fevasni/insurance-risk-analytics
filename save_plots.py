"""
Script to generate and save 10 key EDA visualizations as PNG files
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
DATA_PATH = 'data/insurance_data.csv'
df = pd.read_csv(DATA_PATH, sep='|', low_memory=False, on_bad_lines='skip')

# Data preprocessing
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')
df['PostalCode'] = df['PostalCode'].astype(str)

low_card = [
    'LegalType','Title','Language','Bank','AccountType','MaritalStatus',
    'Gender','Country','Province','MainCrestaZone','SubCrestaZone',
    'ItemType','VehicleType','bodytype','AlarmImmobiliser','TrackingDevice',
    'NewVehicle','WrittenOff','Rebuilt','Converted','CrossBorder',
    'TermFrequency','CoverCategory','CoverType','CoverGroup','Section',
    'Product','StatutoryClass','StatutoryRiskType','ExcessSelected','Citizenship'
]
for col in low_card:
    df[col] = df[col].astype('category')

zero_var = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
drop_cols = list(set(zero_var + ['NumberOfVehiclesInFleet']))
df.drop(columns=drop_cols, inplace=True)

df['flag_negative_premium'] = df['TotalPremium'] < 0
df['flag_negative_claims'] = df['TotalClaims'] < 0

# Lowercase columns for consistency
df.columns = df.columns.str.lower().str.strip()

print("Data loaded and preprocessed. Generating 10 key plots...\n")

# ============================================================================
# PLOT 1: Missing Values Analysis
# ============================================================================
print("Generating Plot 1: Missing Values...")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mdf = pd.DataFrame({'Missing_Count': missing, 'Missing_%': missing_pct})
mdf = mdf[mdf['Missing_Count'] > 0].sort_values('Missing_%', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#d62728' if p > 50 else '#ff7f0e' if p > 10 else '#1f77b4'
          for p in mdf['Missing_%']]
bars = ax.barh(mdf.index, mdf['Missing_%'], color=colors, edgecolor='white')
ax.axvline(50, color='red', linestyle='--', lw=1, label='50% threshold')
ax.axvline(10, color='orange', linestyle=':', lw=1, label='10% threshold')
for bar, pct in zip(bars, mdf['Missing_%']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=8)
ax.set_xlabel('Missing %', fontsize=11)
ax.set_title('Missing Values by Column', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/01_missing_values.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 2: Loss Ratio by Province, VehicleType, Gender
# ============================================================================
print("Generating Plot 2: Loss Ratio Analysis...")
df['lossratio'] = df['totalclaims'] / df['totalpremium'].replace(0, np.nan)
overall_lr = df[df['totalpremium'] > 0]['lossratio'].mean()

lr_province = df[df['totalpremium'] > 0].groupby('province')['lossratio'].mean().sort_values(ascending=True)
lr_vehicle = df[df['totalpremium'] > 0].groupby('vehicletype')['lossratio'].mean().sort_values(ascending=True)
lr_gender = df[df['totalpremium'] > 0].groupby('gender')['lossratio'].mean().sort_values(ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Loss Ratio by Key Dimensions', fontsize=14, fontweight='bold')

axes[0].barh(lr_province.index.astype(str), lr_province.values, color='steelblue')
axes[0].set_title('Loss Ratio by Province')
axes[0].set_xlabel('Loss Ratio')
axes[0].axvline(overall_lr, color='red', linestyle='--', label=f'Overall: {overall_lr:.2%}')
axes[0].legend()

axes[1].barh(lr_vehicle.index.astype(str), lr_vehicle.values, color='coral')
axes[1].set_title('Loss Ratio by VehicleType')
axes[1].set_xlabel('Loss Ratio')
axes[1].axvline(overall_lr, color='red', linestyle='--', label=f'Overall: {overall_lr:.2%}')
axes[1].legend()

axes[2].barh(lr_gender.index.astype(str), lr_gender.values, color='mediumseagreen')
axes[2].set_title('Loss Ratio by Gender')
axes[2].set_xlabel('Loss Ratio')
axes[2].axvline(overall_lr, color='red', linestyle='--', label=f'Overall: {overall_lr:.2%}')
axes[2].legend()

plt.tight_layout()
plt.savefig('plots/02_loss_ratio_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: Premium vs Claims Scatter by Postal Zone
# ============================================================================
print("Generating Plot 3: Premium vs Claims by Postal Zone...")
sample = df[['totalpremium','totalclaims','postalcode']].dropna().copy()
p99_prem = sample['totalpremium'].quantile(0.99)
p99_claim = sample['totalclaims'].quantile(0.99)
sample = sample[
    (sample['totalpremium'] <= p99_prem) &
    (sample['totalclaims'] <= p99_claim) &
    (sample['totalpremium'] >= 0) &
    (sample['totalclaims'] >= 0)
]

sample['postalcode_num'] = pd.to_numeric(sample['postalcode'], errors='coerce')
sample['postalbucket'] = pd.cut(
    sample['postalcode_num'], bins=5,
    labels=['Zone 1','Zone 2','Zone 3','Zone 4','Zone 5']
)

fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette('tab10', 5)
for i, zone in enumerate(['Zone 1','Zone 2','Zone 3','Zone 4','Zone 5']):
    sub = sample[sample['postalbucket'] == zone]
    sub = sub.sample(min(2000, len(sub)), random_state=42)
    ax.scatter(sub['totalpremium'], sub['totalclaims'],
               alpha=0.3, s=8, color=palette[i], label=zone)

ax.set_xlabel('TotalPremium', fontsize=11)
ax.set_ylabel('TotalClaims', fontsize=11)
ax.set_title('TotalPremium vs TotalClaims by Postal Zone', fontweight='bold', fontsize=12)
ax.legend(title='Postal Zone', markerscale=2)
plt.tight_layout()
plt.savefig('plots/03_premium_vs_claims_zones.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 4: Correlation Matrix
# ============================================================================
print("Generating Plot 4: Correlation Matrix...")
num_cols = [
    'totalpremium','totalclaims','suminsured','calculatedpremiumperterm',
    'capitaloutstanding','kilowatts','cubiccapacity','cylinders'
]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix — Key Numerical Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 5: Average Premium and Claims by Province
# ============================================================================
print("Generating Plot 5: Province-level Aggregation...")
prov_mean = df.groupby('province')[['totalpremium','totalclaims']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Average Premium and Claims by Province', fontweight='bold', fontsize=13)

prov_s1 = prov_mean.sort_values('totalpremium', ascending=True)
axes[0].barh(prov_s1['province'].astype(str), prov_s1['totalpremium'], color='steelblue')
axes[0].set_title('Avg TotalPremium by Province')
axes[0].set_xlabel('Mean Premium')

prov_s2 = prov_mean.sort_values('totalclaims', ascending=True)
axes[1].barh(prov_s2['province'].astype(str), prov_s2['totalclaims'], color='coral')
axes[1].set_title('Avg TotalClaims by Province')
axes[1].set_xlabel('Mean Claims')

plt.tight_layout()
plt.savefig('plots/05_province_aggregation.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 6: Cover Type Distribution Heatmap
# ============================================================================
print("Generating Plot 6: Cover Type by Province...")
ct_prov = (
    df.groupby(['province','covertype']).size()
      .unstack(fill_value=0)
)
top_ct = ct_prov.sum().nlargest(10).index
ct_prov = ct_prov[top_ct]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(ct_prov, cmap='YlOrRd', annot=True, fmt=',d',
            linewidths=0.4, ax=ax, cbar_kws={'shrink': 0.7})
ax.set_title('Cover Type Count by Province (Top 10 Cover Types)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('CoverType')
ax.set_ylabel('Province')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('plots/06_cover_type_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 7: Temporal Trends - Claim Frequency & Severity
# ============================================================================
print("Generating Plot 7: Temporal Trends...")
df['yearmonth'] = df['transactionmonth'].dt.to_period('M')

claim_freq = df.groupby('yearmonth').agg(
    claim_count=('totalclaims', lambda x: (x > 0).sum()),
    total_records=('totalclaims', 'count'),
    claim_frequency=('totalclaims', lambda x: (x > 0).sum() / len(x))
).reset_index()
claim_freq['yearmonth'] = claim_freq['yearmonth'].astype(str)

claim_severity = df[df['totalclaims'] > 0].groupby('yearmonth').agg(
    mean_claim_amount=('totalclaims', 'mean'),
    median_claim_amount=('totalclaims', 'median')
).reset_index()
claim_severity['yearmonth'] = claim_severity['yearmonth'].astype(str)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Temporal Trends - Claim Frequency & Severity', fontsize=14, fontweight='bold')

axes[0, 0].plot(claim_freq['yearmonth'], claim_freq['claim_count'], marker='o', color='steelblue', linewidth=2)
axes[0, 0].set_title('Claim Count Over Time')
axes[0, 0].set_ylabel('Count of Claims')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].plot(claim_freq['yearmonth'], claim_freq['claim_frequency'], marker='o', color='coral', linewidth=2)
axes[0, 1].set_title('Claim Frequency Over Time')
axes[0, 1].set_ylabel('Claim Frequency')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].plot(claim_severity['yearmonth'], claim_severity['mean_claim_amount'], marker='o', color='mediumseagreen', linewidth=2)
axes[1, 0].set_title('Mean Claim Amount Over Time')
axes[1, 0].set_ylabel('Mean Claim Amount')
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].plot(claim_severity['yearmonth'], claim_severity['median_claim_amount'], marker='o', color='purple', linewidth=2)
axes[1, 1].set_title('Median Claim Amount Over Time')
axes[1, 1].set_ylabel('Median Claim Amount')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/07_temporal_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 8: Top Vehicle Makes by Mean Claim Amount
# ============================================================================
print("Generating Plot 8: Vehicle Makes Risk Profile...")
claims_df = df[df['totalclaims'] > 0].copy()
make_stats = claims_df.groupby('make').agg(
    mean_claim=('totalclaims', 'mean'),
    claim_count=('totalclaims', 'count')
).sort_values('mean_claim', ascending=False)

top10_makes = make_stats.head(10)
bottom10_makes = make_stats.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Vehicle Makes - Highest & Lowest Claim Amounts', fontsize=14, fontweight='bold')

axes[0].barh(top10_makes.index.astype(str), top10_makes['mean_claim'], color='steelblue')
axes[0].set_title('Top 10 Vehicle Makes by Mean Claim Amount')
axes[0].set_xlabel('Mean Claim Amount')
axes[0].invert_yaxis()

axes[1].barh(bottom10_makes.index.astype(str), bottom10_makes['mean_claim'], color='coral')
axes[1].set_title('Bottom 10 Vehicle Makes by Mean Claim Amount')
axes[1].set_xlabel('Mean Claim Amount')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('plots/08_vehicle_makes_risk.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 9: Postal Code Region Analysis
# ============================================================================
print("Generating Plot 9: Postal Code Region Analysis...")
df['postalcodeprefix'] = df['postalcode'].astype(str).str[:3]
postal_stats = df.groupby('postalcodeprefix').agg(
    policy_count=('policyid', 'count'),
    total_premium=('totalpremium', 'sum'),
    total_claims=('totalclaims', 'sum'),
    mean_premium=('totalpremium', 'mean'),
    mean_claims=('totalclaims', 'mean')
).round(2)
postal_stats = postal_stats[postal_stats['policy_count'] >= 10]
postal_stats['loss_ratio'] = (postal_stats['total_claims'] / postal_stats['total_premium']).fillna(0).round(4)

top20_postal = postal_stats.sort_values('policy_count', ascending=False).head(20)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Geographic Analysis - Top 20 Postal Code Regions', fontsize=14, fontweight='bold')

x_labels = top20_postal.index

axes[0, 0].bar(range(len(top20_postal)), top20_postal['policy_count'], color='skyblue')
axes[0, 0].set_title('Policy Count by Postal Code Region')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticks(range(len(top20_postal)))
axes[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')

axes[0, 1].bar(range(len(top20_postal)), top20_postal['mean_premium'], color='lightcoral')
axes[0, 1].set_title('Mean Premium by Postal Code Region')
axes[0, 1].set_ylabel('Mean Premium')
axes[0, 1].set_xticks(range(len(top20_postal)))
axes[0, 1].set_xticklabels(x_labels, rotation=45, ha='right')

axes[1, 0].bar(range(len(top20_postal)), top20_postal['mean_claims'], color='lightgreen')
axes[1, 0].set_title('Mean Claims by Postal Code Region')
axes[1, 0].set_ylabel('Mean Claims')
axes[1, 0].set_xticks(range(len(top20_postal)))
axes[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')

axes[1, 1].bar(range(len(top20_postal)), top20_postal['loss_ratio'].values, color='gold')
axes[1, 1].set_title('Loss Ratio by Postal Code Region')
axes[1, 1].set_ylabel('Loss Ratio')
axes[1, 1].set_xticks(range(len(top20_postal)))
axes[1, 1].set_xticklabels(x_labels, rotation=45, ha='right')
axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Breakeven')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('plots/09_postal_region_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 10: Outlier Detection - Box Plots
# ============================================================================
print("Generating Plot 10: Outlier Detection...")
outlier_cols = [
    'totalpremium','totalclaims','suminsured',
    'calculatedpremiumperterm','capitaloutstanding',
    'kilowatts','cubiccapacity'
]

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(22, 5))
fig.suptitle('Outlier Detection — Box Plots (clipped at 99.5th pct)',
             fontsize=13, fontweight='bold')

for ax, col in zip(axes, outlier_cols):
    data = df[col].dropna()
    clip = data.quantile(0.995)
    ax.boxplot(
        data[data <= clip], vert=True, patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        medianprops=dict(color='red', lw=2),
        flierprops=dict(marker='.', markersize=2, alpha=0.3)
    )
    ax.set_title(col, fontsize=8)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('plots/10_outlier_detection.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All 10 plots saved successfully to 'plots/' directory:")
print("  01_missing_values.png")
print("  02_loss_ratio_analysis.png")
print("  03_premium_vs_claims_zones.png")
print("  04_correlation_matrix.png")
print("  05_province_aggregation.png")
print("  06_cover_type_heatmap.png")
print("  07_temporal_trends.png")
print("  08_vehicle_makes_risk.png")
print("  09_postal_region_analysis.png")
print("  10_outlier_detection.png")
