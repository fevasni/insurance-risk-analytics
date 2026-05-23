"""
Generates the full eda.ipynb notebook for the insurance-risk-analytics project.
Run from the notebooks/ directory:
    python gen_eda_notebook.py
"""
import json, os

def code_cell(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines if isinstance(lines, list) else [lines],
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src if isinstance(src, list) else [src],
    }

cells = []

# ─────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# Insurance Risk Analytics — Exploratory Data Analysis\n\n"
    "**Dataset:** `MachineLearningRating_v3.csv`  |  **Delimiter:** `|`\n\n"
    "---\n"
    "### Notebook Sections\n"
    "1. Setup & Data Loading\n"
    "2. Dtype Corrections\n"
    "3. Data Quality Assessment\n"
    "4. Univariate Analysis\n"
    "5. Bivariate / Multivariate Analysis\n"
    "6. Geographic Trends\n"
    "7. Outlier Detection\n"
))

# ─────────────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Setup & Data Loading"))

cells.append(code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', None)\n"
    "pd.set_option('display.float_format', '{:,.2f}'.format)\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n"
    "%matplotlib inline"
))

cells.append(code_cell(
    "DATA_PATH = '../data/MachineLearningRating_v3.csv'\n"
    "df = pd.read_csv(DATA_PATH, sep='|', low_memory=False, on_bad_lines='skip')\n"
    "print(f'Shape: {df.shape}')\n"
    "df.head(3)"
))

# ─────────────────────────────────────────────────────────────────────
# 2. DTYPE CORRECTIONS
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 2. Dtype Corrections"))

cells.append(code_cell(
    "# Date columns\n"
    "df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])\n"
    "df['VehicleIntroDate']  = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')\n\n"
    "# Numeric stored as object\n"
    "df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')\n\n"
    "# PostalCode as identifier\n"
    "df['PostalCode'] = df['PostalCode'].astype(str)\n\n"
    "# Low-cardinality -> category\n"
    "low_card = [\n"
    "    'LegalType','Title','Language','Bank','AccountType','MaritalStatus',\n"
    "    'Gender','Country','Province','MainCrestaZone','SubCrestaZone',\n"
    "    'ItemType','VehicleType','bodytype','AlarmImmobiliser','TrackingDevice',\n"
    "    'NewVehicle','WrittenOff','Rebuilt','Converted','CrossBorder',\n"
    "    'TermFrequency','CoverCategory','CoverType','CoverGroup','Section',\n"
    "    'Product','StatutoryClass','StatutoryRiskType','ExcessSelected','Citizenship'\n"
    "]\n"
    "for col in low_card:\n"
    "    df[col] = df[col].astype('category')\n\n"
    "# Drop zero-variance and 100%-missing columns\n"
    "zero_var = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]\n"
    "drop_cols = list(set(zero_var + ['NumberOfVehiclesInFleet']))\n"
    "df.drop(columns=drop_cols, inplace=True)\n\n"
    "# Negative value flags\n"
    "df['flag_negative_premium'] = df['TotalPremium'] < 0\n"
    "df['flag_negative_claims']  = df['TotalClaims']  < 0\n\n"
    "print(f'Shape after corrections: {df.shape}')\n"
    "print(df.dtypes.value_counts())"
))

# ─────────────────────────────────────────────────────────────────────
# 3. DATA QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 3. Data Quality Assessment"))

cells.append(code_cell(
    "missing     = df.isnull().sum()\n"
    "missing_pct = (missing / len(df) * 100).round(2)\n"
    "mdf = pd.DataFrame({'Missing_Count': missing, 'Missing_%': missing_pct})\n"
    "mdf = mdf[mdf['Missing_Count'] > 0].sort_values('Missing_%', ascending=False)\n\n"
    "strategy = {\n"
    "    'NumberOfVehiclesInFleet': 'Dropped — 100% missing',\n"
    "    'CrossBorder':             'Dropped — 99.9% missing + zero variance',\n"
    "    'CustomValueEstimate':     'Keep — only applicable to specific cover types',\n"
    "    'WrittenOff':              'Keep — applicable subset; flag NaN separately',\n"
    "    'Rebuilt':                 'Keep — applicable subset; flag NaN separately',\n"
    "    'Converted':               'Keep — applicable subset; flag NaN separately',\n"
    "    'NewVehicle':              'Impute with mode or label Unknown',\n"
    "    'Bank':                    'Impute with mode or label Unknown',\n"
    "    'AccountType':             'Impute with mode or label Unknown',\n"
    "    'Gender':                  'Map NaN to Not specified',\n"
    "    'MaritalStatus':           'Map NaN to Not specified',\n"
    "}\n"
    "mdf['Handling_Strategy'] = mdf.index.map(strategy).fillna('Drop row — <0.1% missing')\n"
    "display(mdf)"
))

cells.append(code_cell(
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "colors = ['#d62728' if p > 50 else '#ff7f0e' if p > 10 else '#1f77b4'\n"
    "          for p in mdf['Missing_%']]\n"
    "bars = ax.barh(mdf.index, mdf['Missing_%'], color=colors, edgecolor='white')\n"
    "ax.axvline(50,  color='red',    linestyle='--', lw=1, label='50% threshold')\n"
    "ax.axvline(10,  color='orange', linestyle=':',  lw=1, label='10% threshold')\n"
    "for bar, pct in zip(bars, mdf['Missing_%']):\n"
    "    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,\n"
    "            f'{pct:.1f}%', va='center', fontsize=8)\n"
    "ax.set_xlabel('Missing %')\n"
    "ax.set_title('Missing Values by Column', fontweight='bold')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ─────────────────────────────────────────────────────────────────────
# 4. UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 4. Univariate Analysis\n### 4a. Numerical Columns — Histograms"))

cells.append(code_cell(
    "num_cols = [\n"
    "    'TotalPremium','TotalClaims','SumInsured','CalculatedPremiumPerTerm',\n"
    "    'CapitalOutstanding','kilowatts','cubiccapacity','Cylinders',\n"
    "    'RegistrationYear','NumberOfDoors'\n"
    "]\n\n"
    "fig, axes = plt.subplots(2, 5, figsize=(22, 8))\n"
    "fig.suptitle('Univariate — Numerical Features (clipped at 99th pct)', fontsize=14, fontweight='bold')\n\n"
    "for ax, col in zip(axes.flatten(), num_cols):\n"
    "    data = df[col].dropna()\n"
    "    clip_val = data.quantile(0.99)\n"
    "    clipped  = data[data <= clip_val]\n"
    "    ax.hist(clipped, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)\n"
    "    ax.axvline(data.median(), color='red', linestyle='--', lw=1.2,\n"
    "               label=f'Median: {data.median():,.1f}')\n"
    "    ax.set_title(col, fontsize=9)\n"
    "    ax.set_xlabel('Value', fontsize=8)\n"
    "    ax.set_ylabel('Freq',  fontsize=8)\n"
    "    ax.legend(fontsize=7)\n"
    "    ax.tick_params(labelsize=7)\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 4b. Categorical Columns — Bar Charts"))

cells.append(code_cell(
    "cat_plot = [\n"
    "    'Province','VehicleType','Gender','MaritalStatus',\n"
    "    'TermFrequency','AlarmImmobiliser','TrackingDevice',\n"
    "    'LegalType','CoverGroup','Section','bodytype','make'\n"
    "]\n\n"
    "fig, axes = plt.subplots(3, 4, figsize=(22, 14))\n"
    "fig.suptitle('Univariate — Categorical Features', fontsize=14, fontweight='bold')\n\n"
    "for ax, col in zip(axes.flatten(), cat_plot):\n"
    "    counts = df[col].value_counts().head(12)\n"
    "    ax.barh(counts.index.astype(str), counts.values, color='steelblue', edgecolor='white')\n"
    "    ax.set_title(col, fontsize=9)\n"
    "    ax.set_xlabel('Count', fontsize=8)\n"
    "    ax.tick_params(labelsize=7)\n"
    "    for i, v in enumerate(counts.values):\n"
    "        ax.text(v * 1.01, i, f'{v:,}', va='center', fontsize=6)\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ─────────────────────────────────────────────────────────────────────
# 5. BIVARIATE / MULTIVARIATE
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 5. Bivariate / Multivariate Analysis"))

cells.append(md_cell("### 5a. TotalPremium vs TotalClaims — Scatter by Postal Zone"))

cells.append(code_cell(
    "sample = df[['TotalPremium','TotalClaims','PostalCode']].dropna().copy()\n"
    "p99_prem  = sample['TotalPremium'].quantile(0.99)\n"
    "p99_claim = sample['TotalClaims'].quantile(0.99)\n"
    "sample = sample[\n"
    "    (sample['TotalPremium'] <= p99_prem) &\n"
    "    (sample['TotalClaims']  <= p99_claim) &\n"
    "    (sample['TotalPremium'] >= 0) &\n"
    "    (sample['TotalClaims']  >= 0)\n"
    "]\n\n"
    "sample['PostalCode_num'] = pd.to_numeric(sample['PostalCode'], errors='coerce')\n"
    "sample['PostalBucket']  = pd.cut(\n"
    "    sample['PostalCode_num'], bins=5,\n"
    "    labels=['Zone 1','Zone 2','Zone 3','Zone 4','Zone 5']\n"
    ")\n\n"
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "palette = sns.color_palette('tab10', 5)\n"
    "for i, zone in enumerate(['Zone 1','Zone 2','Zone 3','Zone 4','Zone 5']):\n"
    "    sub = sample[sample['PostalBucket'] == zone]\n"
    "    sub = sub.sample(min(2000, len(sub)), random_state=42)\n"
    "    ax.scatter(sub['TotalPremium'], sub['TotalClaims'],\n"
    "               alpha=0.3, s=8, color=palette[i], label=zone)\n\n"
    "ax.set_xlabel('TotalPremium')\n"
    "ax.set_ylabel('TotalClaims')\n"
    "ax.set_title('TotalPremium vs TotalClaims by Postal Zone', fontweight='bold')\n"
    "ax.legend(title='Postal Zone', markerscale=2)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 5b. Aggregated Mean Premium and Claims by PostalCode (Top 20)"))

cells.append(code_cell(
    "postal_agg = (\n"
    "    df.groupby('PostalCode')[['TotalPremium','TotalClaims']]\n"
    "      .mean()\n"
    "      .sort_values('TotalPremium', ascending=False)\n"
    "      .head(20)\n"
    "      .reset_index()\n"
    ")\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
    "fig.suptitle('Mean TotalPremium and TotalClaims — Top 20 PostalCodes', fontweight='bold')\n\n"
    "axes[0].barh(postal_agg['PostalCode'], postal_agg['TotalPremium'], color='steelblue')\n"
    "axes[0].set_title('Avg TotalPremium')\n"
    "axes[0].set_xlabel('Mean Premium')\n\n"
    "axes[1].barh(postal_agg['PostalCode'], postal_agg['TotalClaims'], color='coral')\n"
    "axes[1].set_title('Avg TotalClaims')\n"
    "axes[1].set_xlabel('Mean Claims')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 5c. Correlation Matrix — Key Numerical Features"))

cells.append(code_cell(
    "corr_cols = [\n"
    "    'TotalPremium','TotalClaims','SumInsured','CalculatedPremiumPerTerm',\n"
    "    'CapitalOutstanding','kilowatts','cubiccapacity','Cylinders',\n"
    "    'RegistrationYear','CustomValueEstimate'\n"
    "]\n\n"
    "corr = df[corr_cols].corr()\n\n"
    "fig, ax = plt.subplots(figsize=(11, 9))\n"
    "mask = np.triu(np.ones_like(corr, dtype=bool))\n"
    "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',\n"
    "            cmap='coolwarm', center=0, linewidths=0.5,\n"
    "            ax=ax, cbar_kws={'shrink': 0.8})\n"
    "ax.set_title('Correlation Matrix — Key Numerical Features', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 5d. Pair Plot — Premium, Claims, SumInsured by VehicleType"))

cells.append(code_cell(
    "pp_cols = ['TotalPremium','TotalClaims','SumInsured']\n"
    "pp_data = df[pp_cols + ['VehicleType']].dropna().copy()\n"
    "for c in pp_cols:\n"
    "    pp_data = pp_data[pp_data[c] <= pp_data[c].quantile(0.99)]\n"
    "pp_data = pp_data.sample(5000, random_state=42)\n\n"
    "g = sns.pairplot(\n"
    "    pp_data, vars=pp_cols, hue='VehicleType',\n"
    "    plot_kws={'alpha': 0.4, 's': 15},\n"
    "    diag_kind='kde', corner=True\n"
    ")\n"
    "g.fig.suptitle('Pair Plot — Premium, Claims, SumInsured by VehicleType',\n"
    "               y=1.02, fontweight='bold')\n"
    "plt.show()"
))

# ─────────────────────────────────────────────────────────────────────
# 6. GEOGRAPHIC TRENDS
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 6. Geographic Trends — Province Level"))

cells.append(md_cell("### 6a. Average TotalPremium and TotalClaims by Province"))

cells.append(code_cell(
    "prov_agg = (\n"
    "    df.groupby('Province')[['TotalPremium','TotalClaims']]\n"
    "      .agg(['mean','median','count'])\n"
    "      .round(2)\n"
    ")\n"
    "display(prov_agg)\n\n"
    "prov_mean = df.groupby('Province')[['TotalPremium','TotalClaims']].mean().reset_index()\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
    "fig.suptitle('Average Premium and Claims by Province', fontweight='bold')\n\n"
    "prov_s1 = prov_mean.sort_values('TotalPremium', ascending=True)\n"
    "axes[0].barh(prov_s1['Province'].astype(str), prov_s1['TotalPremium'], color='steelblue')\n"
    "axes[0].set_title('Avg TotalPremium by Province')\n\n"
    "prov_s2 = prov_mean.sort_values('TotalClaims', ascending=True)\n"
    "axes[1].barh(prov_s2['Province'].astype(str), prov_s2['TotalClaims'], color='coral')\n"
    "axes[1].set_title('Avg TotalClaims by Province')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 6b. Top Cover Types by Province — Heatmap"))

cells.append(code_cell(
    "ct_prov = (\n"
    "    df.groupby(['Province','CoverType']).size()\n"
    "      .unstack(fill_value=0)\n"
    ")\n"
    "top_ct   = ct_prov.sum().nlargest(10).index\n"
    "ct_prov  = ct_prov[top_ct]\n\n"
    "fig, ax = plt.subplots(figsize=(14, 6))\n"
    "sns.heatmap(ct_prov, cmap='YlOrRd', annot=True, fmt=',d',\n"
    "            linewidths=0.4, ax=ax, cbar_kws={'shrink': 0.7})\n"
    "ax.set_title('Cover Type Count by Province (Top 10 Cover Types)',\n"
    "             fontsize=13, fontweight='bold')\n"
    "ax.set_xlabel('CoverType')\n"
    "ax.set_ylabel('Province')\n"
    "plt.xticks(rotation=30, ha='right')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 6c. Top Vehicle Makes by Province"))

cells.append(code_cell(
    "top_makes = df['make'].value_counts().head(8).index.tolist()\n"
    "make_prov = (\n"
    "    df[df['make'].isin(top_makes)]\n"
    "      .groupby(['Province','make']).size()\n"
    "      .unstack(fill_value=0)\n"
    ")\n\n"
    "fig, ax = plt.subplots(figsize=(14, 6))\n"
    "sns.heatmap(make_prov, cmap='Blues', annot=True, fmt=',d',\n"
    "            linewidths=0.4, ax=ax, cbar_kws={'shrink': 0.7})\n"
    "ax.set_title('Top 8 Vehicle Makes by Province', fontsize=13, fontweight='bold')\n"
    "ax.set_xlabel('Make')\n"
    "ax.set_ylabel('Province')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 6d. Average Premium by Province and TermFrequency"))

cells.append(code_cell(
    "prov_term = (\n"
    "    df.groupby(['Province','TermFrequency'])['TotalPremium']\n"
    "      .mean()\n"
    "      .unstack()\n"
    ")\n\n"
    "prov_term.plot(kind='bar', figsize=(12, 5), colormap='Set2', edgecolor='white')\n"
    "plt.title('Avg TotalPremium by Province and TermFrequency', fontweight='bold')\n"
    "plt.xlabel('Province')\n"
    "plt.ylabel('Avg TotalPremium')\n"
    "plt.xticks(rotation=30, ha='right')\n"
    "plt.legend(title='TermFrequency')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ─────────────────────────────────────────────────────────────────────
# 7. OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 7. Outlier Detection — Box Plots"))

cells.append(code_cell(
    "outlier_cols = [\n"
    "    'TotalPremium','TotalClaims','SumInsured',\n"
    "    'CalculatedPremiumPerTerm','CapitalOutstanding',\n"
    "    'kilowatts','cubiccapacity'\n"
    "]\n\n"
    "fig, axes = plt.subplots(1, len(outlier_cols), figsize=(22, 5))\n"
    "fig.suptitle('Outlier Detection — Box Plots (clipped at 99.5th pct)',\n"
    "             fontsize=13, fontweight='bold')\n\n"
    "for ax, col in zip(axes, outlier_cols):\n"
    "    data  = df[col].dropna()\n"
    "    clip  = data.quantile(0.995)\n"
    "    ax.boxplot(\n"
    "        data[data <= clip], vert=True, patch_artist=True,\n"
    "        boxprops=dict(facecolor='steelblue', alpha=0.6),\n"
    "        medianprops=dict(color='red', lw=2),\n"
    "        flierprops=dict(marker='.', markersize=2, alpha=0.3)\n"
    "    )\n"
    "    ax.set_title(col, fontsize=8)\n"
    "    ax.set_xticks([])\n"
    "    ax.tick_params(labelsize=7)\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("### 7b. IQR-Based Outlier Summary Table"))

cells.append(code_cell(
    "rows = []\n"
    "for col in outlier_cols:\n"
    "    s  = df[col].dropna()\n"
    "    Q1 = s.quantile(0.25)\n"
    "    Q3 = s.quantile(0.75)\n"
    "    IQR = Q3 - Q1\n"
    "    lo  = Q1 - 1.5 * IQR\n"
    "    hi  = Q3 + 1.5 * IQR\n"
    "    n_out = ((s < lo) | (s > hi)).sum()\n"
    "    rows.append({\n"
    "        'Column': col,\n"
    "        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,\n"
    "        'Lower_Fence': lo, 'Upper_Fence': hi,\n"
    "        'Outlier_Count': n_out,\n"
    "        'Outlier_%': round(n_out / len(s) * 100, 2)\n"
    "    })\n\n"
    "outlier_summary = pd.DataFrame(rows).set_index('Column')\n"
    "display(outlier_summary)"
))

# ─────────────────────────────────────────────────────────────────────
# ASSEMBLE & WRITE
# ─────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(__file__), 'eda.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Total cells: {len(cells)}")
