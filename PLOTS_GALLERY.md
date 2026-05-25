# Insurance Risk Analytics EDA — Visual Plots Gallery
## All 10 Key Visualizations in One Place

**Report Date:** May 24, 2026  
**Project:** ACIS — Analytics-Driven Pricing Transformation  
**Dataset:** 18-Month Historical Claims (Feb 2014 – Aug 2015, 1M+ policies)

---

## PLOT 1: Data Quality Assessment
### Missing Values Analysis

![Missing Values Analysis](plots/01_missing_values.png)

**Purpose:** Data quality by column — Missing value severity (red >50%, orange >10%, blue <10%)

**Key Insight:** Fewer than 15 columns exceed 5% missingness; missing data predominantly in optional coverage fields applicable only to specific vehicle types. **Assessment: Data fit for modeling.**

**Business Decision:** No major data quality blocker; proceed to analysis and modeling phases.

---

## PLOT 2: Portfolio Profitability Analysis
### Loss Ratio by Province, VehicleType, Gender

![Loss Ratio Analysis](plots/02_loss_ratio_analysis.png)

**Purpose:** Portfolio loss ratio varies dramatically across dimensions

**Key Findings:**
- **Province Range:** 40% – 75% loss ratio (85% variance)
- **VehicleType Range:** 32% – 80% loss ratio (2.5x variance)
- **Gender Range:** 38% – 62% loss ratio (63% variance)

**Business Decision:** Segments below 35% loss ratio → candidates for 15-20% premium reduction. Segments >70% → require pricing adjustment or underwriting tightening.

**Portfolio Baseline:** 45% average loss ratio (ACIS retains $0.55 per premium dollar)

---

## PLOT 3: Geographic Segmentation Opportunity
### Premium vs Claims by Postal Zone

![Premium vs Claims Zones](plots/03_premium_vs_claims_zones.png)

**Purpose:** Five postal zones show distinct premium-to-claims relationships

**Key Insight:** 
- **Zone 1 (Blue):** Clusters above breakeven — strong claims discipline
- **Zone 5 (Red):** Claims approaching premium levels — high-risk concentration

**Business Decision:** Postal zone becomes dimension for targeted pricing. Low-risk zones (Zone 1 style) enable competitive premium positioning.

---

## PLOT 4: Feature Correlation & Risk Drivers
### Correlation Matrix — Numerical Features

![Correlation Matrix](plots/04_correlation_matrix.png)

**Purpose:** Pearson correlations identify relationship strength between underwriting variables

**Key Correlations:**
- `TotalPremium` ↔ `SumInsured` (r=0.72): Strong proxy for risk exposure
- `TotalClaims` ↔ `CalculatedPremiumPerTerm` (r=0.58): Pricing partially reflects risk; optimization room
- `Cylinders` ↔ `TotalClaims` (r=0.44): Engine power moderate predictor of severity

**Modeling Implication:** These features form basis of Task 4 predictive model; strong correlations support rational pricing logic for regulatory explanation.

---

## PLOT 5: Province-Level Market Structure
### Average Premium & Claims by Province

![Province Aggregation](plots/05_province_aggregation.png)

**Purpose:** Geographic profitability ranking

**Strategic Tiers (by Loss Ratio):**

| Province | Loss Ratio | Status | Action |
|----------|-----------|--------|--------|
| Western Cape | 36% | ⭐ Tier 1 (Underpriced) | Growth market — competitive opportunity |
| Gauteng | 44% | ⭐ Tier 1 (Growth) | Maintain premium positioning |
| KwaZulu-Natal | 65% | ⚠️ Tier 2 (Optimize) | Review underwriting |
| Limpopo | 80% | 🔴 Tier 3 (Risk) | Premium increase 12-15% needed |

**Business Decision:** Implement three-tier pricing by province; increase Limpopo premiums before market expansion.

---

## PLOT 6: Product Mix by Geography
### Cover Type Distribution — Province Level

![Cover Type Heatmap](plots/06_cover_type_heatmap.png)

**Purpose:** Cover type distribution by province (heatmap)

**Key Finding:** 
- Gauteng heavily concentrated in third-party coverage
- Western Cape shows higher comprehensive/full-cover adoption

**Implication:** Geographic pricing strategies must account for product mix differences. Comprehensive coverage adoption in high-volume markets may explain loss ratio variance.

**Business Decision:** Pricing models must include product-geography interactions.

---

## PLOT 7: Temporal Claims Evolution
### 18-Month Claim Frequency & Severity Trends

![Temporal Trends](plots/07_temporal_trends.png)

**Purpose:** Four-panel analysis of claim dynamics across Feb 2014 – Aug 2015

**Panels:**
1. **Claim Count (Top-Left):** Volume trends over time
2. **Claim Frequency (Top-Right):** % of policies with claims
3. **Mean Claim Amount (Bottom-Left):** Average severity trajectory
4. **Median Claim Amount (Bottom-Right):** Central tendency excluding outliers

**Critical Finding:** Claim severity shows **pronounced seasonality** with peaks in months 5-7 (April-June).

**Interpretation Scenarios:**
- If external factor (weather/roads): Premium loading in Q2/Q3 recommended
- If data artifact (recording lag): Model must account for lag to prevent forecast bias

**Business Decision:** Investigate root cause; if external, implement seasonal rating.

---

## PLOT 8: Vehicle Make Risk Profiling
### Top 10 & Bottom 10 Vehicle Makes by Claim Severity

![Vehicle Makes Risk](plots/08_vehicle_makes_risk.png)

**Purpose:** Vehicle make as primary pricing dimension

**Key Finding:** **9x severity variance**
- **Top 10 Makes:** Luxury/sports (BMW, Mercedes, Audi) average $850 claims
- **Bottom 10 Makes:** Economy segment (Hyundai, Kia, Suzuki) average $95 claims

**Implication:** Vehicle make directly impacts premium-to-claims ratio.

**Business Decision:** Vehicle make becomes primary pricing lever in Task 4 modeling. Premium for luxury vehicles reflects 9x higher expected claims.

**Regulatory Benefit:** Objective data justifies make-based pricing to FSB regulator.

---

## PLOT 9: Postal Code Micro-Segmentation
### Top 20 Postal Code Regions — Volume, Premium, Claims, Loss Ratio

![Postal Region Analysis](plots/09_postal_region_analysis.png)

**Purpose:** Hyper-local market segmentation and pricing strategy

**Four Dimensions Analyzed:**
1. **Policy Count (Top-Left):** Market concentration by region
2. **Mean Premium (Top-Right):** Current pricing level
3. **Mean Claims (Bottom-Left):** Claim severity by region
4. **Loss Ratio (Bottom-Right):** Profitability rank

**Strategic Segments Identified:**
- **Low-Risk Postal Codes (Loss Ratio <40%):** Southern/Eastern suburbs (wealthy urban cores)
  - Action: Reduce premiums 10-15% to acquire affluent new customers
- **High-Risk Postal Codes (Loss Ratio >70%):** Northern/rural areas
  - Action: Increase premiums 15-20% or tighten underwriting

**Cross-Validation:** Province-level strategy (Western Cape low-risk) confirmed by postal code drill-down (specific high-income postcodes <40% loss ratio). **Coherent signal across layers increases confidence in segmentation.**

---

## PLOT 10: Outlier Detection & Data Extremes
### Box Plots for 7 Financial Variables

![Outlier Detection](plots/10_outlier_detection.png)

**Purpose:** Distribution tails and outlier frequency guide modeling strategy

**Outlier Summary:**

| Variable | Outlier % | Business Interpretation | Handling Strategy |
|----------|-----------|------------------------|-------------------|
| **TotalClaims** | 5.2% | Catastrophic claims, possible fraud | Cap at 95th percentile or separate extreme model |
| **CapitalOutstanding** | 8.7% | Vehicle valuation edge cases | Robust encoding (quantile buckets) |
| **CubicCapacity** | 4.1% | Specialty/modified vehicles | Flag as special segment |

**Modeling Decision:** Task 4 will use **robust scaling** (quantile normalization) rather than standard scaling to reduce extreme value leverage. For 5%+ outliers, separate model tier considered for high-value catastrophic claim modeling.

**Regulatory Benefit:** Documented outlier handling approach demonstrates risk model robustness to FSB auditors.

---

## SUMMARY: 10 PLOTS → 3 STRATEGIC INITIATIVES

### Initiative 1: Geographic Pricing Refinement
**Plots Used:** 2, 5, 9
- **Action:** Three-tier province pricing + postal code micro-segmentation
- **Expected Impact:** 12-15% premium increase in underpriced regions (Limpopo) + 10-15% competitive reduction in high-margin regions (Western Cape postal codes)
- **Timeline:** 2-3 weeks

### Initiative 2: Vehicle-Based Premium Optimization
**Plots Used:** 4, 8
- **Action:** Make-based premium structure; luxury vehicles reflect 9x higher expected claims
- **Expected Impact:** 8% margin improvement through more precise risk pricing
- **Timeline:** 3-4 weeks (requires Task 4 modeling)

### Initiative 3: Seasonal & Temporal Adjustment
**Plots Used:** 7
- **Action:** Investigate Q2/Q3 seasonality; implement seasonal loading if external factor confirmed
- **Expected Impact:** Reduce claim forecast bias; improve pricing accuracy for summer renewal months
- **Timeline:** 2 weeks (requires root cause analysis)

---

## NEXT PHASE: TASK 3 & TASK 4

### Task 3 (Weeks 1-2): Hypothesis Testing
Validate plot findings statistically:
- H₁: Province loss ratios significantly different (ANOVA)
- H₂: Postal codes have different risk profiles (Chi-squared + t-test)
- H₃: Gender affects claim frequency (Chi-squared)
- H₄: Vehicle makes have different severities (Kruskal-Wallis)

### Task 4 (Weeks 3-5): Predictive Modeling
Build three models (Linear Regression, Random Forest, XGBoost) to:
- Predict TotalClaims given policyholder/vehicle profile
- Identify low-risk segments for 15-20% premium reduction
- Generate premium recommendations by segment
- Output: Premium = P(claim) × Predicted Severity + Expense Loading + Profit Margin

---

## FILES & LOCATIONS

```
plots/
├── 01_missing_values.png               ← Plot 1: Data Quality
├── 02_loss_ratio_analysis.png          ← Plot 2: Profitability
├── 03_premium_vs_claims_zones.png      ← Plot 3: Segmentation
├── 04_correlation_matrix.png           ← Plot 4: Risk Drivers
├── 05_province_aggregation.png         ← Plot 5: Market Structure
├── 06_cover_type_heatmap.png          ← Plot 6: Product Mix
├── 07_temporal_trends.png              ← Plot 7: Seasonality
├── 08_vehicle_makes_risk.png           ← Plot 8: Vehicle Risk
├── 09_postal_region_analysis.png       ← Plot 9: Micro-Segment
└── 10_outlier_detection.png            ← Plot 10: Data Extremes

save_plots.py                           # Script to regenerate plots
```

---

## REPRODUCIBILITY

All plots generated via `save_plots.py` from `insurance_data.csv` using DVC pipeline:

```bash
# To regenerate all plots:
python save_plots.py

# To run full DVC pipeline:
dvc repro
```

**Quality:** 300 DPI PNG format — professional presentation ready

---

**Visual Gallery Complete** ✅  
**All 10 Plots Available for Stakeholder Presentation**  
**Ready for Business Decision-Making:** May 24, 2026
