import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def load_data(file_path):
    """Load CSV data with proper encoding handling."""
    try:
        # First attempt with utf-8
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Second attempt with latin-1
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    return df


def get_summary_stats(df, numeric_only=True):
    """Get comprehensive summary statistics for dataframe."""
    if numeric_only:
        cols = df.select_dtypes(include=['number']).columns
    else:
        cols = df.columns

    stats_dict = {}
    for col in cols:
        col_stats = {
            'count': df[col].count(),
            'missing': df[col].isna().sum(),
            'unique': df[col].nunique(),
            'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'zeros': (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else None
        }
        stats_dict[col] = col_stats

    return stats_dict


def analyze_category_scores(category_scores, category_name):
    """Analyze category score distributions and ranges."""
    results = {}

    # Find the score column
    score_col = next((col for col in category_scores.columns if col.startswith(f"Cat_{category_name}")
                      or col.startswith(f"ScaledScore_{category_name}")), None)

    if not score_col:
        print(f"No score column found for Category {category_name}")
        return {}

    # Basic distribution stats
    score_series = category_scores[score_col].dropna()

    results["count"] = len(score_series)
    results["mean"] = score_series.mean()
    results["median"] = score_series.median()
    results["std"] = score_series.std()
    results["min"] = score_series.min()
    results["max"] = score_series.max()

    # Check for out-of-range values
    out_of_range = ((score_series < 0) | (score_series > 10)).sum()
    results["out_of_range_count"] = out_of_range
    results["out_of_range_pct"] = out_of_range / len(score_series) * 100 if len(score_series) > 0 else 0

    # Distribution by ranges
    ranges = [0, 2.5, 5, 7.5, 10]
    range_labels = ['0-2.5', '2.5-5', '5-7.5', '7.5-10']
    dist = pd.cut(score_series, ranges, labels=range_labels)
    results["distribution"] = dist.value_counts().to_dict()

    # Correlation with other metrics if available
    if "Location" in category_scores.columns:
        results["locations_count"] = category_scores["Location"].nunique()

    if f"Score_{category_name}" in category_scores.columns and f"ScaledScore_{category_name}" in category_scores.columns:
        results["raw_scaled_correlation"] = category_scores[f"Score_{category_name}"].corr(
            category_scores[f"ScaledScore_{category_name}"])

    return results


def analyze_flag_calculations(flags_df, green_red_list_df):
    """Validate flag calculation consistency."""
    results = {}

    # Flag counts
    if "FlagType" in flags_df.columns:
        flag_counts = flags_df["FlagType"].value_counts().to_dict()
        results["flag_counts"] = flag_counts

        # Calculate per-location flag counts
        location_flag_counts = flags_df.groupby(["Location", "FlagType"]).size().unstack().fillna(0)
        results["locations_with_flags"] = len(location_flag_counts)

        if "Green" in location_flag_counts.columns and "Red" in location_flag_counts.columns:
            red_more_than_green = (location_flag_counts["Red"] > location_flag_counts["Green"]).sum()
            results["locations_with_more_red"] = red_more_than_green

            # Extreme locations
            extremes = {}
            extremes["max_green"] = location_flag_counts["Green"].max()
            extremes["max_green_location"] = location_flag_counts["Green"].idxmax() if location_flag_counts[
                                                                                           "Green"].max() > 0 else None
            extremes["max_red"] = location_flag_counts["Red"].max()
            extremes["max_red_location"] = location_flag_counts["Red"].idxmax() if location_flag_counts[
                                                                                       "Red"].max() > 0 else None
            results["extremes"] = extremes

    # Check for problem status items
    if "Check" in green_red_list_df.columns:
        problem_count = (green_red_list_df["Check"] == "Problem").sum()
        results["problem_status_count"] = problem_count
        results["problem_status_pct"] = problem_count / len(green_red_list_df) * 100 if len(
            green_red_list_df) > 0 else 0

    return results


def analyze_financial_metrics(metrics_df):
    """Analyze core financial metric consistency."""
    results = {}

    # Check for key financial columns
    financial_cols = ["Margin", "Growth"]
    for col in financial_cols:
        if col in metrics_df.columns:
            col_stats = {
                "count": metrics_df[col].count(),  # Add this line to include count
                "mean": metrics_df[col].mean(),
                "median": metrics_df[col].median(),
                "min": metrics_df[col].min(),
                "max": metrics_df[col].max(),
                "std": metrics_df[col].std(),
                "negative_values": (metrics_df[col] < 0).sum(),
                "zero_values": (metrics_df[col] == 0).sum(),
                "null_values": metrics_df[col].isna().sum()
            }
            results[col] = col_stats

    # Check for revenue and volume columns
    revenue_cols = [col for col in metrics_df.columns if "Revenue" in col]
    volume_cols = [col for col in metrics_df.columns if "Volume" in col]

    if revenue_cols and volume_cols:
        # Check division by zero handling
        results["division_by_zero_handling"] = {}

        # Sample for months with zero volume but non-zero revenue
        for vol_col in volume_cols:
            if vol_col.replace("Volume", "Month") in metrics_df.columns:
                month_col = vol_col.replace("Volume", "Month")
                zero_month_nonzero_revenue = ((metrics_df[month_col] == 0) &
                                              (metrics_df[vol_col.replace("Volume", "Revenue")] > 0)).sum()
                results["division_by_zero_handling"][
                    f"{vol_col}_zero_month_nonzero_revenue"] = zero_month_nonzero_revenue

    return results


def analyze_location_specific(tables_dict, sample_locations=None):
    """Perform deep analysis on specific locations."""
    results = {}

    # If no sample locations provided, select diverse locations based on final scores
    if not sample_locations and "FinalScores" in tables_dict:
        final_scores_df = tables_dict["FinalScores"]
        if "Avg_Score" in final_scores_df.columns and "Location" in final_scores_df.columns:
            # Get high, medium, and low scoring locations
            sorted_by_score = final_scores_df.sort_values("Avg_Score")
            num_locations = len(sorted_by_score)

            if num_locations >= 5:
                sample_locations = [
                    sorted_by_score.iloc[0]["Location"],  # Lowest
                    sorted_by_score.iloc[num_locations // 4]["Location"],  # 25th percentile
                    sorted_by_score.iloc[num_locations // 2]["Location"],  # Median
                    sorted_by_score.iloc[3 * num_locations // 4]["Location"],  # 75th percentile
                    sorted_by_score.iloc[-1]["Location"]  # Highest
                ]
            else:
                sample_locations = sorted_by_score["Location"].tolist()

    if not sample_locations:
        print("No sample locations could be determined.")
        return {}

    # Analyze each sample location
    for location in sample_locations:
        location_data = {"metrics": {}}

        # Collect flag data
        if "Flags" in tables_dict:
            flags_df = tables_dict["Flags"]
            if "Location" in flags_df.columns:
                location_flags = flags_df[flags_df["Location"] == location]
                if not location_flags.empty:
                    location_data["metrics"]["green_flags"] = \
                    location_flags[location_flags["FlagType"] == "Green"].shape[0]
                    location_data["metrics"]["red_flags"] = location_flags[location_flags["FlagType"] == "Red"].shape[0]

        # Collect category scores
        for category in ["A", "B", "C", "D"]:
            category_key = f"Category{category}Scores"
            if category_key in tables_dict:
                cat_df = tables_dict[category_key]
                if "Location" in cat_df.columns:
                    location_cat = cat_df[cat_df["Location"] == location]
                    if not location_cat.empty:
                        score_col = next((col for col in location_cat.columns if col.startswith(f"Cat_{category}") or
                                          col.startswith(f"ScaledScore_{category}")), None)
                        if score_col:
                            location_data["metrics"][f"category_{category}_score"] = location_cat[score_col].iloc[0]

        # Collect final score
        if "FinalScores" in tables_dict:
            final_df = tables_dict["FinalScores"]
            if "Location" in final_df.columns and "Avg_Score" in final_df.columns:
                location_final = final_df[final_df["Location"] == location]
                if not location_final.empty:
                    location_data["metrics"]["final_avg_score"] = location_final["Avg_Score"].iloc[0]

        # Store results for this location
        results[location] = location_data

    return results


def analyze_final_scores(final_scores_df):
    """Statistical analysis of final scores."""
    results = {}

    if "Avg_Score" not in final_scores_df.columns:
        return {"error": "No Avg_Score column found"}

    avg_scores = final_scores_df["Avg_Score"].dropna()

    # Basic stats
    results["count"] = len(avg_scores)
    results["mean"] = avg_scores.mean()
    results["median"] = avg_scores.median()
    results["std"] = avg_scores.std()
    results["min"] = avg_scores.min()
    results["max"] = avg_scores.max()

    # Distribution by ranges
    ranges = [0, 2.5, 5, 7.5, 10]
    range_labels = ['0-2.5', '2.5-5', '5-7.5', '7.5-10']
    dist = pd.cut(avg_scores, ranges, labels=range_labels)
    results["distribution"] = dist.value_counts().to_dict()

    # Check for all category scores
    cat_cols = [col for col in final_scores_df.columns if col.startswith("Cat_")]

    # Correlations between category scores and final score
    results["correlations"] = {}
    for col in cat_cols:
        correlation = final_scores_df[col].corr(final_scores_df["Avg_Score"])
        results["correlations"][col] = correlation

    # Missing data analysis
    missing_data = {}
    for col in cat_cols:
        missing_count = final_scores_df[col].isna().sum()
        missing_data[col] = {"count": missing_count, "percentage": missing_count / len(final_scores_df) * 100}
    results["missing_data"] = missing_data

    # Distribution of locations with different score counts
    score_counts = final_scores_df[cat_cols].notna().sum(axis=1)
    results["score_count_distribution"] = score_counts.value_counts().sort_index().to_dict()

    return results


def analyze_segment_calculations(market_summary_df, market_delta_df):
    """Verify segment calculation consistency."""
    results = {}

    # Validate Market Summary calculations
    if market_summary_df is not None:
        # Check SoM calculations
        if all(col in market_summary_df.columns for col in ["SKU", "Total_TMO", "SoM"]):
            # Recalculate SoM and compare with the provided values
            epsilon = 1e-9  # Small value to handle floating point precision issues
            recalculated_som = market_summary_df["SKU"] * 100 / market_summary_df["Total_TMO"].replace(0, epsilon)
            max_diff = (market_summary_df["SoM"] - recalculated_som).abs().max()
            avg_diff = (market_summary_df["SoM"] - recalculated_som).abs().mean()

            results["som_calculation"] = {
                "max_difference": max_diff,
                "avg_difference": avg_diff,
                "valid": max_diff < 0.01  # Assuming small differences due to rounding
            }

    # Validate Market Delta calculations
    if market_delta_df is not None:
        if all(col in market_delta_df.columns for col in ["SoM_PMI", "SoM_Comp", "SKU_Delta"]):
            # Recalculate SKU_Delta and compare with the provided values
            recalculated_delta = market_delta_df["SoM_PMI"] - market_delta_df["SoM_Comp"]
            max_diff = (market_delta_df["SKU_Delta"] - recalculated_delta).abs().max()
            avg_diff = (market_delta_df["SKU_Delta"] - recalculated_delta).abs().mean()

            results["delta_calculation"] = {
                "max_difference": max_diff,
                "avg_difference": avg_diff,
                "valid": max_diff < 0.01  # Assuming small differences due to rounding
            }

    return results


def analyze_paris_output(paris_df):
    """Validate PARIS model output consistency."""
    results = {}

    if paris_df is None or paris_df.empty:
        return {"error": "Empty or missing PARIS_Output table"}

    required_cols = ["Real_So_Segment", "Ideal_So_Segment", "Delta_SoS"]
    if not all(col in paris_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in paris_df.columns]
        return {"error": f"Missing required columns: {missing}"}

    # Check Delta_SoS calculation
    recalculated_delta = paris_df["Ideal_So_Segment"] - paris_df["Real_So_Segment"]
    max_diff = (paris_df["Delta_SoS"] - recalculated_delta).abs().max()
    avg_diff = (paris_df["Delta_SoS"] - recalculated_delta).abs().mean()

    results["delta_sos_calculation"] = {
        "max_difference": max_diff,
        "avg_difference": avg_diff,
        "valid": max_diff < 0.01  # Assuming small differences due to rounding
    }

    # Analysis of segment distributions
    for col in ["Real_So_Segment", "Ideal_So_Segment"]:
        col_stats = {
            "mean": paris_df[col].mean(),
            "median": paris_df[col].median(),
            "min": paris_df[col].min(),
            "max": paris_df[col].max(),
            "std": paris_df[col].std(),
            "zeros": (paris_df[col] == 0).sum(),
            "null_values": paris_df[col].isna().sum()
        }
        results[f"{col}_stats"] = col_stats

    # Statistical distribution of Delta_SoS
    delta_stats = {
        "mean": paris_df["Delta_SoS"].mean(),
        "median": paris_df["Delta_SoS"].median(),
        "min": paris_df["Delta_SoS"].min(),
        "max": paris_df["Delta_SoS"].max(),
        "std": paris_df["Delta_SoS"].std(),
        "positive_pct": (paris_df["Delta_SoS"] > 0).mean() * 100,  # Percentage of positive deltas
        "negative_pct": (paris_df["Delta_SoS"] < 0).mean() * 100  # Percentage of negative deltas
    }
    results["delta_sos_stats"] = delta_stats

    return results


def analyze_all_tables(tables_dir):
    """Main function to analyze all tables and generate reports."""
    # Define all expected tables
    expected_tables = [
        "base_list", "calculation_table", "cat_a_df_vols", "category_a_1",
        "CategoryAScores", "CategoryBScores", "CategoryCScores", "CategoryDScores",
        "country_figures", "df_vols_w_metrics", "df_volume", "dom_ims_data",
        "dom_products", "domestic_volumes", "FinalScores", "Flags",
        "green_red_list", "iata_location_map", "LocationVolumes", "Market_Delta",
        "Market_Mix", "Market_Summary_Comp", "Market_Summary_PMI", "MarketSummary",
        "MC_per_Product", "nationality_country_map", "no_of_sku", "PARIS_Output",
        "pax_data", "pmi_margins", "selma_df_map", "selma_dom_map",
        "similarity_matrix", "sku_by_vols_margins"
    ]

    # Load all tables
    tables = {}
    for table_name in expected_tables:
        file_path = os.path.join(tables_dir, f"{table_name}.csv")
        if os.path.exists(file_path):
            tables[table_name] = load_data(file_path)
        else:
            print(f"Warning: {table_name}.csv not found")

    # Initialize results dictionary
    results = {
        "summary": {
            "tables_analyzed": len(tables),
            "total_tables_expected": len(expected_tables),
            "missing_tables": len(expected_tables) - len(tables),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "tables": {},
        "analyses": {}
    }

    # Basic info for each table
    for name, df in tables.items():
        if df is not None:
            results["tables"][name] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "has_nulls": df.isna().any().any(),
                "numeric_columns": list(df.select_dtypes(include=['number']).columns)
            }

    # Detailed analyses
    

    # 1. Category Score Analyses
    category_analyses = {}
    for category in ["A", "B", "C", "D"]:
        table_key = f"Category{category}Scores"
        if table_key in tables and tables[table_key] is not None:
            category_analyses[category] = analyze_category_scores(tables[table_key], category)
    results["analyses"]["category_scores"] = category_analyses

    # 2. Flag Calculation Validation
    if "Flags" in tables and "green_red_list" in tables:
        results["analyses"]["flag_calculations"] = analyze_flag_calculations(
            tables["Flags"], tables["green_red_list"])

    # 3. Financial Metric Analysis
    if "df_vols_w_metrics" in tables:
        results["analyses"]["financial_metrics"] = analyze_financial_metrics(
            tables["df_vols_w_metrics"])

    # 4. Location-Specific Analysis
    results["analyses"]["location_specific"] = analyze_location_specific(tables)

    # 5. Final Scores Analysis
    if "FinalScores" in tables:
        results["analyses"]["final_scores"] = analyze_final_scores(tables["FinalScores"])

    # 6. Segment Calculation Verification
    if "MarketSummary" in tables and "Market_Delta" in tables:
        results["analyses"]["segment_calculations"] = analyze_segment_calculations(
            tables["MarketSummary"], tables["Market_Delta"])

    # 7. PARIS Model Output Validation
    if "PARIS_Output" in tables:
        results["analyses"]["paris_output"] = analyze_paris_output(tables["PARIS_Output"])

    return results


def print_key_findings(results):
    """Print key findings in a structured format."""
    print("\n" + "=" * 80)
    print(f"PORTFOLIO OPTIMIZATION ANALYSIS RESULTS - {results['summary']['timestamp']}")
    print("=" * 80)

    # Overall summary
    print("\nSUMMARY:")
    print(
        f"- Tables analyzed: {results['summary']['tables_analyzed']} of {results['summary']['total_tables_expected']} expected")
    if results['summary']['missing_tables'] > 0:
        print(f"- Missing tables: {results['summary']['missing_tables']}")

    # Category scores
    print("\nCATEGORY SCORE ANALYSIS:")
    if "category_scores" in results["analyses"]:
        for category, analysis in results["analyses"]["category_scores"].items():
            if analysis:  # If we have analysis for this category
                print(f"\nCategory {category}:")
                print(
                    f"- Score range: {analysis.get('min', 'N/A')} to {analysis.get('max', 'N/A')} (mean: {analysis.get('mean', 'N/A'):.2f})")
                if "out_of_range_count" in analysis and analysis["out_of_range_count"] > 0:
                    print(
                        f"- Out of range values: {analysis['out_of_range_count']} ({analysis['out_of_range_pct']:.2f}%)")
                if "distribution" in analysis:
                    dist_str = ", ".join([f"{k}: {v}" for k, v in analysis["distribution"].items()])
                    print(f"- Distribution: {dist_str}")

    # Flag calculations
    print("\nFLAG CALCULATION ANALYSIS:")
    if "flag_calculations" in results["analyses"]:
        flag_analysis = results["analyses"]["flag_calculations"]
        if "flag_counts" in flag_analysis:
            green_count = flag_analysis["flag_counts"].get("Green", 0)
            red_count = flag_analysis["flag_counts"].get("Red", 0)
            print(f"- Total flags: {green_count + red_count} ({green_count} Green, {red_count} Red)")

        if "problem_status_count" in flag_analysis:
            print(
                f"- Problem status items: {flag_analysis['problem_status_count']} ({flag_analysis['problem_status_pct']:.2f}%)")

        if "extremes" in flag_analysis:
            extremes = flag_analysis["extremes"]
            print(
                f"- Max Green flags: {extremes.get('max_green', 'N/A')} (Location: {extremes.get('max_green_location', 'N/A')})")
            print(
                f"- Max Red flags: {extremes.get('max_red', 'N/A')} (Location: {extremes.get('max_red_location', 'N/A')})")

    # Financial metrics
    print("\nFINANCIAL METRIC ANALYSIS:")
    if "financial_metrics" in results["analyses"]:
        fin_analysis = results["analyses"]["financial_metrics"]
        for metric, stats in fin_analysis.items():
            if metric in ["Margin", "Growth"]:
                print(f"\n{metric}:")
                print(f"- Range: {stats['min']:.4f} to {stats['max']:.4f} (mean: {stats['mean']:.4f})")
                if 'negative_values' in stats:
                    # Make this safe by checking for 'count'
                    count = stats.get('count', 1)  # Default to 1 if count is missing
                    if count > 0:  # Avoid division by zero
                        print(f"- Negative values: {stats['negative_values']} ({stats['negative_values']/count*100:.2f}% of non-null)")
                    else:
                        print(f"- Negative values: {stats['negative_values']} (0.00% of non-null)")
                if 'zero_values' in stats:
                    # Same safety check
                    count = stats.get('count', 1)
                    if count > 0:
                        print(f"- Zero values: {stats['zero_values']} ({stats['zero_values']/count*100:.2f}% of non-null)")
                    else:
                        print(f"- Zero values: {stats['zero_values']} (0.00% of non-null)")
                if 'null_values' in stats:
                    print(f"- Null values: {stats['null_values']}")

    # Final scores summary
    print("\nFINAL SCORES SUMMARY:")
    if "final_scores" in results["analyses"]:
        final_analysis = results["analyses"]["final_scores"]
        print(
            f"- Score range: {final_analysis.get('min', 'N/A'):.2f} to {final_analysis.get('max', 'N/A'):.2f} (mean: {final_analysis.get('mean', 'N/A'):.2f})")

        if "distribution" in final_analysis:
            dist_str = ", ".join([f"{k}: {v}" for k, v in final_analysis["distribution"].items()])
            print(f"- Distribution: {dist_str}")

        if "correlations" in final_analysis:
            correlations = final_analysis["correlations"]
            corr_list = sorted([(k, v) for k, v in correlations.items()], key=lambda x: abs(x[1]), reverse=True)
            if corr_list:
                strongest = corr_list[0]
                print(f"- Strongest correlation with final score: {strongest[0]} ({strongest[1]:.4f})")

        if "score_count_distribution" in final_analysis:
            counts = final_analysis["score_count_distribution"]
            print(f"- Locations with all 4 scores: {counts.get(4, 0)}")
            print(f"- Locations with 3 scores: {counts.get(3, 0)}")
            print(f"- Locations with 2 scores: {counts.get(2, 0)}")
            print(f"- Locations with 1 score: {counts.get(1, 0)}")
            print(f"- Locations with 0 scores: {counts.get(0, 0)}")

    # Selected location examples
    print("\nSAMPLE LOCATION ANALYSIS:")
    if "location_specific" in results["analyses"]:
        loc_analysis = results["analyses"]["location_specific"]
        for location, data in loc_analysis.items():
            if "metrics" in data:
                metrics = data["metrics"]
                print(f"\n{location}:")

                # Category scores
                scores = []
                for cat in ["a", "b", "c", "d"]:
                    score_key = f"category_{cat}_score"
                    if score_key in metrics:
                        scores.append(f"Cat_{cat}: {metrics[score_key]:.2f}")

                if scores:
                    print(f"- Scores: {', '.join(scores)}")

                # Flags
                if "green_flags" in metrics or "red_flags" in metrics:
                    green = metrics.get("green_flags", 0)
                    red = metrics.get("red_flags", 0)
                    print(f"- Flags: {green} Green, {red} Red")

                # Final score
                if "final_avg_score" in metrics:
                    print(f"- Final score: {metrics['final_avg_score']:.2f}")


def generate_visualizations(results, tables, output_dir):
    """Generate visualizations for key analyses."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Category score distributions
    if "category_scores" in results["analyses"]:
        category_data = []
        for category, analysis in results["analyses"]["category_scores"].items():
            if "count" in analysis and analysis["count"] > 0:
                table_key = f"Category{category}Scores"
                if table_key in tables and tables[table_key] is not None:
                    df = tables[table_key]
                    score_col = next((col for col in df.columns if col.startswith(f"Cat_{category}")
                                      or col.startswith(f"ScaledScore_{category}")), None)
                    if score_col:
                        scores = df[score_col].dropna()
                        for score in scores:
                            category_data.append({"Category": f"Category {category}", "Score": score})

        if category_data:
            cat_df = pd.DataFrame(category_data)

            plt.figure(figsize=(10, 6))
            sns.boxplot(x="Category", y="Score", data=cat_df)
            plt.title("Distribution of Category Scores")
            plt.ylim(0, 10.5)  # Ensure y-axis shows 0-10 range
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "category_score_distributions.png"))
            plt.close()

    # 2. Final score distribution
    if "FinalScores" in tables and tables["FinalScores"] is not None:
        final_df = tables["FinalScores"]
        if "Avg_Score" in final_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(final_df["Avg_Score"].dropna(), bins=20, kde=True)
            plt.title("Distribution of Final Average Scores")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "final_score_distribution.png"))
            plt.close()

    # 3. Category score correlation matrix
    if "FinalScores" in tables and tables["FinalScores"] is not None:
        final_df = tables["FinalScores"]
        score_cols = [col for col in final_df.columns if col.startswith("Cat_") or col == "Avg_Score"]
        if len(score_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(final_df[score_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
            plt.title("Correlation Between Category Scores")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "category_score_correlations.png"))
            plt.close()

    # 4. Flag counts by location
    if "Flags" in tables and tables["Flags"] is not None:
        flags_df = tables["Flags"]
        if "Location" in flags_df.columns and "FlagType" in flags_df.columns:
            location_flag_counts = flags_df.groupby(["Location", "FlagType"]).size().unstack().fillna(0)

            # Select top 20 locations by total flags for readability
            if len(location_flag_counts) > 20:
                location_flag_counts['Total'] = location_flag_counts.sum(axis=1)
                location_flag_counts = location_flag_counts.sort_values('Total', ascending=False).head(20)
                location_flag_counts = location_flag_counts.drop('Total', axis=1)

            if not location_flag_counts.empty:
                plt.figure(figsize=(12, 8))
                location_flag_counts.plot(kind='bar', stacked=True)
                plt.title("Flag Counts by Location (Top 20)")
                plt.xlabel("Location")
                plt.ylabel("Count")
                plt.xticks(rotation=90)
                plt.legend(title="Flag Type")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "flag_counts_by_location.png"))
                plt.close()


def main():
    """Main function to run the analysis."""
    tables_dir = input("Enter the path to the tables directory: ").strip()

    if not os.path.exists(tables_dir):
        print(f"Error: Directory {tables_dir} does not exist.")
        return

    print(f"Starting analysis of tables in {tables_dir}...")

    # Load tables and run analysis
    tables = {}
    expected_tables = [
        "base_list", "calculation_table", "cat_a_df_vols", "category_a_1",
        "CategoryAScores", "CategoryBScores", "CategoryCScores", "CategoryDScores",
        "country_figures", "df_vols_w_metrics", "df_volume", "dom_ims_data",
        "dom_products", "domestic_volumes", "FinalScores", "Flags",
        "green_red_list", "iata_location_map", "LocationVolumes", "Market_Delta",
        "Market_Mix", "Market_Summary_Comp", "Market_Summary_PMI", "MarketSummary",
        "MC_per_Product", "nationality_country_map", "no_of_sku", "PARIS_Output",
        "pax_data", "pmi_margins", "selma_df_map", "selma_dom_map",
        "similarity_matrix", "sku_by_vols_margins"
    ]

    for table_name in expected_tables:
        file_path = os.path.join(tables_dir, f"{table_name}.csv")
        if os.path.exists(file_path):
            tables[table_name] = load_data(file_path)
            print(f"Loaded {table_name}.csv - {len(tables[table_name])} rows")
        else:
            print(f"Warning: {table_name}.csv not found")

    results = analyze_all_tables(tables_dir)

    # Print key findings
    print_key_findings(results)

    # Generate visualizations
    output_dir = os.path.join(os.path.dirname(tables_dir), "analysis_results")
    generate_visualizations(results, tables, output_dir)

    # Save full results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"portfolio_analysis_results_{timestamp}.json")

    import json
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):  # Add this line to handle NumPy boolean types
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            return super(NpEncoder, self).default(obj)

    with open(results_file, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=2)

    print(f"\nAnalysis complete. Full results saved to {results_file}")
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()