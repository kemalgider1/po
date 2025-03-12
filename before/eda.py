import pandas as pd
import numpy as np
import os
import glob
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


def get_stats(df):
    """Get summary statistics for all numeric columns in dataframe."""
    stats = {}
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        col_stats = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'non_zero': (df[col] != 0).sum(),
            'null_count': df[col].isna().sum()
        }
        stats[col] = col_stats

    return stats


def analyze_table(file_path):
    """Analyze a single table file and return its summary."""
    file_name = os.path.basename(file_path)
    df = load_data(file_path)

    if df is None:
        return {
            'file_name': file_name,
            'status': 'Error loading file'
        }

    result = {
        'file_name': file_name,
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': list(df.columns),
        'sample_data': df.head(5).to_dict(orient='records'),
        'stats': get_stats(df),
        'missing_values': df.isna().sum().to_dict()
    }

    return result


def compare_location_scores(tables_dir):
    """Compare scores for same locations across different category files."""
    category_files = [
        "CategoryAScores.csv",
        "CategoryBScores.csv",
        "CategoryCScores.csv",
        "CategoryDScores.csv",
        "FinalScores.csv"
    ]

    score_dfs = {}
    for file in category_files:
        path = os.path.join(tables_dir, file)
        if os.path.exists(path):
            df = load_data(path)
            if df is not None:
                score_dfs[file] = df

    # Check if we have all category files
    if len(score_dfs) < len(category_files):
        missing = set(category_files) - set(score_dfs.keys())
        print(f"Warning: Missing category files: {missing}")

    # Try to find location column (might be named differently across files)
    location_cols = {}
    for file, df in score_dfs.items():
        loc_col = None
        for col in df.columns:
            if col.lower() in ['location', 'city', 'country', 'destination', 'region']:
                loc_col = col
                break

        # If no obvious location column, take first column as likely candidate
        if loc_col is None and len(df.columns) > 0:
            loc_col = df.columns[0]

        location_cols[file] = loc_col

    # Find common locations
    if score_dfs and all(location_cols.values()):
        locations_sets = []
        for file, df in score_dfs.items():
            loc_col = location_cols[file]
            locations_sets.append(set(df[loc_col].astype(str)))

        common_locations = set.intersection(*locations_sets) if locations_sets else set()

        # Compare scores for common locations
        comparison = []
        for location in common_locations:
            location_data = {'Location': location}

            for file, df in score_dfs.items():
                loc_col = location_cols[file]
                row = df[df[loc_col].astype(str) == location]

                if not row.empty:
                    # Get score columns (likely to contain 'score' in name)
                    score_cols = [col for col in df.columns if 'score' in col.lower() or 'rank' in col.lower()]
                    # If no obvious score columns, use all numeric columns except the location
                    if not score_cols:
                        score_cols = df.select_dtypes(include=['number']).columns.tolist()

                    # Add scores to the location data
                    for col in score_cols:
                        location_data[f"{file.replace('.csv', '')}_{col}"] = row[col].iloc[0]

            comparison.append(location_data)

        return pd.DataFrame(comparison)

    return pd.DataFrame()


def trace_calculation_for_location(tables_dir, location):
    """Trace calculation chain for a specific location through all relevant tables."""
    all_files = glob.glob(os.path.join(tables_dir, "*.csv"))

    location_data = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        df = load_data(file_path)

        if df is None:
            continue

        # Try to find location column
        loc_col = None
        for col in df.columns:
            if col.lower() in ['location', 'city', 'country', 'destination', 'region']:
                loc_col = col
                break

        # If location column found, check for the specified location
        if loc_col is not None:
            # Convert to string for comparison
            df[loc_col] = df[loc_col].astype(str)
            row = df[df[loc_col].str.lower() == location.lower()]

            if not row.empty:
                location_data[file_name] = row.to_dict(orient='records')[0]

    return location_data


def analyze_tables(tables_dir):
    """Analyze all specified tables and produce a comprehensive report."""
    # Core input tables
    core_tables = [
        "MC_per_Product.csv",
        "cat_a_df_vols.csv",
        "selma_dom_map.csv",
        "base_list.csv"
    ]

    # Intermediate calculation tables
    intermediate_tables = [
        "df_vols_w_financials.csv",
        "df_vols_w_metrics.csv",
        "Market_Mix.csv",
        "MarketSummary.csv",
        "PARIS_Output.csv"
    ]

    # Final scoring tables
    scoring_tables = [
        "CategoryAScores.csv",
        "CategoryBScores.csv",
        "CategoryCScores.csv",
        "CategoryDScores.csv",
        "FinalScores.csv"
    ]

    # All specified tables
    all_targeted_tables = core_tables + intermediate_tables + scoring_tables

    # Results dictionary
    results = {
        'core_tables': {},
        'intermediate_tables': {},
        'scoring_tables': {}
    }

    # Analyze core tables
    for table in core_tables:
        file_path = os.path.join(tables_dir, table)
        if os.path.exists(file_path):
            results['core_tables'][table] = analyze_table(file_path)
        else:
            results['core_tables'][table] = {'status': 'File not found'}

    # Analyze intermediate tables
    for table in intermediate_tables:
        file_path = os.path.join(tables_dir, table)
        if os.path.exists(file_path):
            results['intermediate_tables'][table] = analyze_table(file_path)
        else:
            results['intermediate_tables'][table] = {'status': 'File not found'}

    # Analyze scoring tables
    for table in scoring_tables:
        file_path = os.path.join(tables_dir, table)
        if os.path.exists(file_path):
            results['scoring_tables'][table] = analyze_table(file_path)
        else:
            results['scoring_tables'][table] = {'status': 'File not found'}

    # Get location comparison across scoring tables
    location_comparison = compare_location_scores(tables_dir)

    # Find a location with significant differences for detailed analysis
    # This is a heuristic approach - we're looking for locations where final scores differ significantly
    if not location_comparison.empty and 'Location' in location_comparison.columns:
        # Try to find columns with 'final' in them for comparison
        final_cols = [col for col in location_comparison.columns if 'final' in col.lower()]

        if final_cols:
            # Analyze spread across final score columns
            location_comparison['score_range'] = location_comparison[final_cols].max(axis=1) - location_comparison[
                final_cols].min(axis=1)
            # Get location with biggest range
            most_different_location = location_comparison.loc[location_comparison['score_range'].idxmax()]['Location']
        else:
            # Just take the first location if no 'final' columns
            most_different_location = location_comparison['Location'].iloc[0]

        # Trace calculations for the identified location
        location_trace = trace_calculation_for_location(tables_dir, most_different_location)

        return {
            'table_analyses': results,
            'location_comparison': location_comparison.to_dict(orient='records'),
            'detailed_trace': {
                'location': most_different_location,
                'trace_data': location_trace
            }
        }

    # Return without location comparison if we couldn't perform it
    return {
        'table_analyses': results,
        'location_comparison': None,
        'detailed_trace': None
    }


def print_formatted_report(analysis_results):
    """Print a formatted report of the analysis results."""
    print("\n===== PORTFOLIO ANALYSIS COMPARISON REPORT =====\n")

    # Print core tables analysis
    print("\n----- CORE INPUT TABLES -----\n")
    for table_name, table_data in analysis_results['table_analyses']['core_tables'].items():
        if 'status' in table_data and table_data['status'] != 'File not found':
            print(f"Table: {table_name}")
            print(f"Row count: {table_data.get('row_count', 'N/A')}")
            print(f"Columns: {', '.join(table_data.get('columns', []))}")

            print("\nSample data:")
            sample = table_data.get('sample_data', [])
            if sample:
                for i, row in enumerate(sample[:5]):
                    if i == 0:
                        print(" | ".join([f"{k}" for k in row.keys()]))
                        print("-" * (sum([len(k) for k in row.keys()]) + 3 * (len(row.keys()) - 1)))
                    print(" | ".join([f"{v}" for v in row.values()]))

            print("\nStats:")
            stats = table_data.get('stats', {})
            for col, col_stats in stats.items():
                print(f"  {col}: min={col_stats.get('min', 'N/A'):.2f}, max={col_stats.get('max', 'N/A'):.2f}, "
                      f"mean={col_stats.get('mean', 'N/A'):.2f}, non-zero={col_stats.get('non_zero', 'N/A')}")

            print("\n" + "=" * 50 + "\n")
        else:
            print(f"Table: {table_name} - {table_data.get('status', 'Unknown status')}\n")

    # Print intermediate tables analysis (simplified)
    print("\n----- INTERMEDIATE CALCULATION TABLES -----\n")
    for table_name, table_data in analysis_results['table_analyses']['intermediate_tables'].items():
        if 'status' in table_data and table_data['status'] != 'File not found':
            print(f"Table: {table_name}")
            print(f"Row count: {table_data.get('row_count', 'N/A')}")
            print(f"Column count: {table_data.get('column_count', 'N/A')}")
            print(f"Columns: {', '.join(table_data.get('columns', []))[:100]}...")  # Truncate long column lists

            print("\n" + "-" * 50 + "\n")
        else:
            print(f"Table: {table_name} - {table_data.get('status', 'Unknown status')}\n")

    # Print scoring tables analysis
    print("\n----- FINAL SCORING TABLES -----\n")
    for table_name, table_data in analysis_results['table_analyses']['scoring_tables'].items():
        if 'status' in table_data and table_data['status'] != 'File not found':
            print(f"Table: {table_name}")
            print(f"Row count: {table_data.get('row_count', 'N/A')}")
            print(f"Columns: {', '.join(table_data.get('columns', []))}")

            print("\nSample data:")
            sample = table_data.get('sample_data', [])
            if sample:
                for i, row in enumerate(sample[:5]):
                    if i == 0:
                        print(" | ".join([f"{k}" for k in row.keys()]))
                        print("-" * (sum([len(k) for k in row.keys()]) + 3 * (len(row.keys()) - 1)))
                    print(" | ".join([f"{v}" for v in row.values()]))

            print("\nStats:")
            stats = table_data.get('stats', {})
            for col, col_stats in stats.items():
                print(f"  {col}: min={col_stats.get('min', 'N/A'):.2f}, max={col_stats.get('max', 'N/A'):.2f}, "
                      f"mean={col_stats.get('mean', 'N/A'):.2f}, non-zero={col_stats.get('non_zero', 'N/A')}")

            print("\n" + "=" * 50 + "\n")
        else:
            print(f"Table: {table_name} - {table_data.get('status', 'Unknown status')}\n")

    # Print location comparison
    if analysis_results['location_comparison']:
        print("\n----- LOCATION SCORE COMPARISON -----\n")
        location_comp = analysis_results['location_comparison']

        # Print header
        if location_comp:
            headers = location_comp[0].keys()
            print(" | ".join([f"{h}" for h in headers]))
            print("-" * (sum([len(h) for h in headers]) + 3 * (len(headers) - 1)))

            # Print first few location comparisons
            for row in location_comp[:5]:
                print(" | ".join([f"{v}" for v in row.values()]))

        print(f"\nTotal locations compared: {len(location_comp)}")

    # Print detailed trace for a specific location
    if analysis_results['detailed_trace']:
        trace = analysis_results['detailed_trace']
        print(f"\n----- DETAILED CALCULATION TRACE FOR {trace['location']} -----\n")

        for table_name, table_data in trace['trace_data'].items():
            print(f"Table: {table_name}")
            for k, v in table_data.items():
                print(f"  {k}: {v}")
            print()


def main():
    """Main function to run the analysis."""
    tables_dir = "/Users/kemalgider/Desktop/PORTFOLIO OPT./All_tables/"

    # Check if directory exists
    if not os.path.exists(tables_dir):
        print(f"Error: Directory {tables_dir} does not exist.")
        return

    print(f"Starting analysis of tables in {tables_dir}...")
    analysis_results = analyze_tables(tables_dir)
    print_formatted_report(analysis_results)

    # Save detailed results to CSV files for further analysis
    output_dir = os.path.join(os.path.dirname(tables_dir), "analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    # Save location comparison
    if analysis_results['location_comparison']:
        location_comp_df = pd.DataFrame(analysis_results['location_comparison'])
        location_comp_df.to_csv(os.path.join(output_dir, "location_comparison.csv"), index=False)
        print(f"\nLocation comparison saved to {os.path.join(output_dir, 'location_comparison.csv')}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()