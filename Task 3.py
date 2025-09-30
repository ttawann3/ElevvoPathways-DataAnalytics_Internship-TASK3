import os, re, math, textwrap
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder

def normalize_cols(cols):
    new = []
    for c in cols:
        nc = c.strip()
        nc = re.sub(r"\s+", " ", nc)
        nc = nc.replace("’", "'")
        nc = nc.replace("–", "-")
        nc = nc.lower()
        nc = re.sub(r"[^\w]+", "_", nc)
        nc = re.sub(r"_+", "_", nc).strip("_")
        new.append(nc)
    return new

def multiselect_group(df, qnum):
    prefix = f"q{qnum}_part_"
    parts = [c for c in df.columns if c.startswith(prefix)]
    other = f"q{qnum}_other"
    if other in df.columns:
        parts.append(other)
    return parts

def parse_experience(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if any(x in s for x in ["never written code", "i have never written code"]):
        return 0.0
    if any(x in s for x in ["less than", "< 1", "< 1 years", "< 1 year", "less than 1"]):
        return 0.5
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return (a + b)/2.0
    m = re.search(r"(\d+)\+", s)
    if m:
        a = float(m.group(1))
        return a + 5.0
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return np.nan

def aggregate_multiselect(df, parts):
    cnt = Counter()
    for c in parts:
        if c not in df.columns:
            continue
        vals = df[c].dropna().unique()
        for v in vals:
            vs = str(v).strip()
            if vs in ['', '0', '0.0', 'none', 'None', 'NA', 'nan']:
                continue
            cnt[vs] += df[c].fillna('').apply(lambda x: 1 if str(x).strip()==vs else 0).sum()
    return cnt

def main(input_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)
    orig_shape = df.shape

    # normalize column names
    df.columns = normalize_cols(df.columns)

    # map commonly used columns (if present)
    age_col = 'q1' if 'q1' in df.columns else None
    gender_col = 'q2' if 'q2' in df.columns else None
    country_col = 'q3' if 'q3' in df.columns else None
    education_col = 'q4' if 'q4' in df.columns else None
    job_col = 'q5' if 'q5' in df.columns else None
    experience_col = 'q6' if 'q6' in df.columns else None
    salary_seg_col = 'salary_segment' if 'salary_segment' in df.columns else None
    year_col = 'year' if 'year' in df.columns else None

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop columns with > 60% missing values
    missing_pct = df.isna().mean()
    drop_cols = list(missing_pct[missing_pct > 0.60].index)
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # normalize string columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'nan': np.nan})

    # fill categorical NAs with 'Unknown' for some columns
    cat_fill_cols = [c for c in [gender_col, country_col, education_col, job_col] if c and c in df.columns]
    for c in cat_fill_cols:
        df[c] = df[c].fillna("Unknown")

    # fill numeric columns with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # create numeric years of experience column
    if experience_col and experience_col in df.columns:
        df['yrs_experience_num'] = df[experience_col].apply(parse_experience)
    else:
        df['yrs_experience_num'] = np.nan

    # Aggregate multi-selects we care about: Q7 (languages), Q10 (env), Q14 (viz libs)
    q7_cols = multiselect_group(df, 7)
    q10_cols = multiselect_group(df, 10)
    q14_cols = multiselect_group(df, 14)
    languages_counter = aggregate_multiselect(df, q7_cols) if q7_cols else Counter()
    env_counter = aggregate_multiselect(df, q10_cols) if q10_cols else Counter()
    viz_counter = aggregate_multiselect(df, q14_cols) if q14_cols else Counter()

    # label encode some categorical columns
    le_mappings = {}
    label_cols = [c for c in [job_col, education_col, country_col, gender_col] if c and c in df.columns]
    for c in label_cols:
        le = LabelEncoder()
        df[c + '_enc'] = le.fit_transform(df[c].fillna("Unknown"))
        le_mappings[c] = dict(zip(le.classes_, le.transform(le.classes_)))

    # salary segment numeric mapping
    if salary_seg_col and salary_seg_col in df.columns:
        seg_map = {'Very Low':1, 'Low':2, 'Medium':3, 'High':4, 'Very High':5}
        df['salary_segment_num'] = df[salary_seg_col].map(seg_map)
    else:
        df['salary_segment_num'] = np.nan

    # compute a few insights
    insights = []
    if country_col and country_col in df.columns:
        top_countries = df[country_col].value_counts().head(10)
        insights.append(("Top country by respondents", top_countries.index[0], int(top_countries.iloc[0])))
    if education_col and education_col in df.columns:
        top_edu = df[education_col].value_counts().head(5)
        insights.append(("Top education level", top_edu.index[0], int(top_edu.iloc[0])))
    top_langs = languages_counter.most_common(10)
    if top_langs:
        insights.append(("Top programming language (multi-select)", top_langs[0][0], int(top_langs[0][1])))
    if salary_seg_col and salary_seg_col in df.columns:
        seg_counts = df[salary_seg_col].value_counts()
        top_seg = seg_counts.idxmax()
        insights.append(("Most common salary segment", str(top_seg), int(seg_counts.max())))
    if 'yrs_experience_num' in df.columns and df['yrs_experience_num'].notna().sum()>0 and df['salary_segment_num'].notna().sum()>0:
        corr = df[['yrs_experience_num','salary_segment_num']].dropna().corr().iloc[0,1]
        insights.append(("Correlation (yrs_exp vs salary_segment)", round(float(corr),3), "pearson"))

    # helper to save figures
    saved_figs = []
    def save_fig(fig, name, tight=True):
        path = os.path.join(fig_dir, name)
        if tight:
            fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(path)
        return path

    # Chart 1: Pie - top 7 countries
    if country_col and country_col in df.columns:
        s = df[country_col].value_counts().head(7)
        fig = plt.figure(figsize=(6,6))
        plt.pie(s.values, labels=s.index, autopct='%1.1f%%')
        plt.title("Top 7 Countries by Respondent Share (pie)")
        save_fig(fig, "chart_countries_pie.png")

    # Chart 2: Bar - top 15 job roles
    if job_col and job_col in df.columns:
        s = df[job_col].value_counts().head(15)
        fig = plt.figure(figsize=(10,6))
        plt.bar(s.index.astype(str), s.values)
        plt.xticks(rotation=70, ha='right')
        plt.ylabel("Count")
        plt.title("Top 15 Job Roles (bar)")
        save_fig(fig, "chart_jobs_bar.png")

    # Chart 3: HBar - top programming languages (multi-select)
    if languages_counter:
        most = languages_counter.most_common(20)
        labels = [x for x,_ in most]
        vals = [v for _,v in most]
        fig = plt.figure(figsize=(8,6))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Selections (count)")
        plt.title("Top Programming Languages (multi-select) (hbar)")
        save_fig(fig, "chart_languages_hbar.png")

    # Chart 4: Scatter - years experience vs salary segment
    if 'yrs_experience_num' in df.columns and df['yrs_experience_num'].notna().sum()>50 and df['salary_segment_num'].notna().sum()>50:
        scatter_df = df[['yrs_experience_num','salary_segment_num']].dropna()
        jitter = (np.random.rand(len(scatter_df)) - 0.5) * 0.2
        fig = plt.figure(figsize=(8,6))
        plt.scatter(scatter_df['yrs_experience_num'], scatter_df['salary_segment_num'] + jitter, alpha=0.6, s=10)
        plt.xlabel("Years of Experience (approx)")
        plt.ylabel("Salary Segment (1=VeryLow .. 5=VeryHigh)")
        plt.title("Years Experience vs Salary Segment (scatter)")
        save_fig(fig, "chart_exp_vs_salary_scatter.png")

    # Chart 5: Box - salary by education level
    if education_col and education_col in df.columns and df['salary_segment_num'].notna().sum()>0:
        top_edu_levels = df[education_col].value_counts().head(8).index.tolist()
        box_data = [df.loc[df[education_col]==lvl, 'salary_segment_num'].dropna() for lvl in top_edu_levels]
        fig = plt.figure(figsize=(10,6))
        plt.boxplot(box_data, labels=top_edu_levels, vert=True)
        plt.xticks(rotation=70, ha='right')
        plt.ylabel("Salary Segment Numeric")
        plt.title("Salary Segment by Education Level (box)")
        save_fig(fig, "chart_salary_by_education_box.png")

    # Chart 6: Line - respondents by year
    if year_col and year_col in df.columns:
        s = df[year_col].value_counts().sort_index()
        fig = plt.figure(figsize=(8,5))
        plt.plot(s.index.astype(str), s.values, marker='o')
        plt.title("Number of Respondents by Year (line)")
        plt.xlabel("Survey Year")
        plt.ylabel("Respondent Count")
        save_fig(fig, "chart_respondents_by_year_line.png")

    # Chart 7: Histogram - years of experience distribution
    if 'yrs_experience_num' in df.columns and df['yrs_experience_num'].notna().sum()>0:
        fig = plt.figure(figsize=(8,5))
        plt.hist(df['yrs_experience_num'].dropna(), bins=20)
        plt.title("Distribution of Years of Experience (histogram)")
        plt.xlabel("Years (approx)")
        plt.ylabel("Count")
        save_fig(fig, "chart_yrs_experience_hist.png")

    # Chart 8: Area - cumulative respondents by year
    if year_col and year_col in df.columns:
        s = df[year_col].value_counts().sort_index()
        cum = s.cumsum()
        fig = plt.figure(figsize=(8,5))
        plt.fill_between(cum.index.astype(str), cum.values)
        plt.plot(cum.index.astype(str), cum.values, marker='o')
        plt.title("Cumulative Respondents by Year (area)")
        plt.xlabel("Year")
        plt.ylabel("Cumulative Count")
        save_fig(fig, "chart_cumulative_by_year_area.png")

    # Chart 9: Correlation heatmap
    num_for_corr = ['yrs_experience_num','salary_segment_num']
    num_for_corr = [c for c in num_for_corr if c in df.columns]
    if len(num_for_corr) >= 2:
        corrmat = df[num_for_corr].corr()
        fig = plt.figure(figsize=(6,5))
        plt.imshow(corrmat.values, aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(num_for_corr)), num_for_corr, rotation=45)
        plt.yticks(range(len(num_for_corr)), num_for_corr)
        plt.title("Correlation matrix (heatmap)")
        save_fig(fig, "chart_corr_heatmap.png")

    # Chart 10: Stacked bar - education across top 5 job roles
    if job_col and job_col in df.columns and education_col and education_col in df.columns:
        top_jobs = df[job_col].value_counts().head(5).index.tolist()
        ct = pd.crosstab(df[job_col], df[education_col])
        ct_top = ct.loc[top_jobs]
        fig = plt.figure(figsize=(10,6))
        ct_top.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title("Education distribution across top 5 job roles (stacked bar)")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        save_fig(fig, "chart_education_by_job_stacked.png")

    # Chart 11: Density plot - years experience
    if 'yrs_experience_num' in df.columns and df['yrs_experience_num'].dropna().shape[0] > 50:
        fig = plt.figure(figsize=(8,5))
        df['yrs_experience_num'].dropna().plot(kind='density', ax=plt.gca())
        plt.title("Density of Years of Experience (density plot)")
        plt.xlabel("Years (approx)")
        save_fig(fig, "chart_yrs_experience_density.png")

    # Chart 12: Donut - top 5 languages
    if languages_counter:
        top5 = languages_counter.most_common(5)
        labels = [l for l,_ in top5]
        vals = [v for _,v in top5]
        fig = plt.figure(figsize=(6,6))
        wedges, texts, autotexts = plt.pie(vals, labels=labels, autopct='%1.1f%%')
        centre_circle = plt.Circle((0,0),0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        plt.title("Top 5 Programming Languages (donut)")
        save_fig(fig, "chart_top5_languages_donut.png")

    # Chart 13: Gender pie
    if gender_col and gender_col in df.columns:
        s = df[gender_col].value_counts().head(10)
        fig = plt.figure(figsize=(6,6))
        plt.pie(s.values, labels=s.index, autopct='%1.1f%%')
        plt.title("Gender distribution (pie)")
        save_fig(fig, "chart_gender_pie.png")

    # Chart 14: Education bar
    if education_col and education_col in df.columns:
        s = df[education_col].value_counts().head(12)
        fig = plt.figure(figsize=(10,6))
        plt.bar(s.index.astype(str), s.values)
        plt.xticks(rotation=70, ha='right')
        plt.title("Education level counts (bar)")
        save_fig(fig, "chart_education_bar.png")

    # Chart 15: Top viz libs bar
    if viz_counter:
        most_viz = viz_counter.most_common(15)
        labels = [x for x,_ in most_viz]
        vals = [v for _,v in most_viz]
        fig = plt.figure(figsize=(9,6))
        plt.bar(labels, vals)
        plt.xticks(rotation=70, ha='right')
        plt.title("Top Visualization Libraries (bar)")
        save_fig(fig, "chart_vizlibs_bar.png")

    # Save cleaned CSV
    cleaned_path = os.path.join(out_dir, "cleaned_survey.csv")
    df.to_csv(cleaned_path, index=False)

    # Create PDF report with title page and figures
    report_path = os.path.join(out_dir, "survey_report.pdf")
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11,8.5))
        plt.axis('off')
        title = "Data Science Survey (2018-2021)\nCleaning & Insight Report"
        plt.text(0.5, 0.6, title, ha='center', va='center', fontsize=20, wrap=True)
        subtitle = f"Generated automatically — cleaned rows: {df.shape[0]} (orig {orig_shape[0]}), columns: {df.shape[1]}"
        plt.text(0.5, 0.5, subtitle, ha='center', va='center', fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11,8.5))
        plt.axis('off')
        text_lines = ["Top Insights (automatically extracted):", ""]
        for i,ins in enumerate(insights[:10], start=1):
            text_lines.append(f"{i}. {ins[0]} — {ins[1]} ({ins[2]})")
        plt.text(0.02, 0.98, "\n".join(text_lines), va='top', fontsize=12, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        for p in saved_figs:
            try:
                img = plt.imread(p)
                fig = plt.figure(figsize=(11,8.5))
                plt.axis('off')
                plt.imshow(img)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                fig = plt.figure(figsize=(11,8.5))
                plt.axis('off')
                plt.text(0.5, 0.5, f"Could not include image: {os.path.basename(p)}\\nError: {e}", ha='center')
                pdf.savefig(fig)
                plt.close(fig)

    print("Saved cleaned csv:", cleaned_path)
    print("Saved PDF report:", report_path)
    print("Saved figures to:", fig_dir)
    return {'cleaned_csv': cleaned_path, 'pdf_report': report_path, 'fig_dir': fig_dir, 'saved_figs': saved_figs}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Survey analysis script.")
    parser.add_argument("--input", "-i", default="c:/Users/Abdullah Umer/Desktop/Elevvo Pathways Internship/Task 3/Kaggle Data Science Survey data 2018 to 2021.csv", help="Path to input CSV")
    parser.add_argument("--out", "-o", default="c:/Users/Abdullah Umer/Desktop/Elevvo Pathways Internship/Task 3/outputs", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.out)






















