import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import date, datetime
from tqdm import tqdm
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders


def expand_year_citations(df,
                           year_col="counts_by_year.year",
                           cite_col="counts_by_year.cited_by_count",
                           sep="|"):
    """
    Expand year/citation lists stored as delimited strings
    into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with string columns of delimited years & counts.
    year_col : str
        Name of column with year lists as strings separated by `sep`.
    cite_col : str
        Name of column with citation counts as strings separated by `sep`.
    sep : str
        Separator used in the string (e.g., "|").

    Returns
    -------
    pd.DataFrame
        New DataFrame with added columns like citations_YYYY.
    """
    df = df.copy()

    # Parse into lists
    df["_years"] = df[year_col].str.split(sep)
    df["_counts"] = df[cite_col].str.split(sep)

    # Collect all unique years
    all_years = set()
    for yrs in df["_years"]:
        if isinstance(yrs, list):
            all_years.update(yrs)

    # Create and initialize columns
    for y in sorted(all_years):
        col = f"citations_{y}"
        df[col] = 0

    # Fill columns
    for i, row in df.iterrows():
        yrs = row["_years"]
        cnts = row["_counts"]
        if isinstance(yrs, list) and isinstance(cnts, list):
            for y, c in zip(yrs, cnts):
                col = f"citations_{y}"
                # safe conversion to int
                try:
                    df.at[i, col] = int(c)
                except:
                    pass

    # remove helpers
    df.drop(columns=["_years", "_counts"], inplace=True)

    return df

def get_author_metrics(author_id):
    """
    author_id: OpenAlex author ID (e.g., "A5078165592")
    returns: metrics dict
    """
    a = Authors()[author_id]

    institutions = [affl['institution'] for affl in a.get("affiliations", [])] or []
    countries = set()
    institution_ids = set()

    for inst in institutions:
        iid = inst.get("id")
        if iid:
            institution_ids.add(iid)
        cc = inst.get("country_code")
        if cc:
            countries.add(cc)

    last_inst = a.get("last_known_institution") or {}
    if len(last_inst) == 0:
        last_inst = max(
            a.get("affiliations", []),
            key=lambda affl: max(affl.get("years", []), default=0),
            default={}
        ).get("institution", {})

    metrics = {
        "h_index": a.get("summary_stats", {}).get("h_index"),
        "i10_index": a.get("summary_stats", {}).get("i10_index"),
        "n-citations": a.get("cited_by_count"),
        "n-works": a.get("works_count"),
        "2yr_mean_citedness": a.get("summary_stats", {}).get("2yr_mean_citedness"),

        "institution_id": last_inst.get("id"),
        "institution_name": last_inst.get("display_name"),
        "institution_country": last_inst.get("country_code"),

        "n-institutions": len(institution_ids),
        "n-countries": len(countries),
    }

    return metrics

def get_author_position_metrics(authorship_cell, positions=[0, -1]):
    #fetches all author metrics of a paper (give cell with authors split by |)
    ids = authorship_cell.split("|")
    metrics = {}
    for pos in positions:
        idx = len(ids) + pos if pos < 0 else pos
        if idx < len(ids):
            if idx == 0:
                prefix = "firstauthor"
            elif idx == len(ids) - 1:
                prefix = "lastauthor"
            else:
                prefix = f"author{idx+1}"
            author_metrics = get_author_metrics(ids[idx])
            for k, v in author_metrics.items():
                metrics[f"{prefix}_{k}"] = v

    return metrics

def get_citations_in_year(work_id, year):
    w = Works()[work_id]
    for y in w.get('counts_by_year', []) or []:
        if y.get("year") == year:
            return y.get("cited_by_count", 0)
    return 0

def days_left_in_year(pub_date):
    """
    pub_date: datetime.date or datetime.datetime
    returns: int (number of days remaining in the publication year after pub_date)
    """
    if isinstance(pub_date, datetime):
        pub_date = pub_date.date()

    end_of_year = date(pub_date.year, 12, 31)
    return (end_of_year - pub_date).days

def get_publication_metrics(work_id, ref_date=None, citation_years=[2023,2024,2025]):
    """
    work_id: OpenAlex work ID (e.g., "W2741809807")
    returns: metrics dict
    """
    if ref_date is None:
        ref_date = date.today()
    if isinstance(ref_date, datetime):
        ref_date = ref_date.date()

    w = Works()[work_id]

    # --- authorships: unique institutions & countries ---
    authorships = w.get("authorships", []) or []
    institution_ids = set()
    countries = set()
    author_ids = set()

    for auth in authorships:
        author = auth.get("author", {})
        if author.get("id"):
            author_ids.add(author.get("id"))
        for inst in auth.get("institutions", []) or []:
            if inst.get("id"):
                institution_ids.add(inst.get("id"))
            if inst.get("country_code"):
                countries.add(inst.get("country_code"))

    # --- topics ---
    primary_topic = w.get("primary_topic") or {}
    topics = w.get("topics") or []

    # secondary & tertiary topics by score (excluding primary if duplicated)
    sorted_topics = sorted(
        topics,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    secondary_topic = sorted_topics[1] if len(sorted_topics) > 1 else {}
    tertiary_topic = sorted_topics[2] if len(sorted_topics) > 2 else {}

    # --- MeSH terms (top 5) ---
    mesh_terms = [m for m in w.get("mesh") if m['is_major_topic']] or []
    mesh_names = list(set([m.get("descriptor_name") for m in mesh_terms]))
    mesh1, mesh2, mesh3 = (mesh_names + [np.nan] * 3)[:3]

    # --- publication date ---
    pub_date_raw = w.get("publication_date")
    pub_date = datetime.fromisoformat(pub_date_raw).date() if pub_date_raw else None
    time_since_publication_days = (ref_date - pub_date).days if pub_date else None
    time_since_publication_years = (
        time_since_publication_days / 365.25 if time_since_publication_days is not None else None
    )
    time_since_publication_months = (
        time_since_publication_days / 30.4375 if time_since_publication_days is not None else None
    )
    days_liy = days_left_in_year(pub_date) if pub_date else None
    pub_year = pub_date.year if pub_date else None

    metrics = {
        # basic
        "title": w.get("title"),
        "publication_year": w.get("publication_year"),
        "publication_date": pub_date,
        "article_type": w.get("type"),
        "doi": w.get("doi"),

        # time
        "days_since_publication": time_since_publication_days,
        "months_since_publication": time_since_publication_months,
        "years_since_publication": time_since_publication_years,
        "days_left_in_publication_year": days_liy,

        # counts
        "total_citations": w.get("cited_by_count"),
        "n-authors": len(author_ids),
        "n-institutions": len(institution_ids),
        "n-countries": len(countries),
        "institutions_distinct_count": w.get("institutions_distinct_count"),
        "countries_distinct_count": w.get("countries_distinct_count"),
        "referenced_works_count": w.get("referenced_works_count"),

        # impact
        "fwci": w.get("fwci"),

        # open access
        #"open_access": w.get("open_access", {}).get("is_oa"),
        "oa_status": w.get("open_access", {}).get("oa_status"),

        # topics
        "primary_topic": primary_topic.get("display_name"),
        "primary_topic_subfield": (primary_topic.get("subfield") or {}).get("display_name"),
        "secondary_topic": secondary_topic.get("display_name"),
        "tertiary_topic": tertiary_topic.get("display_name"),
        # mesh
        "mesh1": mesh1,
        "mesh2": mesh2,
        "mesh3": mesh3,
    }

    for yr in citation_years:
        cit_count = get_citations_in_year(work_id, yr)
        metrics[f"citations_{yr}"] = cit_count
        if yr==pub_year:
            #adjust for partial year
            adj_cit = cit_count * (365 / days_liy) if days_liy and days_liy>0 else cit_count
            metrics[f"citations_{yr}_adjusted"] = adj_cit
        elif yr>pub_year:
            #compute adjusted citations per year since publication
            years_since_pub = (yr - pub_year)*365 + days_liy
            adj_cit = cit_count * (365 / years_since_pub) if years_since_pub and years_since_pub>0 else cit_count
            metrics[f"citations_{yr}_adjusted"] = adj_cit

    return metrics

def add_dummified_column(df, column, prefix=None, drop_original=False):
    """
    Add dummified columns to a DataFrame for a specified column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The name of the column to dummify.
    prefix : str, optional
        Prefix for the dummy column names. Defaults to the column name.
    drop_original : bool, optional
        Whether to drop the original column after dummification. Defaults to False.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with dummified columns added.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    prefix = prefix or column
    dummies = pd.get_dummies(df[column], prefix=f'{prefix}_{column}_')
    dummies.columns = [d.replace(' ', '').replace(',','').replace(':','').replace('.','').replace('-','') for d in dummies.columns]

    df = pd.concat([df, dummies], axis=1)

    if drop_original:
        df = df.drop(columns=[column])

    return df


def extract_mesh_and_topics(work_ids, min_count=20):
    """
    work_ids: list of OpenAlex work IDs
    returns: DataFrame with columns ['work_id', 'mesh_terms', 'topics']
    """
    rows = []

    for wid in tqdm(work_ids, 'Extracting MeSH and topics'):
        w = Works()[wid]

        mesh_terms = [
            m.get("descriptor_name")
            for m in (w.get("mesh") or [])
            if m.get("descriptor_name")
        ]

        topics = [
            t.get("display_name")
            for t in (w.get("topics") or [])
            if t.get("display_name")
        ]

        rows.append({
            "work_id": wid,
            "all_mesh": mesh_terms,
            "all_topics": topics,
        })

    all_terms = pd.DataFrame(rows)

    #create mesh and topic dummy variables
    all_mesh = all_terms['all_mesh'].explode().dropna()
    all_topics = all_terms['all_topics'].explode().dropna()
    mesh_counts = all_mesh.value_counts()
    topic_counts = all_topics.value_counts()
    mesh_vars = mesh_counts[mesh_counts > min_count].index
    topic_vars = topic_counts[topic_counts > min_count].index

    #create dummy for all in all_terms with yes no for mesh and topic vars
    for mv in mesh_vars:
        res = all_terms['all_mesh'].apply(lambda x: 1 if mv in x else 0)
        all_terms["mesh_{}".format(mv.replace(' ', '').replace(',','').replace(':','').replace('.','').replace('-',''))] = res
    for tv in topic_vars:
        res = all_terms['all_topics'].apply(lambda x: 1 if tv in x else 0)
        all_terms["topic_{}".format(tv.replace(' ', '').replace(',','').replace(':','').replace('.','').replace('-',''))] = res

    return all_terms


def counts_and_percentages(df, columns, id_col=None):
    # Ensure columns exist
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame.")

    summary = []

    # compute row_totals only if no id_col
    if id_col is not None:
        total = len(df[id_col].unique())

    for col in columns:
        count = df[col].sum()

        if id_col is None:
            total = count

        pct = count / total if total != 0 else 0

        summary.append({
            "variable": col,
            "count": count,
            "total": total,
            "percentage": pct,
        })

    return pd.DataFrame(summary)


def plot_regressions(
    df: pd.DataFrame,
    targets: list[str],
    xvars: list[str] = None,
    vartypes: dict = None,
    plot_formula: bool = True,
    dir_fig: str = None,
    figsize=(6, 5),
    dpi=80,
    scatter_kwargs=None,
    line_kwargs=None,
    text_kwargs=None,
):
    """
    Plot regression scatter + line for specified x variables vs specified target variables.

    Args:
        df (pd.DataFrame): Input dataframe.
        targets (list[str]): One or more target variable names.
        xvars (list[str], optional): Predictors to plot. If None, uses all numeric columns except targets.
        plot_formula (bool): Whether to annotate the regression formula on the plot.
        dir_fig (str | None): Directory where figures will be saved. If None, figures are not saved.
        figsize (tuple): Size of each output figure.
        dpi (int): DPI for saved figures.
        scatter_kwargs (dict): Extra kwargs passed to scatterplot.
        line_kwargs (dict): Extra kwargs passed to regression line plot.
        text_kwargs (dict): Extra kwargs passed to text annotation.

    Saves:
        One PNG per (x, target) pair if dir_fig is provided.
    """

    # Validate targets
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"Target variable '{t}' not in DataFrame.")

    # Default plotting options
    scatter_kwargs = scatter_kwargs or {"s": 20, "alpha": 0.7}
    line_kwargs = line_kwargs or {"color": "red", "linewidth": 2}
    text_kwargs = text_kwargs or {"fontsize": 9, "color": "black"}

    # Determine predictors
    if xvars is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        xvars = [col for col in numeric_cols if col not in targets]
    else:
        missing_x = [col for col in xvars if col not in df.columns]
        if missing_x:
            raise ValueError(f"xvars not in DataFrame: {missing_x}")

    if len(xvars) == 0:
        xvars = df.columns

    # Loop through all xvars and targets
    for target in targets:
        for x in xvars:
            #print(x, vartypes[x], target)
            if dir_fig is not None:
                dir_sav = os.path.join(dir_fig, target)
                os.makedirs(dir_sav, exist_ok=True)

            if x==target:
                continue

            # Prepare data
            sub = df[[x, target]].dropna()

            if not (pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[target])):
                continue

            X = sub[x].values
            Y = sub[target].values

            if len(X) < 2:
                print(f"Skipping {x} vs {target}: not enough data points.")
                continue

            if vartypes[x]=='continuous':
                # Fit linear regression
                try:
                    a, b = np.polyfit(X, Y, 1)
                except Exception as e:
                    print(f"Error fitting regression for {x} vs {target}:\n{e}")
                    continue

                # Create plot
                plt.figure(figsize=figsize, dpi=dpi)
                sns.scatterplot(x=X, y=Y, **scatter_kwargs)

                # Regression line
                xs = np.linspace(X.min(), X.max(), 100)
                ys = a * xs + b
                plt.plot(xs, ys, **line_kwargs)

                # Annotate formula
                if plot_formula:
                    y_pred = a * X + b

                    # R²
                    ss_res = np.sum((Y - y_pred) ** 2)
                    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
                    r2 = 1 - ss_res / ss_tot

                    formula_text = f"y = {a:.3f}·x + {b:.3f}"
                    r2_text = f"R² = {r2:.3f}"

                    plt.text(
                        0.05,
                        0.85,
                        f"{formula_text}\n{r2_text}",
                        transform=plt.gca().transAxes,
                        **text_kwargs
                    )

                # Labels + title
                plt.xlabel(x)
                plt.ylabel(target)
                plt.title(f"{x} vs {target}")
                plt.tight_layout()
            elif vartypes[x]=='categorical':
                plt.figure(figsize=figsize, dpi=dpi)
                sns.boxplot(x=X, y=Y, showfliers=False)
                plt.xlabel(x)
                plt.ylabel(target)
                plt.title(f"{x} vs {target}")
                plt.tight_layout()


            # Save if requested
            if dir_fig is not None:
                fname = f"{x}_vs_{target}.png"
                path = os.path.join(dir_sav, fname)
                plt.savefig(path)

            plt.close()

def remove_outliers_iqr(df, columns, k=3):
    """
    Remove outliers from a DataFrame column using the IQR rule.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column to filter
    k : float
        IQR multiplier (default 1.5)

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe without outliers
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        df[column][(df[column] < lower) | (df[column] > upper)] = np.nan

    return df

if __name__ == '__main__':
    #pip install pyalex
    file = '/home/hvv/Documents/reviews/AJNR/publication_stats/screened_AJNR_publications_2023.xlsx'
    dir_out = os.path.dirname(file)
    file_out = os.path.join(dir_out, 'AJNR_citation_reg_inpv2.xlsx')
    df = pd.read_excel(file)#.rename(columns=cols)

    to_dummy_cols = ['article_type',
                        'primary_topic', 'primary_topic_subfield',
                        'secondary_topic', 'tertiary_topic',
                        'mesh1', 'mesh2', 'mesh3',
                        'firstauthor_institution_name',
                        'firstauthor_institution_country',
                        'lastauthor_institution_name',
                        'lastauthor_institution_country',
                         ]

    if os.path.exists(file_out):
        data = pd.read_excel(file_out, sheet_name='data')
        variable_types = pd.read_excel(file_out, sheet_name='variable_types')
        continuous_vars = variable_types['continuous'].dropna().tolist()
        cat_vars = variable_types['categorical'].dropna().tolist()

        cols = data.columns[data.dtypes==object]

    else:
        out = []
        for work_id, author_ids in tqdm(zip(df['id'], df['authorships.author.id'])):
            author_metrics = get_author_position_metrics(author_ids, positions=[0,1,2,3, -1])
            work_metrics = get_publication_metrics(work_id, ref_date=date(2026, 1, 1), citation_years=[2023,2024,2025])
            out.append([work_id, author_ids, *work_metrics.values(), *author_metrics.values()])

        cols = ['work_id', 'author_ids'] + list(work_metrics.keys()) + list(author_metrics.keys())
        data = pd.DataFrame(out, columns=cols)
        data = data[~np.isin(data['article_type'], ['book-chapter', 'book', 'dataset', 'editorial', 'letter'])]
        #add dummy variables
        for col in to_dummy_cols:
            if col not in data:
                print(f"Column {col} not found in data for dummification.")
                continue
            data = add_dummified_column(data, column=col, prefix='cat', drop_original=False)
        #also dummy variables for mesh terms and topics
        all_terms= extract_mesh_and_topics(data['work_id'].tolist())
        cat_vars = [c for c in all_terms.columns if c.startswith('mesh_') or c.startswith('topic_')]
        cat_vars.extend([c for c in data if 'cat_' in c])

        data = data.merge(all_terms, left_on='work_id', right_on='work_id', how='left')
        continuous_vars = [data.columns[i] for i in range(len(data.columns)) if data.dtypes[i] not in [str, list, tuple] and data.columns[i] not in cat_vars]

        vartypes = pd.DataFrame([continuous_vars, cat_vars], index=['continuous', 'categorical']).T

        with pd.ExcelWriter(file_out) as writer:
            # Write df_out to the first sheet
            data.to_excel(writer, sheet_name='data', index=False)

            # Calculate missing values count and write to the second sheet
            missing_values = data.isnull().sum()
            missing_values.to_frame(name='missing_count').to_excel(writer, sheet_name='missing_values')

            data.describe().to_excel(writer, sheet_name='descriptive_stats')
            vartypes.to_excel(writer, sheet_name='variable_types', index=False)

            counts_and_percentages(data, cat_vars,
                                   id_col='work_id').to_excel(writer,sheet_name='cat_descriptives')

    cat_vars_select = [c for c in cat_vars if data[c].sum()>20]
    continuous_vars_select = [c for c in continuous_vars if not (('citations' in c or 'fwci' in c) and not 'author' in c)] #exclude citation counts and fwci from continuous vars to plot (as they are targets)

    target_vars = ["total_citations","fwci",
                  "citations_2023",
                  "citations_2023_adjusted",
                  "citations_2024",
                  "citations_2024_adjusted",
                  "citations_2025",
                  "citations_2025_adjusted"]

    vartype_dct = {}
    for c in data.columns:
        if c in continuous_vars_select:
            vartype_dct[c] = 'continuous'
        elif c in cat_vars_select:
            vartype_dct[c] = 'categorical'

    data = remove_outliers_iqr(data, continuous_vars_select, k=3)
    plot_regressions(data,
                    xvars=list(vartype_dct.keys()),
                    targets=target_vars,
                    vartypes=vartype_dct,
                    dir_fig=os.path.join(dir_out,'figures2'),
                    plot_formula=True)



    #plot associations of variables with tot citations and early citations


    #run R regression with variale selection

    print(1)