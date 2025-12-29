import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        if 0 <= idx < len(ids):
            prefix = "firstauthor" if idx == 0 else "lastauthor" if idx == len(ids) - 1 else f"author{idx+1}"
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


if __name__ == '__main__':
    #pip install pyalex
    file = '/home/hvv/Documents/reviews/AJNR/publication_stats/screened_AJNR_publications_2023.xlsx'
    dir_out = os.path.dirname(file)
    file_out = os.path.join(dir_out, 'AJNR_citation_reg_inp.xlsx')
    df = pd.read_excel(file)#.rename(columns=cols)

    out = []
    for work_id, author_ids in tqdm(zip(df['id'], df['authorships.author.id'])):
        author_metrics = get_author_position_metrics(author_ids, positions=[0,1,2 -1])
        work_metrics = get_publication_metrics(work_id, ref_date=date(2026, 1, 1), citation_years=[2023,2024,2025])
        out.append([work_id, author_ids, *work_metrics.values(), *author_metrics.values()])

    cols = ['work_id', 'author_ids'] + list(work_metrics.keys()) + list(author_metrics.keys())
    df_out = pd.DataFrame(out, columns=cols)
    with pd.ExcelWriter(file_out) as writer:
        # Write df_out to the first sheet
        df_out.to_excel(writer, sheet_name='data', index=False)

        # Calculate missing values count and write to the second sheet
        missing_values = df_out.isnull().sum()
        missing_values.to_frame(name='missing_count').to_excel(writer, sheet_name='missing_values')

    """
    Check if we can extract all DOIs from Karen

    Plan:
    Outcomes: citation count, field weighted citation index, citation percentile
    X variables:
    - topic1
    - secondary topics
    - institution
    - country
    - keyword
    - concept
    - mesh
    
    first and last author institution/country

    To make variables:
    - H-index first/last avg authors
    - I-index first/last avg authors
    - time since publication
    - citation factors of references

    """

    print(1)