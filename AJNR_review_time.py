import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

col_name_dict = {
    'Submission Date - Original': 'org_sub_date',
    'Manuscript ID - Original': 'man_id',
    'Manuscript Title': 'man_title',
    'Primary Manuscript Sub-Type': 'man_sub_type',
    'Editor Full Name': 'editor',
    '# Reviewers Invited': 'n_invited',
    '# Reviewers Assigned': 'n_assigned',
    'Reviewer Person ID': 'reviewer_id',
    'Date Reviewer Invited': 'date_invited',
    'Date Reviewer Responded': 'date_responded',
    'Reviewer Invitation Response': 'invitation_response',
    'Date Score Sheet Completed': 'date_scoresheet',
    '# Days From Review Assignment to Completion': 'time_review_complete_days',
    'Quality of the Review': 'quality_review',
}


"""
steps I take:
1) Cleanup: 0 assigned --> is rebuttal, datetime adjustment
2) groupby: unique reviewer, unique editor
3) metrics: responsetime in hours (per reviewer/topic/editor)
--> how many reviews by reviewers
--> time to accept/decline --> rate of accept/decline
--> time to finish review
--> quality review

"""

def performance_on_group(df, group_col, performance_cols, response_time_cols=None):

    data = df[df['date_scoresheet'].notna()]
    counts = data.groupby(group_col).count().sort_values(by='date_scoresheet', ascending=False)
    out = pd.DataFrame(index=counts.index, data=counts['date_scoresheet'].values, columns=['n_reviews'])
    tmp = pd.DataFrame(df[group_col].unique(), index = df[group_col].unique())
    out  = pd.merge(out, tmp, left_index=True, right_index=True, how='outer').drop(columns=tmp.columns)

    mn_perf = data.groupby(group_col)[performance_cols].mean()
    quality_counts = pd.crosstab(data[group_col], data['quality_review'])
    quality_percentage = quality_counts.div(quality_counts.sum(axis=1), axis=0)
    quality_percentage.columns = [c+'_percentage' for c in quality_percentage]
    quality_combined = quality_counts.astype(str) + " (" + quality_percentage.round().astype(str) + "%)"

    #add accept/decline rates
    #accept_decline = df.groupby(group_col)['invitation_response'].value_counts().unstack(fill_value=0)
    ad = pd.crosstab(df[group_col], df['invitation_response'])
    ad_pct = ad.div(ad.sum(axis=1), axis=0)
    ad_pct.columns = [c + '_percentage' for c in ad_pct]

    out = pd.concat([out, mn_perf, quality_counts, quality_percentage, ad, ad_pct], axis=1)

    if response_time_cols is not None:
        rt_med = data.groupby(group_col)[response_time_cols].median()  #data[[group_col,*response_time_cols]]
        rt_med.columns = [c+'_med' for c in rt_med.columns]
        rt_p75 = data.groupby(group_col)[response_time_cols].quantile(0.75)
        rt_p75.columns = [c + '_p75' for c in rt_p75.columns]
        rt_p90 = data.groupby(group_col)[response_time_cols].quantile(0.9)
        rt_p90.columns = [c + '_p90' for c in rt_p90.columns]
        rt_p99 = data.groupby(group_col)[response_time_cols].quantile(0.99)
        rt_p99.columns = [c + '_p99' for c in rt_p99.columns]

        out = pd.concat([out, rt_med, rt_p75, rt_p90, rt_p99], axis=1)

    out.sort_values(by='n_reviews', ascending=False, inplace=True)

    return out

def response_per_n_reviewers(df):
    accepted_responses = df[df['invitation_response'] == 'Agreed']

    # Sort by manuscript ID and invitation date
    accepted_responses = accepted_responses.sort_values(by=['man_id', 'date_invited'])

    # Compute the cumulative response times for each manuscript
    accepted_responses['response_rank'] = accepted_responses.groupby('man_id').cumcount() + 1

    # Pivot to get response times for the first, second, third, and fourth reviewers
    response_times = accepted_responses.pivot(index='man_id', columns='response_rank', values='initial_response_h')

    # Rename columns for clarity
    response_times.columns = [f'response_to_{int(col)}_h' for col in response_times.columns]

    return response_times


def excel_multtabs(dfs, file):
    with pd.ExcelWriter(file) as writer:
        for c, file in dfs.items():
            file.to_excel(writer, sheet_name=c)
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file = '/home/hvv/Documents/reviews/AJNR/reviewer_stats/Reviewer Stats_v2.xlsx'
    file_out = '/home/hvv/Documents/reviews/AJNR/reviewer_stats/grouped_stats.xlsx'
    file_out2 = '/home/hvv/Documents/reviews/AJNR/reviewer_stats/many_rev_grouped_stats.xlsx'
    df = pd.read_excel(file).rename(columns=col_name_dict)
    df = df[df['n_assigned']>0]
    datetime_cols = [c for c in df.columns if 'date' in c]
    df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime)

    df['inv_ass_ratio'] = df['n_invited']/df['n_assigned']
    df['initial_response_time'] = df['date_responded']-df['date_invited']
    df['initial_response_h'] = df['initial_response_time'].dt.total_seconds() / 3600
    df["more_4_revs"] = df['n_assigned']>4
    df["more_6_revs"] = df['n_assigned']>6
    df['all'] = np.ones(len(df))

    response_times = response_per_n_reviewers(df)
    rt_cols = response_times.columns[:6]
    df = df.merge(response_times, on='man_id', how='left')

    #for mid in df['man_id'].unique():
    df4 = df[df["more_4_revs"]]
    df6 = df[df["more_6_revs"]]

    groupby_cols = ['reviewer_id', 'editor', 'man_sub_type', 'all']
    dfs = {}
    dfs2 = {}
    for gbc in groupby_cols:
        dfs[gbc] = performance_on_group(df,
                                     group_col=gbc,
                                     performance_cols=['initial_response_h', 'time_review_complete_days'],
                                     response_time_cols=rt_cols)

        #df4 and df5 represent manuscripts with >4 and >6 reviewers
        dfs2[f'{gbc}-4'] = performance_on_group(df4,
                                     group_col=gbc,
                                     performance_cols=['initial_response_h', 'time_review_complete_days'])

        dfs2[f'{gbc}-6'] = performance_on_group(df6,
                                     group_col=gbc,
                                     performance_cols=['initial_response_h', 'time_review_complete_days'])


    excel_multtabs(dfs, file_out)
    excel_multtabs(dfs2, file_out2)



    print(1)

