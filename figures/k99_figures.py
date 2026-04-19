
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cta_mra_trends(
    file_path,
    year_col='Year',
    brain_cta_col='Brain CTA',
    other_cta_col='Other organ CTA',
    brain_mra_col='Brain MRA',
    other_mra_col='Other organ MRA'
):
    # ---- LOAD ----
    df = pd.read_excel(file_path, index_col=0).T
    df[year_col] = df.index.astype(int)

    # ---- PLOT ----
    plt.figure(figsize=(5, 5))

    plt.plot(df[year_col], df[brain_cta_col], label='Brain CTA', marker='s')
    plt.plot(df[year_col], df[other_cta_col], label='Other organ CTA', marker='x')
    plt.plot(df[year_col], df[brain_mra_col], label='Brain MRA', marker='s')
    plt.plot(df[year_col], df[other_mra_col], label='Other organ MRA', marker='x')

    # ---- FORMAT ----
    plt.xlabel('Year')
    plt.ylabel('Exams per 10,000 beneficiaries')
    plt.title('CTA and MRA exams for Medicare Beneficiaries')
    plt.legend()
    #plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), 'CTA_MRA_Trends.png'), dpi=150)
    plt.show()


def plot_stack_cta_mra(
    file_path,
    year_col='Year',
    class_cols=['Brain MRA', 'Brain CTA',],
    title='CTA and MRA exams for Medicare Beneficiaries',
    ylabel='Exams per 10,000 beneficiaries',
    figsize=(5, 5),
    save_name='CTA_MRA_Stacked.png'
):
    # ---- LOAD ----
    df = pd.read_excel(file_path, index_col=0).T
    df[year_col] = df.index.astype(int)

    # If not provided, take all columns except year
    if class_cols is None:
        class_cols = [c for c in df.columns if c != year_col]

    # ---- SORT BY YEAR (important) ----
    df = df.sort_values(year_col)

    # ---- PREPARE DATA ----
    x = df[year_col].values
    y = [df[col].values for col in class_cols]

    # ---- STYLE ----
    sns.set_theme(style="white")  # or "ticks", "darkgrid"

    # ---- PLOT ----
    plt.figure(figsize=figsize)

    plt.stackplot(
        x,
        *y,
        labels=class_cols,
        alpha=0.8
    )

    # ---- FORMAT ----
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left')

    plt.tight_layout()

    # ---- SAVE ----
    plt.savefig(os.path.join(os.path.dirname(file_path), save_name), dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_stack_cta_mra('/home/hvv/Documents/projects/K99/figures/MRA_CTA_Tables.xlsx')

    plot_cta_mra_trends('/home/hvv/Documents/projects/K99/figures/MRA_CTA_Tables.xlsx')

