"""Generate and normalize global features for new formula/mpid/gap rows."""
import argparse
import numpy as np
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.element import Stoichiometry, BandCenter, ElementFraction
from matminer.featurizers.composition.orbital import ValenceOrbital
from matminer.featurizers.composition.ion import IonProperty


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='CSV with formula, mpid, gap (eV)')
    p.add_argument('--reference', default='data/global-feat.csv.gz')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    df = pd.read_csv(a.input)
    if df.mpid.duplicated().any(): raise ValueError('Input must contain one row per mpid')
    df = StrToComposition().featurize_dataframe(df, 'formula')
    for f in (ElementProperty.from_preset('magpie'), Stoichiometry(), ValenceOrbital(), IonProperty(), BandCenter(), ElementFraction()):
        df = f.featurize_dataframe(df, 'composition', ignore_errors=False)
    df = df.drop(columns='composition').replace({True: 1, False: 0})
    ref = pd.read_csv(a.reference)
    missing = ref.columns.difference(df.columns)
    if len(missing): raise ValueError(f'Missing generated columns: {list(missing)}; check matminer version')
    df = df[ref.columns].copy()
    cols = ref.columns.drop(['formula', 'mpid'])
    low, high = ref[cols].min(), ref[cols].max()
    scale = (high - low).replace(0, 1)
    df[cols] = (df[cols] - low) / scale
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all(): raise ValueError('Non-finite descriptor values')
    df.to_csv(a.output, index=False)
    print(f'Wrote {len(df)} feature rows to {a.output}')

if __name__ == '__main__':
    main()
