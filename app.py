from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dashboard_data'
CRIMES = ['murder','attempted_murder','rape','assault_gbh','kidnapping']
st.set_page_config(page_title='Ekurhuleni Safety Analytics', layout='wide')
st.title('Ekurhuleni | Recorded Violent-Crime Hotspot Analytics')
st.caption('Historical police-station-level indicators — not a measurement of individual or female-specific victimisation risk.')

@st.cache_data
def read_csv(name):
    p = DATA/name
    return pd.read_csv(p) if p.exists() else None

raw = read_csv('cleaned_crime.csv')
if raw is None:
    st.error('Missing dashboard_data/cleaned_crime.csv. Export this file from Stage 5 and include it in the project folder.')
    st.stop()

required=['year','station','longitude','latitude']+CRIMES
missing=[c for c in required if c not in raw.columns]
if missing:
    st.error(f'Cleaned data missing required columns: {missing}')
    st.stop()

raw['year']=raw['year'].astype(str)
raw['station']=raw['station'].astype(str)
for c in CRIMES+['longitude','latitude']:
    raw[c]=pd.to_numeric(raw[c].astype(str).str.replace(',','.',regex=False),errors='coerce')
raw['selected_violent_total']=raw[CRIMES].sum(axis=1,min_count=len(CRIMES))

with st.sidebar:
    st.header('Filters')
    years=sorted(raw['year'].unique().tolist())
    selected_years=st.multiselect('Financial years', years, default=years[-3:])
    stations=sorted(raw['station'].unique().tolist())
    selected_stations=st.multiselect('Police-station areas', stations, default=stations)
    category=st.selectbox('Recorded crime category', CRIMES+['selected_violent_total'], index=len(CRIMES))

if not selected_years or not selected_stations:
    st.warning('Select at least one year and station.')
    st.stop()
filtered=raw[raw['year'].isin(selected_years)&raw['station'].isin(selected_stations)].copy()
if filtered.empty:
    st.warning('No records match these filters.')
    st.stop()

# Reuse fixed year-specific quartile labels based on all stations, not the current filtered selection.
raw['threshold']=raw.groupby('year')['selected_violent_total'].transform(lambda x:x.quantile(.75))
raw['historical_hotspot']=(raw['selected_violent_total']>=raw['threshold']).astype(int)
filtered=filtered.merge(raw[['year','station','threshold','historical_hotspot']],on=['year','station'],how='left',validate='one_to_one')

st.info('Hotspots are comparative, based on the top 25% of station-level selected crime counts in each year. Population-adjusted risk is not available. Coordinates represent police stations, not incident sites.')
metric_cols=st.columns(4)
metric_cols[0].metric('Station-year records',f'{len(filtered):,}')
metric_cols[1].metric('Stations',filtered['station'].nunique())
metric_cols[2].metric('Selected recorded incidents',f'{filtered[category].sum():,.0f}')
metric_cols[3].metric('Historical hotspot records',int(filtered['historical_hotspot'].sum()))

tabs=st.tabs(['Overview & EDA','Geography','Model evaluation','Three hotspots & patterns','Census context','Data & limitations'])
with tabs[0]:
    a,b=st.columns(2)
    with a:
        st.subheader('Annual trends')
        trends=filtered.groupby('year',as_index=False)[category].sum().sort_values('year')
        st.line_chart(trends.set_index('year')[category])
    with b:
        st.subheader('Station comparison')
        by_station=filtered.groupby('station')[category].sum().sort_values(ascending=False).head(12)
        st.bar_chart(by_station)
    a,b=st.columns(2)
    with a:
        st.subheader('Count distribution')
        fig,ax=plt.subplots(figsize=(6,4)); ax.hist(filtered[category].dropna(),bins=20)
        ax.set(xlabel='Recorded incidents per station-year',ylabel='Frequency'); st.pyplot(fig); plt.close(fig)
    with b:
        st.subheader('Crime category correlations')
        corr=filtered[CRIMES].corr(); fig,ax=plt.subplots(figsize=(6,4))
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(len(CRIMES)),labels=CRIMES,rotation=45,ha='right')
        ax.set_yticks(range(len(CRIMES)),labels=CRIMES)
        fig.colorbar(im,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader('Murder vs assault GBH')
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(filtered['assault_gbh'],filtered['murder'],alpha=.55)
    ax.set(xlabel='Assault GBH count',ylabel='Murder count');st.pyplot(fig);plt.close(fig)

with tabs[1]:
    st.subheader('Police-station locations and recorded count')
    loc=filtered.groupby('station',as_index=False).agg(latitude=('latitude','first'),longitude=('longitude','first'),count=(category,'sum'),historical_hotspot=('historical_hotspot','max'))
    loc=loc.dropna(subset=['latitude','longitude'])
    st.map(loc,latitude='latitude',longitude='longitude',size='count')
    st.dataframe(loc.sort_values('count',ascending=False),use_container_width=True,hide_index=True)
    st.caption('Point positions are station coordinates and do not depict station-area boundaries or locations of individual offences.')

with tabs[2]:
    metrics=read_csv('model_evaluation.csv')
    cm=read_csv('confusion_matrix.csv')
    preds=read_csv('test_predictions.csv')
    validation=read_csv('validation_results.csv')
    if metrics is None or cm is None or preds is None:
        st.warning('Stage 8 exports required: model_evaluation.csv, confusion_matrix.csv, and test_predictions.csv in dashboard_data.')
    else:
        st.subheader('Held-out test evaluation')
        record=metrics.iloc[0]
        cols=st.columns(4)
        for col,name in zip(cols,['Accuracy','Precision','Recall','F1']):
            col.metric(name,f'{float(record[name]):.1%}')
        st.caption(f"Selected model: {record['Model']}")
        st.subheader('Confusion matrix (actual rows × predicted columns)')
        matrix=cm.iloc[:,-2:].apply(pd.to_numeric,errors='coerce').to_numpy()
        fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(matrix,cmap='Blues')
        ax.set_xticks([0,1],labels=['Non-hotspot','Hotspot']);ax.set_yticks([0,1],labels=['Non-hotspot','Hotspot'])
        ax.set(xlabel='Predicted',ylabel='Actual')
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{int(matrix[i,j])}',ha='center',va='center')
        st.pyplot(fig);plt.close(fig)
        st.subheader('Unseen test predictions')
        st.dataframe(preds,use_container_width=True,hide_index=True)
    if validation is not None:
        st.subheader('Validation-model comparison')
        st.dataframe(validation,use_container_width=True,hide_index=True)
    st.caption('The evaluation tab displays exported, already-computed results. Geographic/category sidebar filters do not recalculate the held-out test metrics.')

with tabs[3]:
    latest=raw['year'].max()
    latest_df=raw[raw['year']==latest].sort_values('selected_violent_total',ascending=False)
    tops=latest_df[latest_df['historical_hotspot']==1].head(3)
    names=tops['station'].tolist()
    st.subheader(f'Three high-count hotspots in {latest}')
    st.dataframe(tops[['station','selected_violent_total','threshold','historical_hotspot']],hide_index=True,use_container_width=True)
    if names:
        profile=raw[raw['station'].isin(names)].pivot(index='year',columns='station',values='selected_violent_total').sort_index()
        st.line_chart(profile)
        st.subheader('Selected crime categories in latest year')
        st.bar_chart(tops.set_index('station')[CRIMES])
    clusters=read_csv('station_clusters.csv')
    if clusters is not None and 'Cluster' in clusters.columns:
        st.subheader('Unsupervised pattern groups (K-Means)')
        st.dataframe(clusters[['station','Cluster']+[c for c in ['mean_violent_total'] if c in clusters.columns]],hide_index=True,use_container_width=True)
        st.caption('Cluster IDs describe similarities in counts; they are not danger ratings.')

with tabs[4]:
    st.subheader('Stats SA Census 2011 and 2022 — Ekurhuleni')
    census=read_csv('census_ekurhuleni.csv')
    if census is None:
        st.warning('Census data file not found.')
    else:
        st.dataframe(census,use_container_width=True,hide_index=True)
        st.subheader('Municipality population by census year')
        st.bar_chart(census.set_index('Year')['Population'])
        st.subheader('Female-headed households (%)')
        st.bar_chart(census.set_index('Year')['Female_Headed_Households_Percent'])
        if 2011 in census['Year'].values and 2022 in census['Year'].values:
            p1=float(census.loc[census['Year']==2011,'Population'].iloc[0])
            p2=float(census.loc[census['Year']==2022,'Population'].iloc[0])
            st.metric('Population change 2011–2022', f'{(p2/p1-1):.1%}')
        st.info('Census figures describe Ekurhuleni municipality, not individual police stations. They are contextual indicators, not station-level crime-rate denominators. Correlation does not establish causation.')

with tabs[5]:
    st.subheader('Filtered source records')
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    st.download_button('Download filtered CSV',filtered.to_csv(index=False),file_name='ekurhuleni_filtered.csv',mime='text/csv')
    st.markdown('''**Sources:** SAPS Annual Crime Records 2005–2026 (UCT DataFirst), https://datafirst.uct.ac.za/dataportal/index.php/catalog/1012/study-description; Stats SA Census 2022 Municipal Fact Sheet, https://census.statssa.gov.za/assets/documents/2022/Census_2022_Municipal_factsheet-Web.pdf.\n\n**Geographic alignment:** The Census is municipality-level contextual evidence, not a denominator for individual police stations. We do not perform a station-level Census join.\n\n**Limitations:** General SAPS categories do not consistently identify victim sex; recorded crime is affected by underreporting and differences in reporting. Raw totals are not per-person risk rates. Annual data cannot establish times of day. Historical associations and model labels do not prove causation, characterize residents, or connect recent cases.''')
