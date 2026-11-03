import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from demo.constants import COLORS

def finish(fig,height=340):
    fig.update_layout(height=height,margin=dict(l=10,r=10,t=25,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='DM Sans, sans-serif'),legend_title_text='',legend=dict(orientation='h',y=1.17,x=0),hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False,zeroline=False,automargin=True)
    fig.update_yaxes(gridcolor='rgba(148,163,184,.14)',zerolinecolor='rgba(148,163,184,.3)',automargin=True)
    return fig

def bars(rows,methods,title_metric=None):
    df=pd.DataFrame([r for r in rows if r['method'] in methods]).copy()
    df['Evidence budget']=df['budget'].astype(int).astype(str)
    fig=px.bar(df,x='Evidence budget',y='quality',color='method',barmode='group',color_discrete_map=COLORS,text=df['quality'].map(lambda x:f'{x:.3f}'),hover_data=['n_queries'])
    fig.update_traces(textposition='outside',cliponaxis=False)
    fig.update_yaxes(title=title_metric or 'Task score',range=[0,min(1.,max(df.quality)*1.22)])
    return finish(fig)

def deltas(rows,label='Paired difference'):
    fig=go.Figure()
    for r in rows:
        point=r['point'];lo=r['ci_low'];hi=r['ci_high']
        y=f"{r['dataset']} · {int(r['budget'])}"
        fig.add_trace(go.Scatter(x=[point],y=[y],mode='markers',showlegend=False,marker=dict(size=10,color='#22c3ad'),error_x=dict(type='data',array=[max(0,hi-point)],arrayminus=[max(0,point-lo)],color='#22c3ad'),hovertemplate=y+'<br>Difference %{x:.5f}<extra></extra>'))
    fig.add_vline(x=0,line_dash='dot',line_color='#94a3b8');fig.update_xaxes(title=label)
    return finish(fig,max(260,55*len(rows)))

def attribution(units):
    df=pd.DataFrame(units)
    fig=px.scatter(df,x='relevance',y='importance',color='category',color_discrete_map=COLORS,custom_data=['unit_id'],hover_data=['granularity','tokens'],labels={'relevance':'Semantic / retrieval relevance','importance':'Leave-one-out downstream utility'})
    fig.update_traces(marker=dict(size=10,line=dict(width=1,color='#17263a')))
    fig.add_hline(y=0,line_color='#94a3b8',line_dash='dot')
    return finish(fig,330)
