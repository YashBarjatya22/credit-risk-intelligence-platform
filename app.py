import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import hmac,json,time
from pathlib import Path
import numpy as np,pandas as pd,streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.data.preprocessor import RAW,CAT,features
from src.ml.predict import load_model,predict,explain
from src.talk_to_data.nl_to_sql import ask,configured,grounded_answer
from src.talk_to_data.query_runner import execute_query
from src.talk_to_data.prompt_templates import PATTERNS

ROOT=Path(__file__).resolve().parent;load_dotenv(ROOT/'.env')
st.set_page_config(page_title='CreditScope | Risk Intelligence',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp{background:#f5f8fb;color:#152738} h1,h2,h3{letter-spacing:-.025em}
[data-testid="stSidebar"]{background:#0b2235} [data-testid="stSidebar"] *{color:#eef6ff}
[data-testid="stMetric"]{background:white;padding:18px;border:1px solid #dbe5ed;border-radius:10px;border-top:3px solid #147d92}
.eyebrow{font-size:14px;letter-spacing:.16em;color:#147d92;font-weight:700;text-transform:uppercase}
.stButton>button[kind="primary"],.stFormSubmitButton>button{background:#087e8b;color:white;border:0}
.block-container{padding-top:2rem;padding-bottom:3rem} [data-testid="stCaptionContainer"]{font-size:14px}
</style>''',unsafe_allow_html=True)
password=os.getenv('APP_ACCESS_PASSWORD','')
if password and not st.session_state.get('authenticated'):
    st.title('CreditScope');st.write('Enter the access password provided with the demo.')
    entered=st.text_input('Access password',type='password')
    if st.button('Open workspace'):
        if hmac.compare_digest(entered,password):st.session_state.authenticated=True;st.rerun()
        else:st.error('Incorrect password.')
    st.stop()

@st.cache_resource
def bundle():return load_model(ROOT/'models/risk_model.joblib')
@st.cache_data
def report(name):return json.loads((ROOT/f'reports/{name}.json').read_text())

b=bundle();m=report('metrics');eda=report('eda')
with st.sidebar:
    st.title('◈ CreditScope');st.caption('CREDIT RISK INTELLIGENCE')
    page=st.radio('Workspace',['Overview','Explore data','Risk assessment','Model evidence','Decision rules','Talk to data'])
    st.divider();st.caption('Home Credit · candidate demonstration')
    st.caption('Yash Barjatya | NeoStats assignment')
    st.caption('AI chat configured' if configured() else 'AI chat needs configuration')
st.markdown('<div class="eyebrow">Credit intelligence / '+page+'</div>',unsafe_allow_html=True)
st.title(page)

if page=='Overview':
    st.write('Explore applicant patterns, estimate repayment difficulty and inspect the evidence behind each prediction.')
    a,c,d,e=st.columns(4)
    a.metric('Applicants',f"{eda['total_rows']:,}");c.metric('Development difficulty rate',f"{eda['target_rate']:.1%}")
    d.metric('Test ROC-AUC',f"{m['test']['roc_auc']:.3f}");e.metric('Test average precision',f"{m['test']['pr_auc_average_precision']:.3f}")
    l,r=st.columns([1.35,1])
    with l:
        st.subheader('Where the observed risk differs')
        g=pd.DataFrame(eda['groups']['NAME_INCOME_TYPE']).set_index('group')
        st.bar_chart(g[['rate']],color='#147d92',horizontal=True)
        st.caption('Development population; groups with at least 500 applicants. Rates are fractions.')
    with r:
        st.subheader('Read the result correctly')
        st.write('The score estimates the dataset’s payment-difficulty outcome. It is not a credit bureau score or an automatic lending decision.')
        st.write('Risk bands are relative segments set on validation data. SHAP explains the model’s calculation, not the cause of repayment difficulty.')
        st.info('Current scope: application data only. Detailed bureau and repayment transaction files have not been supplied.')
    st.subheader('A traceable workflow')
    st.write('Raw applications → deterministic features → trained model → probability calibration → risk bands and SHAP explanations.')
    st.caption('Natural-language questions follow a separate path through validated, read-only SQL. The LLM does not assign applicant risk scores.')

elif page=='Explore data':
    st.caption('Insights use development rows only, keeping the final holdout separate during model development.')
    chosen=st.selectbox('Compare applicant groups',list(eda['groups']),format_func=lambda s:s.replace('_',' ').title())
    g=pd.DataFrame(eda['groups'][chosen]);left,right=st.columns([1.3,1])
    with left:st.bar_chart(g.set_index('group')[['rate']],color='#147d92',horizontal=True)
    with right:st.dataframe(g.rename(columns={'group':'Group','count':'Applicants','positives':'Difficulty cases','rate':'Difficulty rate'}),hide_index=True,use_container_width=True)
    st.subheader('Six findings from the data')
    for i,insight in enumerate(eda['insights'],1):st.write(f'{i}. {insight}')
    with st.expander('Data quality and feature coverage'):
        st.write(f"{eda['columns']} source columns; {eda['numeric_columns']} numeric and {eda['categorical_columns']} categorical. Duplicate applicant IDs: {eda['duplicate_ids']}.")
        st.write(f"Employment sentinel 365243 occurs {eda['employment_sentinel_count']:,} times in development data; it becomes missing with an explicit indicator.")
        st.dataframe(pd.Series(eda['missing'],name='Missing fraction').to_frame())
        st.dataframe(pd.DataFrame(report('feature_catalog')).T,use_container_width=True)
    st.caption('Associations are not causal. Small groups are excluded from rate rankings; confounding and group fairness require further study.')

elif page=='Risk assessment':
    pfile=ROOT/'runtime/profiles.json'
    mode=st.radio('Input source',['Manual scenario']+(['Held-out applicant'] if pfile.exists() else []),horizontal=True)
    row={c:None for c in RAW}
    if mode=='Held-out applicant':
        profiles=json.loads(pfile.read_text());selection=st.selectbox('Applicant',[p['SK_ID_CURR'] for p in profiles])
        row=next(p.copy() for p in profiles if p['SK_ID_CURR']==selection)
        st.caption('A real held-out application. The observed outcome is not included in model inputs.')
    else:st.caption('An editable hypothetical scenario, not a real applicant. Unprovided fields use training preprocessing defaults.')
    with st.form('assessment_form'):
        c1,c2,c3=st.columns(3)
        def number(col,label,key,default):return col.number_input(label,min_value=0.01,value=float(row.get(key) or default))
        row['AMT_INCOME_TOTAL']=number(c1,'Annual income (dataset units)','AMT_INCOME_TOTAL',180000)
        row['AMT_CREDIT']=number(c2,'Credit amount (dataset units)','AMT_CREDIT',600000)
        row['AMT_ANNUITY']=number(c3,'Annuity (dataset units)','AMT_ANNUITY',30000)
        age=c1.number_input('Age in years',18,100,int(-row['DAYS_BIRTH']/365.25) if row.get('DAYS_BIRTH') else 35)
        row['DAYS_BIRTH']=-age*365.25
        income_types=list(b['categories']['NAME_INCOME_TYPE'])
        row['NAME_INCOME_TYPE']=c2.selectbox('Income type',income_types,index=income_types.index(row['NAME_INCOME_TYPE']) if row.get('NAME_INCOME_TYPE') in income_types else 0)
        opts=list(b['categories']['NAME_EDUCATION_TYPE'])
        row['NAME_EDUCATION_TYPE']=c3.selectbox('Education',opts,index=opts.index(row['NAME_EDUCATION_TYPE']) if row.get('NAME_EDUCATION_TYPE') in opts else 0)
        for col,key in zip([c1,c2,c3],['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
            value=row.get(key)
            row[key]=col.number_input(key.replace('_',' ').title()+' (optional)',min_value=0.0,max_value=1.0,value=float(value) if value is not None else None,step=.01)
        with st.expander('Remaining input fields'):
            st.caption('Preserved for held-out profiles; imputed in manual mode. A sparse scenario is less reliable.')
            st.json({k:v for k,v in row.items() if k!='SK_ID_CURR'})
        submitted=st.form_submit_button('Assess repayment risk',type='primary')
    if submitted:
        try:
            raw=pd.DataFrame([row]);result=predict(b,raw).iloc[0]
            with st.spinner('Calculating model explanation…'):details=explain(b,raw)
            st.session_state.assessment=(result.to_dict(),details,int(features(raw)[b['numeric']].isna().sum().sum()))
        except (ValueError,RuntimeError) as exc:st.error(str(exc))
    if 'assessment' in st.session_state:
        result,details,missing=st.session_state.assessment
        st.caption('Results from the last submitted assessment; submit again after editing.')
        c1,c2,c3,c4=st.columns(4);c1.metric('Estimated difficulty probability',f"{result['probability']:.1%}");c2.metric('Risk score / 100',f"{result['risk_score']:.1f}");c3.metric('Relative risk band',result['risk_band']);c4.metric('Missing numeric features',missing)
        if missing>10:st.warning('Many inputs are missing. Treat this as an incomplete scenario, not a validated applicant assessment.')
        vals=pd.Series(details['values'][0]*100,index=details['feature_names']).sort_values(key=abs,ascending=False).head(10).sort_values()
        fig,ax=plt.subplots(figsize=(9,4));ax.barh(vals.index,vals.values,color=['#d65b56' if v>0 else '#147d92' for v in vals.values]);ax.axvline(0,color='#bcc9d2');ax.set_xlabel('Contribution to probability (percentage points)');ax.set_title('Why the model gave this estimate');fig.tight_layout();st.pyplot(fig);plt.close(fig)
        positive=vals[vals>0].index.tolist();negative=vals[vals<0].index.tolist()
        st.write('Factors increasing the estimate: '+(', '.join(positive) or 'none among the displayed factors')+'.')
        st.write('Factors decreasing the estimate: '+(', '.join(negative) or 'none among the displayed factors')+'.')
        st.caption('Permutation SHAP explains the complete calibrated model relative to a training-summary reference. It is an approximation, not a causal effect. Correlated features can share attribution; masked combinations may be unrealistic.')
        st.download_button('Download assessment',json.dumps(result,indent=2,default=str),'assessment.json','application/json')

elif page=='Model evidence':
    st.write('Selection uses validation average precision. Calibration uses a separate split; final test metrics follow these choices.')
    st.dataframe(pd.DataFrame(m['split_counts'],index=['Applicants']),use_container_width=True)
    scores=[]
    for label,mm in [('Logistic baseline · validation',m['baseline_validation']),('Selected model · validation',m['validation']),('Selected model · final test',m['test'])]:
        scores.append({'Evaluation':label,**{k:round(mm[k],4) for k in ['roc_auc','pr_auc_average_precision','precision','recall','f1','brier','threshold']}})
    st.dataframe(pd.DataFrame(scores),hide_index=True,use_container_width=True)
    c1,c2=st.columns(2);c1.image(str(ROOT/'reports/figures/ROC.png'));c2.image(str(ROOT/'reports/figures/Precision_Recall.png'))
    c1,c2=st.columns(2)
    c1.dataframe(pd.DataFrame(m['test']['confusion_matrix'],index=['Actual no difficulty','Actual difficulty'],columns=['Screen negative','Screen positive']))
    c2.image(str(ROOT/'reports/figures/Calibration.png'))
    st.write('Threshold choice: '+m['threshold_rationale']);st.write('Band choice: '+m['bands_rationale'])
    st.warning('The threshold prioritizes recall and produces many false positives. A real credit policy needs explicit costs, external/time-based validation and fairness assessment.')
    st.caption('This application-only model does not use detailed bureau or repayment histories.')

elif page=='Decision rules':
    rules=report('rules');st.metric('Validation agreement with model bands',f"{rules['validation_agreement']:.1%}")
    st.write('A depth-three tree approximates the main model’s bands using five interpretable features. It is fitted on training predictions.')
    st.write('Examples in plain language (rounded thresholds):')
    st.write('• Mean external score at most 0.34 and age at most 56.35 years: the surrogate assigns High risk.')
    st.write('• Mean external score above 0.34 and at most 0.53: the surrogate assigns Medium risk.')
    st.write('• Mean external score above 0.53: the surrogate assigns Low risk.')
    st.caption('These examples summarize fitted branches. The exact saved tree and its imputation define the rule predictions.')
    st.code(rules['text'],language='text');st.info(rules['note'])
    st.write('EXT_SOURCE_MEAN combines external scores. CREDIT_INCOME_RATIO divides credit by annual income. ANNUITY_INCOME_RATIO divides annuity by annual income.')

elif page=='Talk to data':
    db=ROOT/'runtime/analytics.db';st.write('Ask about observed patterns. Inspect SQL and evidence with every answer.')
    if not db.exists():st.warning('Mount the dataset and run bootstrap.py before database questions can execute.')
    st.session_state.setdefault('history',[])
    if st.button('Clear conversation'):st.session_state.history=[];st.rerun()
    with st.expander('Five verified SQL examples (no AI call)'):
        example=st.selectbox('Example question',[p[0] for p in PATTERNS]);sql=dict(PATTERNS)[example];st.code(sql,language='sql')
        if st.button('Run example'):
            try:
                result=execute_query(db,sql);st.write(grounded_answer(result));st.dataframe(pd.DataFrame(result['rows']),hide_index=True)
            except ValueError as exc:st.error(str(exc))
    if not configured():st.info('Live AI chat is not configured. Add LLM_API_KEY and LLM_MODEL privately. Examples above are SQL demonstrations, not AI responses.')
    for turn in st.session_state.history:
        with st.chat_message('user'):st.write(turn['question'])
        with st.chat_message('assistant'):
            st.write(turn['answer'])
            if turn.get('sql'):st.code(turn['sql'],language='sql')
            if turn.get('rows'):st.dataframe(pd.DataFrame(turn['rows']),hide_index=True)
            if turn.get('usage'):st.caption('Provider-reported token usage: '+str(turn['usage']))
    question=st.chat_input('Which income types have higher difficulty rates?',disabled=not configured() or not db.exists(),max_chars=1000)
    if question:
        now=time.monotonic()
        if now-st.session_state.get('last_chat',0)<3:st.warning('Please wait a few seconds between questions.')
        else:
            st.session_state.last_chat=now
            try:
                with st.spinner('Generating and validating SQL…'):answer=ask(question,st.session_state.history,db)
                st.session_state.history.append(answer);st.session_state.history=st.session_state.history[-10:];st.rerun()
            except ValueError as exc:st.error(str(exc))
    st.caption('Only schema, question and three prior question/SQL pairs go to the provider. Results are summarized locally, without sending applicant rows or inventing numerical conclusions.')
st.divider();st.caption('Candidate demonstration · application data only · decision support, not loan approval.')
