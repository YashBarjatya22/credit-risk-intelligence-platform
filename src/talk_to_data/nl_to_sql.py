"""OpenAI-compatible Chat Completions client for natural-language-to-SQL queries."""
import json, os, threading, time
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse
from .prompt_templates import SYSTEM,VERSION
from .query_runner import execute_query

_lock=threading.Lock(); _calls=[]

def configured():return bool(os.getenv('LLM_API_KEY') and os.getenv('LLM_MODEL'))

def complete(messages):
    if not configured():raise ValueError('AI chat needs LLM_API_KEY and LLM_MODEL in the private environment configuration.')
    base=os.getenv('LLM_BASE_URL','https://api.openai.com/v1').rstrip('/')
    parsed=urlparse(base)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username:
        raise ValueError('The LLM endpoint must be a valid HTTPS URL.')
    with _lock:
        now=time.monotonic(); _calls[:]=[t for t in _calls if now-t<3600]
        if len(_calls)>=int(os.getenv('LLM_MAX_CALLS_PER_HOUR','60')):
            raise ValueError('The hourly AI request budget has been reached. Try again later.')
        _calls.append(now)
    body={'model':os.environ['LLM_MODEL'],'messages':messages,
          'response_format':{'type':'json_object'},'max_completion_tokens':700}
    req=Request(base+'/chat/completions',data=json.dumps(body).encode(),headers={
        'Authorization':'Bearer '+os.environ['LLM_API_KEY'],'Content-Type':'application/json'})
    payload=None
    for attempt in range(3):
        try:
            with urlopen(req,timeout=35) as res:payload=json.loads(res.read(100000))
            break
        except HTTPError as e:
            if e.code in {429,500,502,503,504} and attempt<2:
                time.sleep(2**attempt);continue
            raise ValueError(f'LLM provider returned HTTP {e.code}. Check model access, API billing and configuration.') from e
        except (URLError,TimeoutError) as e:
            if attempt<2:
                time.sleep(2**attempt);continue
            raise ValueError('The LLM service could not be reached. No response was returned.') from e
    try:
        choice=payload['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('Incomplete LLM response. Try a shorter question.')
        result=json.loads(choice['message']['content'])
        if not isinstance(result,dict):raise ValueError('LLM response was not a JSON object.')
        return result,payload.get('usage',{})
    except (KeyError,IndexError,TypeError,json.JSONDecodeError) as e:raise ValueError('LLM returned an invalid response; no query was executed.') from e

def grounded_answer(result):
    rows=result['rows']
    if not rows:return 'No records matched this question.'
    if len(rows)==1:
        def fmt(k,v):
            if v is None:return f'{k.replace("_"," ")}: unavailable'
            if isinstance(v,(float,int)):
                return f'{k.replace("_"," ")}: {v:.2%}' if 'rate' in k.lower() else f'{k.replace("_"," ")}: {v:,.2f}'
            return f'{k.replace("_"," ")}: {v}'
        return '; '.join(fmt(k,v) for k,v in rows[0].items())+'.'
    return f'The query returned {len(rows)} groups or records. The table shows the calculated results in SQL order. Rate columns are fractions (0.08 means 8%). Results are capped at 100 rows.'

def ask(question,history,db):
    if not isinstance(question,str) or not question.strip() or len(question)>1000:raise ValueError('Ask a question between 1 and 1,000 characters.')
    messages=[{'role':'system','content':SYSTEM}]
    for turn in history[-3:]:
        messages.extend([{'role':'user','content':turn['question'][:1000]},
                         {'role':'assistant','content':json.dumps({'sql':turn.get('sql'),'clarification':None})}])
    messages.append({'role':'user','content':question})
    output,usage=complete(messages)
    if output.get('sql') is None:
        return dict(question=question,sql=None,answer=str(output.get('clarification') or 'Please clarify the requested measure and population.'),rows=[],usage=usage,prompt_version=VERSION)
    result=execute_query(db,output['sql'])
    return dict(result,question=question,answer=grounded_answer(result),usage=usage,prompt_version=VERSION)
