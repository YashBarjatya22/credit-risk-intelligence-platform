"""Defense in depth: AST allowlist + SQLite read-only URI + authorizer + limits."""
import sqlite3, time
from pathlib import Path
import sqlglot
from sqlglot import exp

COLUMNS={'applicant_id','payment_difficulty','income','credit','annuity','age','income_type',
 'education','housing','contract_type','occupation','credit_income_ratio','split'}
FUNCTIONS={'count','avg','sum','min','max','round','coalesce','nullif','abs','cast','floor','ceil','ceiling'}

def validate_sql(sql):
    if not isinstance(sql,str) or not sql.strip() or len(sql)>6000:
        raise ValueError('Provide a single short SELECT query.')
    try: statements=sqlglot.parse(sql,read='sqlite')
    except sqlglot.errors.ParseError as e: raise ValueError('SQL could not be parsed.') from e
    if len(statements)!=1 or not isinstance(statements[0],exp.Select):
        raise ValueError('Only one SELECT query is allowed.')
    tree=statements[0]
    if len(list(tree.find_all(exp.Select)))!=1 or tree.args.get('with') or tree.args.get('with_') or tree.find(exp.Join):
        raise ValueError('Use one table without joins, CTEs or nested SELECT queries.')
    tables=list(tree.find_all(exp.Table))
    if len(tables)!=1 or tables[0].name.lower()!='applicants' or tables[0].db or tables[0].catalog:
        raise ValueError('Only the applicants table is available.')
    aliases={a.alias.lower() for a in tree.find_all(exp.Alias)}
    for col in tree.find_all(exp.Column):
        if col.name.lower() not in COLUMNS|aliases:
            raise ValueError('Unknown or unavailable column: '+col.name)
    for star in tree.find_all(exp.Star):
        if not isinstance(star.parent,exp.Count):raise ValueError('List named columns instead of SELECT *.')
    for fn in tree.find_all(exp.Func):
        if isinstance(fn,(exp.Case,exp.If,exp.And,exp.Or,exp.Not)):continue
        name=(fn.name if isinstance(fn,exp.Anonymous) else fn.sql_name()).lower()
        if name not in FUNCTIONS:raise ValueError('Function not allowed: '+name)
    limit=100
    if tree.args.get('limit'):
        v=tree.args['limit'].expression
        if not isinstance(v,exp.Literal) or not v.is_int or int(v.this)<1:raise ValueError('LIMIT must be a positive integer.')
        limit=min(100,int(v.this))
    if tree.args.get('offset'):raise ValueError('OFFSET is not available.')
    tree.set('limit',None)
    return tree.limit(limit).sql(dialect='sqlite')

def execute_query(path,sql,timeout=2.0):
    safe=validate_sql(sql)
    target=Path(path).resolve()
    if not target.is_file():raise ValueError('Analytics database is not ready. Run bootstrap.py first.')
    con=sqlite3.connect(target.as_uri()+'?mode=ro',uri=True,timeout=timeout)
    deadline=time.monotonic()+timeout
    try:
        con.execute('PRAGMA query_only=ON'); con.enable_load_extension(False)
        def authorize(action,a,b,db,trigger):
            if action==sqlite3.SQLITE_SELECT:return sqlite3.SQLITE_OK
            if action==sqlite3.SQLITE_READ and a=='applicants' and (not b or b in COLUMNS):return sqlite3.SQLITE_OK
            if action==sqlite3.SQLITE_FUNCTION and str(b or a).lower() in FUNCTIONS:return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        con.set_authorizer(authorize)
        con.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        cursor=con.execute(safe); names=[c[0] for c in cursor.description]
        rows=[dict(zip(names,r)) for r in cursor.fetchmany(100)]
        return {'sql':safe,'rows':rows,'row_count':len(rows),'limit':100}
    except sqlite3.Error as e:raise ValueError('Query failed validation, exceeded its time limit, or could not execute.') from e
    finally:con.close()
