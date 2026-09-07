"""Evaluate the five reference questions against the configured LLM provider."""
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from src.talk_to_data.nl_to_sql import ask, configured
from src.talk_to_data.prompt_templates import PATTERNS
from src.talk_to_data.query_runner import execute_query

ROOT = Path(__file__).resolve().parent


def main():
    load_dotenv(ROOT / '.env')
    if not configured():
        raise SystemExit('Set LLM_API_KEY and LLM_MODEL before running this evaluation.')
    db = ROOT / 'runtime/analytics.db'
    if not db.exists():
        raise SystemExit('Run bootstrap.py with the dataset first.')
    results = []
    for question, reference_sql in PATTERNS:
        expected = execute_query(db, reference_sql)
        try:
            actual = ask(question, [], db)
            results.append({'question': question, 'reference': expected,
                            'actual': actual, 'query_executed': actual.get('sql') is not None,
                            'manual_semantic_review': 'required: compare grouping, filters and values'})
            print(question, '->', 'SQL executed' if actual.get('sql') else 'clarification')
        except ValueError as exc:
            results.append({'question': question, 'error': str(exc), 'query_executed': False})
            print(question, '->', str(exc))
    report = {'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
              'live_calls_attempted': len(results), 'results': results,
              'note': 'Execution success alone does not prove semantic correctness. Review all comparisons.'}
    destination = ROOT / 'reports/live_chat_evaluation.json'
    destination.write_text(json.dumps(report, indent=2, default=str))
    print('Saved', destination)
    if not all(r['query_executed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
