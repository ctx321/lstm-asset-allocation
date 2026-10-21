"""Validate student API request/response examples and optionally POST a real fixture.

Uses Python standard library only.
"""
import argparse, json, math, sys
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("QQQ", "SPY", "IWM", "IEF", "TLT", "GLD", "SHY")
WEIGHTS = (*ASSETS, "CASH")
FIELDS = ("date", *ASSETS, "VIX", "IRX")

def iso_day(s):
    if not isinstance(s, str) or len(s) != 10 or date.fromisoformat(s).isoformat() != s:
        raise ValueError("Date must be ISO YYYY-MM-DD")
    return s

def num(v, positive=False, nonnegative=False):
    if type(v) not in (float, int) or not math.isfinite(v):
        raise ValueError("Expected finite JSON number")
    if positive and v <= 0: raise ValueError("Expected positive number")
    if nonnegative and v < 0: raise ValueError("Expected nonnegative number")

def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"as_of_date", "history"}:
        raise ValueError("Unexpected request keys")
    last = iso_day(request["as_of_date"])
    hist = request["history"]
    if not isinstance(hist, list) or len(hist) != 252:
        raise ValueError("history must contain exactly 252 rows")
    prev = None
    for i, row in enumerate(hist):
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError(f"Unexpected market row fields at index {i}")
        dt = iso_day(row["date"])
        if prev is not None and dt <= prev:
            raise ValueError("Dates must be strictly increasing")
        prev = dt
        for ticker in ASSETS: num(row[ticker], positive=True)
        num(row["VIX"], nonnegative=True)
        if row["IRX"] is not None: num(row["IRX"])
    if prev != last: raise ValueError("as_of_date must match last history date")
    return True

def validate_response(response):
    if not isinstance(response, dict) or set(response) != {"weights"}:
        raise ValueError("Response must contain only weights")
    w = response["weights"]
    if not isinstance(w, dict) or set(w) != set(WEIGHTS):
        raise ValueError("Response must provide exactly the eight required weights")
    for k in WEIGHTS:
        num(w[k], nonnegative=True)
        if w[k] > 1: raise ValueError(f"Weight {k} exceeds 1")
    if abs(math.fsum(w[k] for k in WEIGHTS) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to one within 1e-6")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="POST fixture to a running /predict endpoint")
    parser.add_argument("--request-file", default=str(ROOT / 'examples/sample_request_252.json'))
    parser.add_argument("--response-file", default=str(ROOT / 'examples/sample_response.json'))
    args=parser.parse_args()
    req_obj=json.loads(Path(args.request_file).read_text(encoding='utf-8'))
    validate_request(req_obj)
    print(f"PASS request fixture: 252 rows ending {req_obj['as_of_date']}")
    if args.url:
        body=json.dumps(req_obj).encode('utf-8')
        try:
            http_req=Request(args.url,data=body,headers={'Content-Type':'application/json'},method='POST')
            with urlopen(http_req, timeout=30) as response:
                body=response.read()
            resp_obj=json.loads(body.decode('utf-8'))
        except HTTPError as e:
            raise SystemExit(f"FAIL HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:400]}")
        except (URLError,TimeoutError) as e: raise SystemExit(f"FAIL connecting to API: {e}")
    else:
        resp_obj=json.loads(Path(args.response_file).read_text(encoding='utf-8'))
    validate_response(resp_obj)
    print('PASS portfolio response: 8 weights, finite, in [0,1], sum ~= 1')
    if args.url: print('PASS live endpoint compatible with the required contract')
if __name__ == '__main__':
    try: main()
    except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
        print(f'FAIL: {e}',file=sys.stderr)
        sys.exit(1)
