#!/usr/bin/env python3
"""Rapid-fire triage demo: streams support tickets through a running `fern serve` and
prints one routed line per ticket as fast as the model answers (~30 ms each).

    uv run fern serve runs/fern            # in another shell
    uv run python scripts/demo.py [-n 300] [--url http://127.0.0.1:8000] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import httpx

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Payments, invoices, refunds, payouts, charges",
            "technical": "Bugs, errors, outages, API, integrations, login problems",
            "sales": "Pricing, plans, upgrades, enterprise, quotes",
            "abuse": "Spam, harassment, fraud, policy violations",
            "other": "Anything else, chit-chat, unclear",
        },
    },
    "severity": {"type": "score", "instructions": "How severe is the impact on the customer?", "criteria": ["Low", "Medium", "High", "Critical"]},
    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},
    "angry": {"type": "noul", "instructions": "Is the customer angry or frustrated?"},
}

TICKETS = [
    "Help! My payouts have been failing for 3 days.",
    "I was charged twice for last month's invoice. Please refund one.",
    "Login page shows a blank screen after the update. Chrome, macOS.",
    "Webhook deliveries stopped at 02:14 UTC, our orders are piling up!!!",
    "What's the difference between the Pro and Team plans?",
    "Can we get a quote for 500 seats with SSO?",
    "Someone is spamming my inbox using your platform, this is harassment.",
    "Thanks, everything works great now :)",
    "API returns 500 on POST /v2/orders since this morning. Production is down.",
    "Is there a student discount?",
    "I think my account was hacked, there are charges I didn't make.",
    "Where do I download invoices for tax purposes?",
    "Your app crashes when I upload a PDF larger than 10MB.",
    "Just saying hi, love the new dashboard.",
    "決済が3日間失敗し続けています。至急対応してください。",
    "先月の請求が二重になっています。返金をお願いします。",
    "ログイン後に画面が真っ白になります。昨日のアップデートからです。",
    "エンタープライズプランの見積もりをください。200席です。",
    "このユーザーから嫌がらせのメッセージが大量に届いています。",
    "新しいUI、すごく使いやすいです。ありがとう。",
    "支付连续失败三天了,我们的业务受到严重影响!",
    "上个月的发票重复扣款了,请退款。",
    "API 从今天早上开始返回 500 错误,生产环境已经停止。",
    "请问团队版和专业版有什么区别?",
    "Die Zahlung schlägt seit drei Tagen fehl, wir brauchen sofort Hilfe.",
    "Mir wurde die Rechnung doppelt berechnet.",
    "Nach dem Update ist die Anmeldeseite leer.",
    "Gibt es einen Rabatt für gemeinnützige Organisationen?",
    "El pago lleva tres días fallando. Es urgente.",
    "Me cobraron dos veces la factura del mes pasado.",
    "La API devuelve error 500 desde esta mañana, producción caída.",
    "¿Cuánto cuesta el plan Enterprise para 100 usuarios?",
    "Le paiement échoue depuis trois jours, c'est urgent.",
    "J'ai été facturé deux fois ce mois-ci.",
    "L'application plante quand j'importe un fichier CSV.",
    "Quelle est la différence entre les offres Pro et Team ?",
    "결제가 3일째 실패하고 있습니다. 긴급합니다.",
    "지난달 청구서가 이중으로 결제되었습니다.",
    "업데이트 이후 로그인 화면이 하얗게 나옵니다.",
    "엔터프라이즈 플랜 가격이 궁금합니다.",
]

TEAM_COLOR = {"billing": "33", "technical": "31", "sales": "32", "abuse": "35", "other": "90"}
SEV = ["░", "▒", "▓", "█"]


def color(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m"


def fit(s: str, width: int) -> str:
    """Truncate/pad to `width` terminal cells (CJK counts as 2)."""
    cells = [2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s]
    if sum(cells) <= width:
        return s + " " * (width - sum(cells))
    out, used = [], 0
    for c, w in zip(s, cells):
        if used + w > width - 1:
            break
        out.append(c)
        used += w
    return "".join(out) + "…" + " " * (width - used - 1)


def one_line(i: int, state: str, ans: dict, ms: float) -> str:
    team = ans["team"]["choice"]
    conf = ans["team"]["confidence"]
    sev = ans["severity"]["score"]
    bar = "".join(SEV[min(3, int(sev))] if k <= sev else "·" for k in range(4))
    urgent = color("1;31", "URGENT") if ans["urgent"]["noul"] > 0.5 else "      "
    angry = color("33", "😠") if ans["angry"]["noul"] > 0.5 else "  "
    text = fit(state, 44)
    return (
        f"{color('90', f'{i:04d}')} {color(TEAM_COLOR[team], f'{team:<9}')} {conf:4.2f} "
        f"{color('36', bar)} {sev:3.1f} {urgent} {angry} {text} {color('90', f'{ms:5.1f}ms')}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--key", default=os.environ.get("FERN_API_KEY", ""), help="Bearer key (or FERN_API_KEY)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    states = [rng.choice(TICKETS) for _ in range(args.n)]
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    client = httpx.Client(base_url=args.url, timeout=30, headers=headers)

    def run(i: int) -> tuple[int, str, dict, float]:
        t0 = time.perf_counter()
        r = client.post("/v1/systemone", json={"state": states[i], "questions": QUESTIONS})
        r.raise_for_status()
        return i, states[i], r.json()["answers"], (time.perf_counter() - t0) * 1000

    print(color("1", f"fern demo — {args.n} tickets × {len(QUESTIONS)} questions, concurrency {args.concurrency}\n"))
    t_start = time.perf_counter()
    lat: list[float] = []
    with ThreadPoolExecutor(args.concurrency) as pool:
        for i, state, ans, ms in pool.map(run, range(args.n)):
            print(one_line(i, state, ans, ms), flush=True)
            lat.append(ms)
    wall = time.perf_counter() - t_start
    lat.sort()
    print(
        color("1", f"\n{args.n} tickets, {args.n * len(QUESTIONS)} decisions in {wall:.2f}s → ")
        + color("1;32", f"{args.n / wall:.1f} tickets/s, {args.n * len(QUESTIONS) / wall:.0f} decisions/s")
        + color("90", f"   p50 {lat[len(lat) // 2]:.0f}ms  p90 {lat[int(len(lat) * .9)]:.0f}ms")
    )


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        sys.exit("server not reachable — start it with: uv run fern serve runs/fern")
