IB Gateway Ops — Tools, Safety Rails, and Usage

Context: This repo now includes first-class helpers for Interactive Brokers Gateway (paper). They live under tools/ and auto-use the project venv (.venv). These are intended for diagnostics, safe order validation, and quick snapshots that Codex can run and reason about.

Paths & Assumptions

    Tools (tracked):

        tools/ibg_dump.py + wrapper tools/ibg_dump

        tools/ibg_order_smoke.py + wrapper tools/ibg_order_smoke

    Python env: project venv at ./.venv (wrappers call .venv/bin/python).

    IB Gateway (paper) is expected on the same machine with:

        IBKR_HOST=127.0.0.1

        IBKR_PORT=4002

        IBKR_CLIENT_ID=9001 (adjust if needed)

Safety Rails (VERY IMPORTANT)

    Default behavior is safe: do not transmit real orders unless explicitly permitted.

    An order will be actually sent to paper only if all are true:

        No --whatif flag on the command, and

        ALLOW_ORDERS=1 in the environment, and

        IBKR_PAPER=1 in the environment

    You can add --auto-cancel to immediately cancel a transmitted paper order (keeps smoke tests tidy).

Operating Modes (quick map)
Mode	How to invoke	What happens
Simulation (what-if)	add --whatif	IBKR returns margin/feasibility only; no order is sent
Blocked transmit	(no --whatif) but missing either ALLOW_ORDERS=1 or IBKR_PAPER=1	Tool forces simulation anyway
Paper transmit	no --whatif and ALLOW_ORDERS=1 and IBKR_PAPER=1	Sends order to paper; optional --auto-cancel

Commands Codex Can Run (copy/paste)

Health & snapshots (JSON for analysis/diffing):

    Accounts:
    tools/ibg_dump accounts

    Positions:
    tools/ibg_dump positions

    Open orders:
    tools/ibg_dump orders

    Account values (full):
    tools/ibg_dump account --all

    Executions since UTC midnight:
    tools/ibg_dump executions --since "$(date -u +%Y-%m-%dT00:00:00Z)"

Order smoke — safe by default (what-if):

    Stock example (AAPL limit far from market):
    tools/ibg_order_smoke --asset STK --symbol AAPL --order LMT --limit 0.01 --whatif

    Futures example (ES front month, limit placeholder):
    tools/ibg_order_smoke --asset FUT --symbol ES --resolve-front-month --order LMT --limit 1.00 --whatif

Paper transmit (only when explicitly intended):

export ALLOW_ORDERS=1
export IBKR_PAPER=1
tools/ibg_order_smoke --asset STK --symbol AAPL --order LMT --limit 0.01 --auto-cancel

    Omit --whatif on purpose to transmit; --auto-cancel will cancel immediately after submission.

When Codex Should Use These

    Before strategy runs or debugging: run snapshots (positions/orders) and summarize deltas vs prior output.

    When validating order logic/contract resolution/margin impact: run order smoke in --whatif.

    When verifying end-to-end paper connectivity: run order smoke with env guards + --auto-cancel, then confirm via tools/ibg_dump orders and/or positions.

Contract Resolution Notes

    FUT with --resolve-front-month auto-qualifies the nearest non-expired month on CME (e.g., ES on CME).

    You can override with --expiry YYYYMM if needed.

Troubleshooting (Codex checklist)

    Timeout or connected:false: confirm IBG is logged in, 127.0.0.1 is in Trusted IPs, port is 4002 (paper), and try again.

    Order rejected in simulation: examine returned orderState fields (initMarginChange, maintMarginChange, etc.); adjust size, price, or contract.

    No .venv: create it and install ib_insync:

        python3 -m venv .venv && . .venv/bin/activate && pip install -U pip wheel setuptools ib-insync

Guardrails & Etiquette

    Do not run a real transmit unless the user explicitly asks for it and confirms paper mode.

    Prefer --whatif first, then paper transmit with --auto-cancel for smoke.

    Never store credentials or secrets in the repo.

    If parsing/transforming JSON for reports, keep raw outputs available for auditability.

Acceptance Criteria (for Codex self-check)

    Running tools/ibg_dump positions returns valid JSON with zero exceptions.

    Running tools/ibg_order_smoke … --whatif returns JSON including whatIf: true and margin fields (no transmit).

    With ALLOW_ORDERS=1 + IBKR_PAPER=1 and no --whatif, response includes an orderId, and (if --auto-cancel) the order is promptly canceled.

This section tells Codex what the new tools are, how and when to use them, and the safety policy it must follow.

