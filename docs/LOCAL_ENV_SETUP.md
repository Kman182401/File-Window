Local Environment Setup
=======================

Overview
- IB Gateway (Paper) on `127.0.0.1:4002`, Read-Only API OFF.
- Local config lives at `~/.config/m5_trader/env.local`.
- Project venv at `./.venv`.

Steps
- Create local config:
  - `mkdir -p ~/.config/m5_trader`
  - `cp config/env.example ~/.config/m5_trader/env.local`
  - Edit values as needed.

- Activate venv and install deps:
  - `python3 -m venv .venv && . .venv/bin/activate`
  - `pip install -r requirements.txt`

- Launch IB Gateway (Paper):
  - `~/Jts/ibgateway/1040/ibgateway &`
  - In UI: Enable ActiveX/Socket, Read-Only API OFF, Port 4002, Trusted IPs: 127.0.0.1.

- Quick connectivity check:
  - `python -c "from ib_insync import IB; ib=IB(); ib.connect('127.0.0.1',4002,clientId=9001,timeout=7); print(ib.isConnected()); ib.disconnect()"`

Notes
- Do not commit `~/.config/m5_trader/env.local` or any secrets.
- Logs/data are stored outside the repo (see env vars).

