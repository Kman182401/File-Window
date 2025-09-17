IBKR API Settings — Paper Account Snapshot
=========================================

Source: user‑provided screenshot of API Settings (Paper) and live connectivity checks. This snapshot is treated as stable until we change it here or you explicitly tell me it changed.

Captured on: 2025‑09‑17 (repo local time)

Assumptions
- These values are considered “fixed” for automation and docs until updated through this process.
- Items marked “needs confirm” were present in the UI but not perfectly legible in the screenshot; I listed them for completeness and we can flip them to confirmed on request.

Confirmed Settings (from screenshot and live checks)
- Socket port: 4002 (Paper)
- Logging level: Error
- Encode API messages, instrument names: ASCII 7 (Python, Java, …)
- Timeout to send bulk data to API: 30 seconds
- Split historical data into parts (MB): 16
- Allow connections from localhost only: Enabled
- Trusted IPs: 127.0.0.1
- Paper connectivity verified: connected: True, server version observed via ib_insync (latest check in this repo).

Likely/Visible Options (needs confirm)
- Read‑Only API: expected OFF for paper order testing (UI toggle present)
- Download open orders on connection: toggle present
- Include virtual FX positions when sending portfolio: toggle present
- Prepare DailyPNL when downloading positions: toggle present
- Send status updates for Volatility orders with “Continuous Update” flag: toggle present
- Use negative numbers to bind automatic orders: toggle present
- Create API message log file: toggle present
- Include market data in API log file: toggle present
- Expose entire trading schedule to API: toggle present
- Split Insured Deposit from Cash Balance: toggle present
- Send zero positions for today’s opening positions only: toggle present
- Use Account Groups with Allocation Methods: toggle present
- Master API client ID: field present (value not captured)
- Component Exch. Separator: field present (value not captured)
- Show Forex data in 1/10 pips / Allow Forex trading in 1/10 pips: toggles present
- Round Account values to nearest whole number: toggle present
- Send market data in lots for US stocks for dual‑mode API clients: toggle present
- Show advanced order reject in UI always: toggle present
- Reject messages above maximum allowed message rate vs applying pacing: toggle present
- Maintain connection upon receiving incorrectly formatted fields: toggle present
- Compatibility Mode options (e.g., ISLAND for US stocks on NASDAQ): selector present
- Send instrument‑specific attributes for dual‑mode API client in: selector (Instrument timezone selected)
- Send Forex market data in compatibility mode in integer units: toggle present
- Automatically report Netting Event Contract trades: toggle present
- Option exercise requests are: selector (shows “editable until cutoff time (varies by clearing house)”) 
- Reset API order ID sequence: button present

Operational Notes
- Our code and docs assume Paper API on 127.0.0.1:4002 with clientId defaults 9001 (smokes) and 9002 (pipeline). See `~/.config/m5_trader/env.local`.
- For placing paper orders, Read‑Only API should be OFF.
- If you want me to promote any “needs confirm” items to confirmed, say the word and I’ll record them verbatim.

File Ownership
- This snapshot is maintained by the migration process. Do not edit manually; request updates here so we keep provenance.
