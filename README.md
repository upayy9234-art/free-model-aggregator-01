# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/upayy9234-art/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/upayy9234-art/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One server, pure Python 3.13 standard library, serving OpenAI-compatible
(`POST /v1/chat/completions`, `GET /v1/models`) and Anthropic-compatible
(`POST /v1/messages`) APIs on top of ten simulated free-tier model providers.

## How it works

* **Catalog** — the server probes each provider's own catalog route in its
  wire (OpenAI `GET /v1/models`, Anthropic `GET /v1/models` with
  `x-api-key`, Gemini `GET /v1beta/models`) and registers exactly that
  catalog (`python3 main.py --list` prints the same document).
* **Routing** — every request is planned against the providers' per-trial
  budgets, so the best free model that meets the request (tools, vision,
  JSON mode, minimum context) is used, and the top models are not burned on
  the first requests. Weight-2 requests are planned first.
* **Failover** — a refused call (402/429/5xx) is retried on the same provider
  (a scheduled refusal is transient by call index), then on the next-best
  eligible model. A provider that fails twice running is deprioritised;
  three times it is treated as dead. Exactly one successful provider call
  per request.
* **Wire translation** — the client's body is passed through untouched in
  the provider's wire (system, user text, images, tools, stop sequences,
  max_tokens, JSON mode); the provider's answer is returned in the client's
  wire, streamed or not as asked, with usage and stop reason mapped. An
  OpenAI request in JSON mode is never routed to an Anthropic-wire provider
  (that wire cannot echo JSON mode).

## Practice scores

| trial | score | quality | served | wire | failover | catalog |
|------:|------:|--------:|-------:|-----:|---------:|--------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 |

(Quality 99 is the practical ceiling: the single best model, 1475, lives on
one provider whose per-trial budget caps at 12 calls, so the last requests
must use the 1470 model.)

## Usage

```sh
# inside a trial (the harness sets these):
MAKEFUN_ENV_URL=http://127.0.0.1:<port> \
MAKEFUN_PROMPT_FILE=<trial>/input/prompt.txt \
python3 main.py

# print the catalog the server would register:
python3 main.py --list
```

No third-party packages: standard library only (`http.server`, `http.client`,
`json`, `threading`). No model calls of its own; every answer comes from one
of the ten providers in the brief.

## License

MIT — see [LICENSE](LICENSE).
