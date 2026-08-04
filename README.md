# chat-flashback

Analyzes a Facebook Messenger export and generates reports about a group chat:
yearly recaps, member personalities, reaction dynamics, response-speed leaderboards,
swear-word stats, custom term tracking, conversation starters, reply chains, ghosting
stats, sentiment, word clouds, and a weirdest statements section. Also ships a local
web UI to read the chat like a messaging app.

Runs locally. Your data is not uploaded anywhere.

## Features

- Yearly recaps. Top member, top word, record day per year.
- Member personalities. Signature words, emojis, peak posting hour, night-owl percentage.
- Reaction dynamics. Most-reacted messages, reactor rankings.
- Response-speed leaderboard. Median time to reply per member, ghosted percentage.
- Swear-word analytics. Per-member counts and signature swear words.
- Custom term tracking. Count and chart any words or phrases with `--track` or `--track-file`.
- Conversation starters. Sessions split on 30-minute gaps, longest single back-and-forth.
- Reply chains. Longest reply chains reconstructed from `reply_to_message_id`.
- Ghosting stats. Longest silence between each member's own messages.
- Activity heatmap. GitHub-style calendar grid plus pace trends (messages/day, calls, media).
- Pair dynamics. Heatmaps of who replies to whom and who reacts to whose messages.
- Hourly radar profiles. Each member's 24-hour activity shape.
- Word clouds. Overall and per-member, generated with `wordcloud`.
- Monologues and unsent messages. Longest solo runs ("could've been an email") and `is_unsent`.
- Message-length and word trends over time.
- Sentiment (VADER). Average mood per member and per year.
- Weirdest statements. All-caps, 3am, punctuation-spiral, and extreme-length messages.
- Self-contained `report.html` with all charts embedded (share one file). Sticky nav,
  dark/light toggle, and filterable, sortable tables.
- Local chat reader (`--serve`). Browse the chat in a Messenger-style web UI.
- `--anonymize`. Replaces names with Person A, Person B in all output.
- Supports group and 1-on-1 chats. Detects multiple threads in one export.

## Requirements

Python 3.8+.

```bash
pip install -r requirements.txt
```

Optional extras that the tool skips gracefully if missing:
- `vaderSentiment` powers sentiment (English-only, may be noisy on mixed-language chats).
- `wordcloud` powers the word clouds.

## Usage

```bash
python analyze_chat.py --input data --output output
```

Options:

| Flag | Description |
|---|---|
| `--input, -i` | Path to a thread folder or an export `messages/` folder (default: `data/`) |
| `--output, -o` | Output folder for charts and `summary.md` (default: `output/`) |
| `--anonymize` | Replace member names with Person A, Person B in all output |
| `--track` | Comma-separated words or phrases to count, e.g. `--track "lol, bro"` |
| `--track-file` | File with tracked terms, one per line (`#` comments and blank lines ignored) |
| `--year` | Analyze only one year, e.g. `--year 2017` |
| `--top` | Number of entries in leaderboards (default: 10) |
| `--json` | Also write `summary.json` with the report data as structured JSON |
| `--serve` | Start the local chat reader web UI instead of writing reports |
| `--port` | Port for `--serve` (default: 8080) |

Writes `summary.md`, `report.html`, and PNG charts into `output/<thread>/`.

## Chat reader

```bash
python analyze_chat.py --input <thread> --serve
```

Starts a local server on `127.0.0.1:8080` (localhost only, no authentication) and
opens a Messenger-style reader:

- Infinite-scroll feed grouped by day, newest or oldest first
- Sender color chips, inline reactions, media thumbnails, shares and call messages
- Reply threading, "N years ago" badges, subtle sentiment tint
- Debounced full-text search and per-member filters
- A jump-to-date control and a link to the full report

Media files are served from the export folder only; paths are resolved inside the
thread directory so nothing outside it can be read. Search is a simple substring
scan over the messages, so it is fast enough for everyday use on large chats.

## Getting your Messenger data

1. Facebook Settings > Your information > Download your information.
2. Select Messages and the chats you want.
3. Set format to JSON and media quality to Low.
4. Download and extract. Point `--input` at:
   `youraccount_.../your_activity_across_facebook/messages/inbox/<thread>/`

A thread folder contains `message_1.json`, `message_2.json`, ... that you can point
`--input` at directly.

## Privacy

- All processing happens locally.
- `--anonymize` replaces real names in every chart and report, including names that
  appear inside quoted message text.
- `--serve` binds to `127.0.0.1` only, so the reader is never reachable from the network.
- `.gitignore` excludes `data/` and `output/` so an export cannot be committed.

## Sample data

`sample_data/` contains a small synthetic thread used for testing:

```bash
python analyze_chat.py --input sample_data --track "shawarma, bro" --json
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Examples

Sample output from the synthetic thread is in `examples/`:

- `example_summary.md` - a full generated report
- `report.html` - the same report as a single self-contained HTML file
- `summary.json` - the report data as structured JSON
- PNG charts: messages per year, activity heatmap, pace trends, activity by
  hour/weekday, top members, top words, word clouds, yearly recap, pair dynamics,
  hourly radar, reaction dynamics, response speed, swear stats, tracked terms,
  domains, media, reply chains, ghosting, monologues, conversation starters,
  monthly timeline, word trends, and sentiment

![Activity heatmap](examples/activity_heatmap.png)

![Reply matrix](examples/reply_matrix.png)

![Monthly timeline](examples/monthly_timeline.png)

![Hourly radar](examples/hourly_radar.png)

![Word cloud](examples/wordcloud.png)

## License

MIT
