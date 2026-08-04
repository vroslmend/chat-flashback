# chat-flashback

Analyzes a Facebook Messenger export and generates reports about a group chat:
yearly recaps, member personalities, reaction dynamics, response-speed leaderboards,
swear-word stats, custom term tracking, conversation starters, reply chains, ghosting
stats, sentiment, and a weirdest statements section.

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
- Message-length and word trends over time.
- Sentiment (VADER). Average mood per member and per year.
- Weirdest statements. All-caps, 3am, punctuation-spiral, and extreme-length messages.
- Self-contained `report.html` with all charts embedded (share one file).
- `--anonymize`. Replaces names with Person A, Person B in all output.
- Supports group and 1-on-1 chats. Detects multiple threads in one export.

## Requirements

Python 3.8+.

```bash
pip install -r requirements.txt
```

`vaderSentiment` powers the sentiment section. If it is not installed the tool
still runs and simply skips sentiment. Sentiment is English-only and may be noisy
on mixed-language chats.

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

Writes `summary.md`, `report.html`, and PNG charts into `output/<thread>/`.

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
- PNG charts: messages per year, activity by hour/weekday, top members, top words,
  yearly recap, reaction dynamics, response speed, swear stats, tracked terms,
  domains, media, reply chains, ghosting, conversation starters, monthly timeline,
  word trends, and sentiment

![Messages per year](examples/messages_by_year.png)

![Yearly recap: messages and top member](examples/yearly_recap.png)

![Most-reacted messages](examples/most_reacted.png)

![Reply chains](examples/reply_chains.png)

![Monthly timeline](examples/monthly_timeline.png)

## License

MIT
