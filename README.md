# chat-flashback

Analyzes a Facebook Messenger export and generates reports about a group chat:
yearly recaps, member personalities, reaction dynamics, response-speed leaderboards,
swear-word stats, custom term tracking, and a weirdest statements section.

Runs locally. Your data is not uploaded anywhere.

## Features

- Yearly recaps. Top member, top word, record day per year.
- Member personalities. Signature words, emojis, peak posting hour, night-owl percentage.
- Reaction dynamics. Most-reacted messages, reactor rankings.
- Response-speed leaderboard. Median time to reply per member, ghosted percentage.
- Swear-word analytics. Per-member counts and signature swear words.
- Custom term tracking. Count and chart any words or phrases with `--track`.
- Weirdest statements. All-caps, 3am, punctuation-spiral, and extreme-length messages.
- `--anonymize`. Replaces names with Person A, Person B in all output.
- Supports group and 1-on-1 chats. Detects multiple threads in one export.

## Requirements

Python 3.8+.

```bash
pip install -r requirements.txt
```

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
| `--year` | Analyze only one year, e.g. `--year 2017` |
| `--top` | Number of entries in leaderboards (default: 10) |

Writes `summary.md` plus PNG charts into `output/<thread>/`.

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
- `--anonymize` replaces real names in every chart and report.
- `.gitignore` excludes `data/` and `output/` so an export cannot be committed.

## Sample data

`sample_data/` contains a small synthetic thread used for testing:

```bash
python analyze_chat.py --input sample_data --track "shawarma, bro"
```

## Examples

Sample output from the synthetic thread is in `examples/`:

- `example_summary.md` - a full generated report
- PNG charts: messages per year, activity by hour/weekday, top members, top words,
  yearly recap, reaction dynamics, response speed, swear stats, tracked terms

## License

MIT
