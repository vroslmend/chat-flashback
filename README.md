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
- Response-speed leaderboard. Median time to reply per member, plus the share of
  each member's turns that got no reply within an hour.
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
- Emoji report. Emoji counts per member and a timeline of favorite emojis over the years.
- Question dynamics. Who asks questions, who answers, who gets left on read, answer speed.
- What the chat was about. TF-IDF topic words per year.
- Running jokes. Repeated phrases that look like inside jokes (frequency, members, years).
- Year in review. A page per year (monthly activity, top words/emojis, jokes) plus an index.
- Message-length and word trends over time.
- Sentiment (VADER). Average mood per member and per year.
- Weirdest statements. All-caps, 3am, punctuation-spiral, and extreme-length messages.
- Media leaderboard. Photos, stickers, GIFs, videos, audio, and file attachments per member.
- Handles newer-format export fields: `gifs`, `videos`, `audio_files`, `files`, `polls`,
  and `is_taken_down`, and de-duplicates messages that appear in multiple files.
- `--check`. Validate an export before analyzing: unknown message types/keys, empty
  messages, media files missing on disk, duplicate messages, and gaps between files.
- Self-contained `report.html` with all charts embedded (share one file). Sticky nav,
  dark/light toggle, and filterable, sortable tables.
- Local chat reader (`--serve`). Browse the chat in a Messenger-style web UI, with
  "on this day" nostalgia, random memories, regex search, and a theme toggle.
- `--anonymize`. Replaces names with Person A, Person B in all output.
- `--tz`, `--config`, `--progress`, and `--incremental` for timezones, config files,
  progress output, and skipping unchanged threads.
- Supports group and 1-on-1 chats. Detects multiple threads in one export.

## Requirements

Python 3.8+.

```bash
pip install -r requirements.txt
```

Or install as a package, which gives you a `chatflashback` command:

```bash
pip install .
chatflashback --input data --output output
```

Optional extras that the tool skips gracefully if missing:
- `vaderSentiment` powers sentiment (English-only, may be noisy on mixed-language chats).
- `wordcloud` powers the word clouds.

Install both with `pip install ".[full]"`.

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
| `--stopwords-file` | Extra stopwords to ignore in word stats, one per line (the built-in list is English only) |
| `--year` | Analyze only one year, e.g. `--year 2017` |
| `--top` | Number of entries in leaderboards (default: 10) |
| `--json` | Also write `summary.json` with the report data as structured JSON |
| `--serve` | Start the local chat reader web UI instead of writing reports |
| `--port` | Port for `--serve` (default: 8080) |
| `--tz` | Timezone for analysis, e.g. `+03:00` or `America/New_York` (Messenger timestamps are UTC; default is your system timezone) |
| `--config` | JSON config file with any of the options above |
| `--skip` | Skip analyses: `jokes`, `sentiment`, `wordcloud`, `topics` (comma-separated) |
| `--progress` | Show phase progress while analyzing |
| `--incremental` | Skip threads that are unchanged since the last run |
| `--check` | Validate the export instead of analyzing (always exits 0) |

Writes `summary.md`, `report.html`, PNG charts, and `year_<year>.html` year-in-review
pages into `output/<thread>/`.

### Config file

Pass options in a JSON file instead of on the command line. CLI flags still win.

```json
{
  "input": "data",
  "output": "out",
  "top": 15,
  "json": true,
  "anonymize": false,
  "track": "lol, bro"
}
```

```bash
python analyze_chat.py --config config.json
```

### Non-English chats

The built-in stopword list is English only, so in a chat that mixes languages the
other language's function words take over the top-word, topic and running-joke
sections. `stopwords/hinglish.txt` ships with common Urdu/Hindi words as typed in
Latin script:

```bash
python analyze_chat.py --input data --stopwords-file stopwords/hinglish.txt
```

Any file works: one word per line, `#` for comments.

### Very large chats

Everything is held in memory, so a chat with hundreds of thousands of messages
needs headroom. Two analyses dominate: running jokes counts every 2-4 word
phrase in the chat and drives peak memory, and sentiment is the slowest phase.
Drop them if a run is too heavy:

```bash
python analyze_chat.py --input data --skip jokes,sentiment
```

Every other report, chart and page is still produced; only those sections are
left out.

### Incremental runs

`--incremental` records a fingerprint (file names, sizes, mtimes) for each thread in
`output/.chatflashback_state.json` and skips threads that have not changed since the
last run. Changing flags like `--year` or `--top` forces a re-run.

### Validating an export

Before analyzing a fresh download, run `--check` to surface anything the tool may not
handle yet and to find broken attachments:

```bash
python analyze_chat.py --input data --check
```

It reports, per thread: message types, unknown `type` values and unknown top-level
message keys (so new export formats are easy to spot), empty messages, media
attachments that are missing on disk, duplicate messages (it de-duplicates them
automatically when analyzing), and gaps over 90 days between message files. Add
`--json` to also write `check.json`. `--check` always exits 0 and never analyzes.

## Chat reader

```bash
python analyze_chat.py --input <thread> --serve
```

Starts a local server on `127.0.0.1:8080` (localhost only, no authentication) and
opens a Messenger-style reader:

- Infinite-scroll feed grouped by day, newest or oldest first
- Sender color chips, inline reactions, media thumbnails, shares and call messages
- Photo/GIF thumbnails, inline video and audio players, and download links for files
- Reply threading, "N years ago" badges, subtle sentiment tint
- Full-text search (with a `.*` regex toggle) and per-member filters
- A jump-to-date control, an "On this day" view across the years, and a random-memory
  "Surprise me" button
- A light/dark theme toggle (remembered between visits)
- Links to the full report and the year-in-review pages

Media files are served from the export folder only; paths are resolved inside the
thread directory so nothing outside it can be read, and files are streamed with
`Range` support so video and audio can seek. Anything that is not an image, video
or audio downloads rather than rendering, since an export can contain `.html` or
`.svg` attachments. Search is a substring scan over the messages (regex is
opt-in) that reports the true match count and shows the first page of hits.

Text that comes from the export — thread titles, names, message bodies — is
escaped everywhere it is rendered, in the reader and in the generated reports.

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
- `year_in_review.html` and `year_<year>.html` - one page per year
- PNG charts: messages per year, activity heatmap, pace trends, activity by
  hour/weekday, top members, top words, word clouds, emoji timeline, yearly recap,
  pair dynamics, hourly radar, reaction dynamics, question dynamics, topics per
  year, running jokes, response speed, swear stats, tracked terms, domains, media
  leaderboard, reply chains, ghosting, monologues, conversation starters, monthly
  timeline, word trends, and sentiment

![Activity heatmap](examples/activity_heatmap.png)

![Reply matrix](examples/reply_matrix.png)

![Monthly timeline](examples/monthly_timeline.png)

![Hourly radar](examples/hourly_radar.png)

![Word cloud](examples/wordcloud.png)

## License

MIT
