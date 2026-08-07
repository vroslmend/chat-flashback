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
- Word clouds. Overall and for the six busiest members, generated with `wordcloud`.
- Monologues and unsent messages. Longest solo runs ("could've been an email") and `is_unsent`.
- Emoji report. Emoji counts per member and a timeline of favorite emojis over the years.
- Question dynamics. Who asks questions, who answers, who gets left on read, answer speed.
- What the chat was about. TF-IDF topic words per year.
- Running jokes. Repeated phrases that look like inside jokes (frequency, members, years).
- Year in review. A page per year (monthly activity, top words/emojis, jokes) plus an index.
- Group history. Every name the group gave itself and every nickname it gave its
  members, as dated ranges, read back out of Messenger's own event messages.
- Member pages. One per member: their years, the words that set each of their years
  apart from their own others, who they answer, and their most-reacted messages.
- Relationships. Pairs by year, pairs that drifted, who speaks first after a day of
  silence, who gets the last word, and who goes unanswered most.
- Eras. The chat cut into periods where its volume or its vocabulary turned over,
  each named for the word it uses most out of proportion, plus the words the chat
  picked up and stopped saying each year.
- Conversations. The chat cut into conversations wherever nobody spoke for 30
  minutes: who opens them, who has the last word, how long they actually run, the
  longest one ever, and the chat's own longest silences with the message that
  ended each.
- Sleep schedules. Every member's posting hours, year by year rather than
  averaged over all of them, so somebody's hours sliding later reads as a move
  and their 3am tail vanishing reads as a job.
- Guess who said this. A quiz built out of each member's signature words, so the
  answer is gettable rather than a coin flip.
- Message-length and word trends over time.
- Sentiment (VADER). Average mood per member and per year.
- Weirdest statements. All-caps, 3am, punctuation-spiral, and extreme-length messages.
- Media leaderboard. Photos, stickers, GIFs, videos, audio, and file attachments per member.
- Handles newer-format export fields: `gifs`, `videos`, `audio_files`, `files`, `polls`,
  and `is_taken_down`, and de-duplicates messages that appear in multiple files.
- `--check`. Validate an export before analyzing: unknown message types/keys, empty
  messages, media files missing on disk, duplicate messages, and gaps between files.
- Copy-paste floods handled. Vocabulary counts each word or emoji at most three
  times per message, so one pasted wall of the same word cannot decide the top
  words, the topics of a year, or somebody's signature word. Volume stats still
  count every keystroke, and the totals report how many floods there were.
- Bots flagged. Members that are obviously software (`Meta AI`) are labelled
  `(bot)` and kept out of the human awards: fastest replier, best vibes, and the
  weirdest-statements reel.
- Self-contained `report.html` with every table and chart from `summary.md`
  embedded (share one file). Sticky nav, dark/light toggle, and filterable,
  sortable tables.
- Local chat reader (`--serve`). Browse the chat in a Messenger-style web UI, with
  "on this day" nostalgia, random memories, regex search, and a theme toggle.
- Word explorer in the reader. Look up any word, phrase or emoji and get who says
  it, how often per 1,000 messages, when it started, who picked it up from whom,
  the words it keeps company with, and real messages you can jump straight to.
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
| `--names` | Names of people the export doesn't list (e.g. a deleted account shown as "Facebook user"), so they don't read as topic words |
| `--stopwords-file` | Extra stopwords to ignore in word stats, one per line (the built-in list is English only) |
| `--year` | Analyze only one year, e.g. `--year 2017` |
| `--top` | Number of entries in leaderboards and charts (default: 10) |
| `--json` | Also write `summary.json` with the report data as structured JSON |
| `--serve` | Start the local chat reader web UI instead of writing reports |
| `--port` | Port for `--serve` (default: 8080) |
| `--no-index` | Skip the word-search index when serving. Starts faster; the word explorer is unavailable |
| `--tz` | Timezone for analysis, e.g. `+03:00` or `America/New_York` (Messenger timestamps are UTC; default is your system timezone) |
| `--config` | JSON config file with any of the options above |
| `--skip` | Skip analyses: `jokes`, `sentiment`, `wordcloud`, `topics`, `narratives` (comma-separated) |
| `--progress` | Show phase progress while analyzing |
| `--incremental` | Skip threads that are unchanged since the last run |
| `--check` | Validate the export instead of analyzing (always exits 0) |

Writes `summary.md`, `report.html`, PNG charts, `year_<year>.html` year-in-review
pages, `group_history.html`, `relationships.html`, `eras.html`, `sessions.html`,
`quiz.html` and a `member_<name>.html` per member into `output/<thread>/`. Every
page is linked from the report's top bar.

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

### Group history, relationships and eras

Five pages sit beside the report, plus one page per member.

**Group history** reads back the messages Messenger writes about the group itself
— renames, nicknames, joins and removals. They are dropped from the vocabulary
(otherwise "named the group" ranks as a running joke said by everyone for years),
but they are a record nobody has read: on a nine-year chat, 520 group names and
369 nickname changes. Nicknames are shown as dated ranges, because the question
people ask is what someone was called in 2021, not when a name was set.

**Relationships** counts a pair's interactions — a reply inside the hour, or a
reaction — per year, and flags a pair as drifted when its share of either
member's own interaction moves by more than half between the pair's peak year and
the most recent year the export covers end to end. The share is what makes it
drift rather than the chat simply going quiet. It also reports who speaks first
after a day of silence, who gets the last word, and whose messages go unanswered,
as a share of their own messages so the loudest member does not win by volume.

**Eras** cuts the chat into periods. The rule: a month opens a new era when the
three months from it carry less than half or more than double the messages of the
three before it, or when fewer than a third of the previous quarter's top words
survive into this one. Quarters too small to have a character of their own cannot
open an era, and eras shorter than six months are merged into their neighbour. An
era is named for the word it uses most out of proportion to the rest of the chat,
which is not the same as its most common word — on a real chat, tf-idf named four
eras out of five after the chat's single commonest word. Underneath it, the words
the chat first said and last said each year, which is the plainest version of the
same story.

**Conversations** splits the chat wherever nobody spoke for 30 minutes — the same
gap the rest of the report uses, so the two cannot disagree about where a
conversation ends. Openers and closers are given as a count and as a rate per 100
of that member's own messages, because the count alone just ranks by who talks
most; the rate is what separates someone who always speaks first from someone who
is simply always there. Conversation length is reported as percentiles rather than
an average, since the distribution runs from two-message exchanges to all-nighters
and a mean describes neither. Underneath sits the inverse: the chat's longest
silences, and the message that broke each one.

**Quiz** is "guess who said this". A message only qualifies if it uses one of its
sender's signature words, so the answer is gettable; messages that name somebody
give it away and are left out, as are bots. It is seeded, so regenerating the
report does not reshuffle the questions.

### Non-English chats

The built-in stopword list is English only, so in a chat that mixes languages the
other language's function words take over the top-word, topic and running-joke
sections. `stopwords/hinglish.txt` ships with common Urdu/Hindi words as typed in
Latin script:

```bash
python analyze_chat.py --input data --stopwords-file stopwords/hinglish.txt
```

Any file works: one word per line, `#` for comments. Chat spelling is not
standard, so a word usually needs several entries (`mein`, `mei`, `mai`) before
it stops showing up in the results.

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
- Photo/GIF thumbnails, inline video and audio players, and download links for files.
  Media that is not on disk says so in place of the thumbnail, rather than leaving a
  blank message — most exports ship with only a fraction of their attachments
- Reply threading, "N years ago" badges, subtle sentiment tint
- Full-text search (with a `.*` regex toggle) and per-member filters
- A jump-to-date control, an "On this day" view across the years, and a random-memory
  "Surprise me" button
- A light/dark theme toggle (remembered between visits)
- The word explorer, behind the "Words" button
- Links to the full report and the year-in-review pages

### Word explorer

Type a word into the panel behind the reader's "Words" button and it comes back
with the whole life of that word in the chat: total uses and how many messages
they land in, a per-member table with a per-1,000-messages rate so a quiet member
who says it constantly is not buried under a chatty one, the year it peaked, how
often it is sent on its own, whether it pulls more reactions than the chat's
average, who said it first and how many days everyone else took to pick it up,
the words that sit beside it more often than chance predicts, and example
messages with a button that jumps the feed to that moment.

Matching is exact: `bruh` does not silently include `bruhh`. Other spellings that
differ only in held-down letters are listed separately, and "count spellings
together" folds them into the totals. Autocomplete suggests words by how often
they are used.

Type more than one word and it becomes a phrase, counted only where those words
sit side by side — `full send` ignores "send me the full list". Emoji are indexed
as words, so `😂` works the same way, and so does `😂😂` or `lol 😂`. Punctuation
and capitals are ignored, and a phrase may be built entirely out of stopwords
(`the end` is a fair question even though `the` is not).

"Show all N in the feed" turns the reader into every message holding that word,
oldest first, grouped by day. Click any one of them and the feed opens the whole
conversation around it, so you can read what the word was actually about.

The index is built once at startup and lives in memory. On a 1.79M-message chat
it takes about 30 seconds and peaks near 180 MB while building, for 140,755
distinct words. A lookup then takes milliseconds for an ordinary word, and up to
about 3 seconds for one of the most common words in the chat, since every
statistic is computed on demand over the messages that matched. Phrases need no
index of their own — the candidate messages are the ones containing every word,
so `in the` came back in 1.0 s and `what the hell` in 0.04 s on the same chat.
Pass `--no-index` to skip the build if you only want to read.

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
  hour/weekday, hours by year, top members, top words, word clouds, emoji timeline, yearly recap,
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
