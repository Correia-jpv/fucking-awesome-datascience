# Contributing

Thanks for taking the time to contribute.

## What you can contribute

- Add new resources (links, tools, courses, books, etc.)
- Fix broken links
- Remove outdated resources
- Improve organization (move items to a more appropriate section)
- Fix typos and formatting issues

## How to add a new resource

- Add the resource to the most relevant section in `README.md`.
- Use the existing style for that section.
- Prefer authoritative sources (official docs, original GitHub repo, publisher site).

### Entry format

Use a consistent bullet format:

- `[Name](URL)` - Short description.

Guidelines:

- Keep descriptions short (ideally one sentence).
- Use title case for the link text when it matches surrounding entries.
- Avoid marketing language and superlatives.
- If there are similar items already listed, add yours near them.
- If the section is alphabetized, keep it alphabetized.

## Links

- Prefer `https://` when available.
- Avoid URL shorteners.
- If a site blocks automated link checking but is a valid resource, it may need to be ignored by the link checker configuration.

## Running link checks locally

This repository contains CI configurations for link checking.

### Option A: awesome_bot (Ruby)

If you have Ruby installed:

1. Install the gem:

   `gem install awesome_bot`

2. Run:

   `awesome_bot README.md`

### Option B: markdown-link-check (Node)

If you have Node installed:

1. Install:

   `npm i -g markdown-link-check`

2. Run:

   `markdown-link-check README.md -c mlc_config.json`

## Pull request checklist

- Your change is in the right section.
- The link works.
- The description is short and matches the repo style.
- Link checks pass (or you explained why they fail and proposed an ignore pattern).

## Code of Conduct

By participating in this project, you agree to abide by the `CODE_OF_CONDUCT.md`.
