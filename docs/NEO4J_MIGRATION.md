# Neo4j: moving an existing Enterprise volume to Community

## Why this is not a tag change

Volumes created by Neo4j **Enterprise** use its proprietary **`block`** store
format. Community only opens **`aligned`**. Point Community at a `block` volume
and the server **refuses to start** — a store-format incompatibility, not a
version downgrade. Neo4j converts formats only through an **offline** migration
(`neo4j-admin database migrate --to-format`).

The repo default is **Community**, which is correct for the empty volume a fresh
clone gets. If your volume was created by Enterprise, pin Enterprise in your
local (gitignored) `.env` and leave it there until this runbook is done:

```
NEO4J_IMAGE=neo4j:2026-enterprise
NEO4J_ACCEPT_LICENSE_AGREEMENT=eval    # or yes — see "Licence" below
```

`scripts/preflight_neo4j.py` blocks the mismatch automatically; `soultuner.ps1 up`
and `scripts/doctor.py` both run it.

### Licence

Enterprise is commercial software. Neo4j's Docker image accepts exactly two
values and this repo sets neither for you — an Enterprise image with the variable
unset prints the licence and exits:

| Value | Meaning |
|---|---|
| `yes` | you hold a commercial licence agreement |
| `eval` | you accept the Neo4j Evaluation Licence (non-production, time-limited) |

There is no "free for non-commercial" tier. Every Enterprise command below reads
the value from your environment rather than hardcoding one — set it first, in the
shell you are running the migration from.

## Step 0 — confirm the format (read-only, safe on the live DB)

```powershell
docker exec soultuner-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD --format plain `
  "SHOW DATABASES YIELD name, store, currentStatus;"
```

```
name,     store,               currentStatus
"neo4j",  "block-block-1.1",   "online"     <- Enterprise-only: migration needed
"system", "record-aligned-1.1","online"     <- always aligned, ignore it
```

If `neo4j` already says `aligned`, you need none of this — just switch the image.

## Step 1 — record the acceptance numbers FROM THE LIVE DB

You cannot verify the clone without knowing what "correct" is. Capture these
first and keep them next to you:

```powershell
docker exec soultuner-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD --format plain `
  "MATCH (n) RETURN count(n) AS nodes;"
docker exec soultuner-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD --format plain `
  "MATCH ()-[r]->() RETURN count(r) AS rels;"
docker exec soultuner-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD --format plain `
  "MATCH (s:Song) RETURN count(s) AS songs, count(s.muq_embedding) AS muq,
   count(s.m2d2_embedding) AS m2d, count(s.omar_embedding) AS omar,
   count(s.clamp3_embedding) AS clamp3;"
docker exec soultuner-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD --format plain `
  "SHOW INDEXES YIELD name, type, labelsOrTypes, properties
   RETURN name, type, labelsOrTypes, properties ORDER BY name;"
```

Vector indexes are the ones that silently break retrieval if they do not come
across, so check them by name — at the time of writing:
`song_muq_index`, `song_m2d2_index`, `song_omar_index`, `song_clamp3_index`.

## Step 2 — stop Neo4j and back it up

```powershell
docker compose stop neo4j
mkdir neo4j-backup -Force
docker run --rm `
  -v soultuner-agent_neo4j_data:/data `
  -v "${PWD}/neo4j-backup:/backup" `
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=$env:NEO4J_ACCEPT_LICENSE_AGREEMENT `
  neo4j:2026-enterprise `
  neo4j-admin database dump neo4j --to-path=/backup
```

<details><summary>bash</summary>

```bash
docker compose stop neo4j
mkdir -p neo4j-backup
docker run --rm \
  -v soultuner-agent_neo4j_data:/data \
  -v "$PWD/neo4j-backup:/backup" \
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT="$NEO4J_ACCEPT_LICENSE_AGREEMENT" \
  neo4j:2026-enterprise \
  neo4j-admin database dump neo4j --to-path=/backup
```
</details>

## Step 3 — clone the volume (everything after this touches the clone only)

```powershell
docker volume create soultuner-neo4j-aligned
docker run --rm -v soultuner-agent_neo4j_data:/from -v soultuner-neo4j-aligned:/to `
  alpine sh -c "cp -a /from/. /to/"
```

## Step 4 — convert `block` -> `aligned` on the clone, still under Enterprise

Only Enterprise can read the source format, so the conversion itself runs under
Enterprise. Community comes in afterwards.

```powershell
docker run --rm -v soultuner-neo4j-aligned:/data `
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=$env:NEO4J_ACCEPT_LICENSE_AGREEMENT `
  neo4j:2026-enterprise `
  neo4j-admin database migrate neo4j --to-format=aligned
```

## Step 5 — smoke the clone under Community

```powershell
docker run --rm -p 7475:7474 -p 7688:7687 `
  -v soultuner-neo4j-aligned:/data `
  -e NEO4J_AUTH=neo4j/$env:NEO4J_PASSWORD `
  neo4j:2026.03.1-community
```

Deliberately on **7475/7688**: the live Enterprise instance keeps 7474/7687, so a
failed smoke never takes the working system down with it. Then, in another shell,
re-run **every query from Step 1** against `-a bolt://localhost:7688` and compare.

Acceptance — all of these must match the Step 1 numbers exactly:

- [ ] total node count
- [ ] total relationship count
- [ ] `Song` count
- [ ] `muq_embedding` / `m2d2_embedding` / `omar_embedding` / `clamp3_embedding` coverage
- [ ] `SHOW INDEXES` — same set, same names, **all four VECTOR indexes present and `ONLINE`**
- [ ] one real query returns results, e.g.
      `MATCH (s:Song) WHERE s.muq_embedding IS NOT NULL RETURN s.title LIMIT 5;`

Any mismatch: **stop**. Delete the clone (`docker volume rm soultuner-neo4j-aligned`)
and stay on Enterprise. The live volume was never touched, so there is nothing to
undo.

## Step 6 — point compose at the aligned volume

The compose file declares `neo4j_data` as a project-scoped volume, so it resolves
to `soultuner-agent_neo4j_data` — the Enterprise one. Switching means declaring
the aligned volume as external, which is what `docker-compose.neo4j-aligned.yml`
does. It is an override file, so it only applies when you pass it explicitly:

```powershell
docker compose -f docker-compose.yml -f docker-compose.neo4j-aligned.yml up -d neo4j
```

and in `.env`, drop back to the repo default by removing your Enterprise pin:

```
# NEO4J_IMAGE=neo4j:2026-enterprise            <- delete these two lines
# NEO4J_ACCEPT_LICENSE_AGREEMENT=eval
```

Verify the switch actually took effect — not that it started, but that it is on
the right store and the right edition:

```powershell
python scripts/preflight_neo4j.py     # expects: community + aligned
docker exec soultuner-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD --format plain `
  "SHOW DATABASES YIELD name, store;"
```

Make the override permanent by adding it to the compose invocation in
`soultuner.ps1`, or by moving the `external: true` volume declaration into
`docker-compose.yml`.

## Step 7 — retire the old volume (not before)

Keep `soultuner-agent_neo4j_data` **and** `neo4j-backup/` until the app has run
normally for a few days — recommendations, ingest, feedback. Only then:

```powershell
docker volume rm soultuner-agent_neo4j_data
```

## Rollback

At any point before Step 7, roll back by removing the override file from the
compose command and restoring the Enterprise pin in `.env`. The original volume
is untouched and still `block`.
