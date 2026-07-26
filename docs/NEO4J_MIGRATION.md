# Neo4j: moving an existing volume from Enterprise to Community

## Why this is not just a tag change

The default `docker-compose.yml` runs **Enterprise** on purpose. An existing
SoulTuner volume was created by Enterprise and uses its proprietary **`block`**
store format (`store = block-block-1.1`). Community only supports the **`aligned`**
format. Pointing Community at a `block` volume **fails to start** — this is a
store-format incompatibility, not a version downgrade, and Neo4j only converts
formats through an **offline** migration.

See the Neo4j operations manual, "Store formats", for the `block` vs `aligned`
distinction.

- **Fresh, empty volume** → you may run Community directly:
  `NEO4J_IMAGE=neo4j:2026.03.1-community` (Community ignores the licence variable).
- **Existing `block` volume** → follow the steps below. Never switch the tag on
  the live volume.

## Runbook (work on a COPY, never the live volume)

Confirm the current format first:

```bash
docker exec soultuner-neo4j \
  cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "CALL db.info() YIELD name RETURN name;"     # sanity check it's up
docker exec soultuner-neo4j \
  neo4j-admin database info neo4j               # shows storeFormat = block-...
```

1. **Keep running Enterprise for now.** Do not rebuild the Neo4j service.
2. **Dump a backup** (offline dump of the live DB):
   ```bash
   docker compose stop neo4j
   docker run --rm -v soultuner-agent_neo4j_data:/data \
     -v "$PWD/neo4j-backup:/backup" neo4j:2026.03.1-enterprise \
     neo4j-admin database dump neo4j --to-path=/backup
   ```
3. **Clone the data volume** so the original is untouched:
   ```bash
   docker volume create soultuner-neo4j-aligned
   docker run --rm -v soultuner-agent_neo4j_data:/from \
     -v soultuner-neo4j-aligned:/to alpine \
     sh -c "cp -a /from/. /to/"
   ```
4. **Convert `block` → `aligned` on the clone**, still under Enterprise
   (`--to-format` performs the format migration):
   ```bash
   docker run --rm -v soultuner-neo4j-aligned:/data \
     -e NEO4J_ACCEPT_LICENSE_AGREEMENT=eval neo4j:2026.03.1-enterprise \
     neo4j-admin database migrate neo4j --to-format=aligned
   ```
5. **Load the aligned clone under Community** and verify:
   ```bash
   docker run --rm -p 7474:7474 -p 7687:7687 \
     -v soultuner-neo4j-aligned:/data \
     -e NEO4J_AUTH=neo4j/"$NEO4J_PASSWORD" neo4j:2026.03.1-community
   ```
   Then check counts and indexes match the Enterprise original:
   ```cypher
   MATCH (n) RETURN count(n);                       // node count
   MATCH ()-[r]->() RETURN count(r);                // relationship count
   SHOW INDEXES;                                     // incl. the vector index
   ```
6. **Only after the counts, relationships, indexes and the vector index all
   match**, repoint compose at the aligned volume + Community image and retire the
   Enterprise volume.

If anything fails to match, discard the clone and stay on Enterprise — the live
volume was never touched.
