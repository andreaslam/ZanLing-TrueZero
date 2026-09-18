use rusqlite::{params, Connection};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_GRAPH_PLAYERS: usize = 10_000;
pub const GRAPH_REEXPAND_INTERVAL_SECS: u64 = 1_800;

pub fn open_player_graph_database(
    path: &str,
) -> Result<Connection, Box<dyn std::error::Error + Send + Sync>> {
    let connection = Connection::open(path)?;

    connection.execute_batch("PRAGMA foreign_keys = ON;")?;

    let has_legacy_players = connection.query_row(
        "SELECT EXISTS(
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'players'
                )",
        [],
        |row| row.get::<_, bool>(0),
    )? && !connection
        .prepare("PRAGMA table_info(players)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|column| column == "id");

    if has_legacy_players {
        connection.execute_batch("BEGIN IMMEDIATE;")?;
        connection.execute_batch(
            r#"
            ALTER TABLE players RENAME TO players_legacy;
            ALTER TABLE player_edges RENAME TO player_edges_legacy;
            ALTER TABLE human_challenges RENAME TO human_challenges_legacy;
            "#,
        )?;
    }

    connection.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            is_bot INTEGER CHECK (is_bot IS NULL OR is_bot IN (0, 1)),
            first_seen_at INTEGER NOT NULL,
            last_seen_online_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS discovery_sources (
            id INTEGER PRIMARY KEY,
            source_key TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS player_discoveries (
            player_id INTEGER NOT NULL REFERENCES players(id),
            source_id INTEGER NOT NULL REFERENCES discovery_sources(id),
            first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL,
            PRIMARY KEY (player_id, source_id)
        );

        CREATE TABLE IF NOT EXISTS player_expansions (
            player_id INTEGER PRIMARY KEY REFERENCES players(id),
            last_expanded_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_edges (
            player_id INTEGER NOT NULL REFERENCES players(id),
            opponent_id INTEGER NOT NULL REFERENCES players(id),
            first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL,
            PRIMARY KEY (player_id, opponent_id),
            CHECK (player_id <> opponent_id)
        );

        CREATE TABLE IF NOT EXISTS human_challenges (
            player_id INTEGER PRIMARY KEY REFERENCES players(id),
            last_challenged_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_player_expansions_due
            ON player_expansions(last_expanded_at);

        CREATE INDEX IF NOT EXISTS idx_players_bot
            ON players(is_bot);

        CREATE INDEX IF NOT EXISTS idx_player_edges_opponent
            ON player_edges(opponent_id);

        CREATE INDEX IF NOT EXISTS idx_player_discoveries_source
            ON player_discoveries(source_id);
        "#,
    )?;

    if has_legacy_players {
        connection.execute_batch(
            r#"
            INSERT INTO players (
                username,
                is_bot,
                first_seen_at,
                last_seen_online_at
            )
            SELECT username, is_bot, first_seen_at, last_seen_online_at
            FROM players_legacy;

            INSERT INTO discovery_sources (source_key)
            SELECT DISTINCT discovered_from
            FROM players_legacy
            WHERE discovered_from IS NOT NULL;

            INSERT INTO player_discoveries (
                player_id,
                source_id,
                first_seen_at,
                last_seen_at
            )
            SELECT p.id, s.id, p.first_seen_at, p.first_seen_at
            FROM players_legacy old
            JOIN players p ON p.username = old.username
            JOIN discovery_sources s
                ON s.source_key = old.discovered_from
            WHERE old.discovered_from IS NOT NULL;

            INSERT INTO player_expansions (player_id, last_expanded_at)
            SELECT p.id, old.last_expanded_at
            FROM players_legacy old
            JOIN players p ON p.username = old.username
            WHERE old.last_expanded_at IS NOT NULL;

            INSERT INTO player_edges (
                player_id,
                opponent_id,
                first_seen_at,
                last_seen_at
            )
            SELECT source.id, opponent.id, old.last_seen_at, old.last_seen_at
            FROM player_edges_legacy old
            JOIN players source
                ON source.username = old.username
            JOIN players opponent
                ON opponent.username = old.opponent
            WHERE source.id <> opponent.id;

            INSERT INTO human_challenges (player_id, last_challenged_at)
            SELECT p.id, old.last_challenged_at
            FROM human_challenges_legacy old
            JOIN players p ON p.username = old.username;

            DROP TABLE players_legacy;
            DROP TABLE player_edges_legacy;
            DROP TABLE human_challenges_legacy;
            "#,
        )?;
        connection.execute_batch("COMMIT;")?;
    }

    Ok(connection)
}

pub fn unix_time_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/*
 * Insert a discovered player if we have not seen them before.
 *
 * `is_bot` is deliberately nullable because discovering somebody from a game
 * does not itself tell us whether the account is a bot.
 */
pub fn graph_insert_player(
    db: &Connection,
    username: &str,
    is_bot: Option<bool>,
    source: &str,
) -> Result<(), rusqlite::Error> {
    let now = unix_time_now();

    db.execute(
        r#"
        INSERT INTO players (
            username,
            is_bot,
            first_seen_at
        )
        VALUES (?1, ?2, ?3)
        ON CONFLICT(username) DO UPDATE SET
            is_bot = COALESCE(
                excluded.is_bot,
                players.is_bot
            )
        "#,
        params![
            username,
            is_bot.map(|value| {
                if value {
                    1
                } else {
                    0
                }
            }),
            now,
        ],
    )?;

    db.execute(
        "INSERT INTO discovery_sources (source_key)
         VALUES (?1)
         ON CONFLICT(source_key) DO NOTHING",
        params![source],
    )?;

    db.execute(
        r#"
        INSERT INTO player_discoveries (
            player_id,
            source_id,
            first_seen_at,
            last_seen_at
        )
        SELECT p.id, s.id, ?1, ?1
        FROM players p
        CROSS JOIN discovery_sources s
        WHERE p.username = ?2 AND s.source_key = ?3
        ON CONFLICT(player_id, source_id) DO UPDATE SET
            last_seen_at = excluded.last_seen_at
        "#,
        params![now, username, source],
    )?;

    Ok(())
}

pub fn graph_mark_online(db: &Connection, username: &str) -> Result<(), rusqlite::Error> {
    let now = unix_time_now();

    db.execute(
        r#"
        UPDATE players
        SET last_seen_online_at = ?1
        WHERE username = ?2
        "#,
        params![now, username,],
    )?;

    Ok(())
}

pub fn graph_mark_expanded(db: &Connection, username: &str) -> Result<(), rusqlite::Error> {
    let now = unix_time_now();

    db.execute(
        r#"
        INSERT INTO player_expansions (player_id, last_expanded_at)
        SELECT id, ?1
        FROM players
        WHERE username = ?2
        ON CONFLICT(player_id) DO UPDATE SET
            last_expanded_at = excluded.last_expanded_at
        "#,
        params![now, username,],
    )?;

    Ok(())
}

pub fn graph_insert_edge(
    db: &Connection,
    username: &str,
    opponent: &str,
) -> Result<(), rusqlite::Error> {
    let now = unix_time_now();

    db.execute(
        r#"
        INSERT INTO player_edges (
            player_id,
            opponent_id,
            first_seen_at,
            last_seen_at
        )
        SELECT source.id, opponent.id, ?3, ?3
        FROM players source
        CROSS JOIN players opponent
        WHERE source.username = ?1
          AND opponent.username = ?2
          AND source.id <> opponent.id
        ON CONFLICT(player_id, opponent_id) DO UPDATE SET
            last_seen_at = excluded.last_seen_at
        "#,
        params![username, opponent, now,],
    )?;

    Ok(())
}

/*
 * Keep the graph bounded. Players with a recorded game relationship are
 * retained ahead of players that were only discovered from an online scan.
 * Within either group, the least recently online players are evicted first.
 */
pub fn graph_prune_players(db: &Connection) -> Result<usize, rusqlite::Error> {
    let player_count: usize = db.query_row("SELECT COUNT(*) FROM players", [], |row| row.get(0))?;

    let remove_count = player_count.saturating_sub(MAX_GRAPH_PLAYERS);

    if remove_count == 0 {
        return Ok(0);
    }

    db.execute_batch("BEGIN IMMEDIATE;")?;

    let result = (|| {
        let mut statement = db.prepare(
            r#"
                    SELECT players.id
                    FROM players
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM human_challenges
                        WHERE human_challenges.player_id = players.id
                    )
                    ORDER BY
                        CASE WHEN EXISTS (
                            SELECT 1
                            FROM player_edges
                            WHERE player_edges.player_id = players.id
                               OR player_edges.opponent_id = players.id
                        ) THEN 1 ELSE 0 END ASC,
                        COALESCE(players.last_seen_online_at, 0) ASC,
                        players.first_seen_at ASC,
                        players.id ASC
                    LIMIT ?1
                    "#,
        )?;

        let ids = statement
            .query_map(params![remove_count as i64], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        for id in &ids {
            db.execute(
                "DELETE FROM player_edges
                     WHERE player_id = ?1 OR opponent_id = ?1",
                params![id],
            )?;
            db.execute(
                "DELETE FROM player_discoveries
                     WHERE player_id = ?1",
                params![id],
            )?;
            db.execute(
                "DELETE FROM player_expansions
                     WHERE player_id = ?1",
                params![id],
            )?;
            db.execute(
                "DELETE FROM human_challenges
                     WHERE player_id = ?1",
                params![id],
            )?;
            db.execute("DELETE FROM players WHERE id = ?1", params![id])?;
        }

        Ok(ids.len())
    })();

    match result {
        Ok(removed) => {
            db.execute_batch("COMMIT;")?;
            Ok(removed)
        }
        Err(error) => {
            let _ = db.execute_batch("ROLLBACK;");
            Err(error)
        }
    }
}

pub fn graph_get_expansion_candidates(
    db: &Connection,
    limit: usize,
) -> Result<Vec<String>, rusqlite::Error> {
    let cutoff = unix_time_now() - GRAPH_REEXPAND_INTERVAL_SECS as i64;

    let mut statement = db.prepare(
        r#"
            SELECT username
            FROM players
            LEFT JOIN player_expansions
                ON player_expansions.player_id = players.id
            WHERE player_expansions.last_expanded_at IS NULL
               OR player_expansions.last_expanded_at < ?1
            ORDER BY
                CASE
                    WHEN player_expansions.last_expanded_at IS NULL
                    THEN 0
                    ELSE 1
                END,
                player_expansions.last_expanded_at ASC
            LIMIT ?2
            "#,
    )?;

    let rows = statement.query_map(params![cutoff, limit as i64,], |row| {
        row.get::<_, String>(0)
    })?;

    let mut result = Vec::new();

    for row in rows {
        result.push(row?);
    }

    Ok(result)
}
