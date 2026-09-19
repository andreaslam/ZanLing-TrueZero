use super::*;
use tzrust::lichess_graph::*;

pub(super) async fn fetch_player_opponents(
    username: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let url =
        format!(
            "https://lichess.org/api/games/user/{}?max={}&moves=false&clocks=false&evals=false&opening=false&literate=false&ongoing=false&finished=true",
            username,
            GRAPH_GAMES_PER_PLAYER,
        );

    let response = reqwest::Client::new()
        .get(&url)
        .header(
            reqwest::header::USER_AGENT,
            "TrueZero Lichess player discovery",
        )
        .header(reqwest::header::ACCEPT, "application/x-ndjson")
        .send()
        .await?
        .error_for_status()?;

    let body = response.text().await?;

    let mut opponents = Vec::new();

    let mut seen = HashSet::new();

    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let game: serde_json::Value = match serde_json::from_str(line) {
            Ok(value) => value,

            Err(error) => {
                eprintln!("Failed to parse game JSON for {}: {}", username, error);

                continue;
            }
        };

        let white = game
            .get("players")
            .and_then(|players| players.get("white"))
            .and_then(|white| white.get("user"))
            .and_then(|user| user.get("name"))
            .and_then(|name| name.as_str());

        let black = game
            .get("players")
            .and_then(|players| players.get("black"))
            .and_then(|black| black.get("user"))
            .and_then(|user| user.get("name"))
            .and_then(|name| name.as_str());

        let opponent = match (white, black) {
            (Some(white), Some(black)) if white.eq_ignore_ascii_case(username) => Some(black),

            (Some(white), Some(black)) if black.eq_ignore_ascii_case(username) => Some(white),

            /*
             * This also handles cases where the API gives a
             * username with different casing.
             */
            (Some(white), Some(black)) => {
                if !white.eq_ignore_ascii_case(username) {
                    Some(white)
                } else if !black.eq_ignore_ascii_case(username) {
                    Some(black)
                } else {
                    None
                }
            }

            _ => None,
        };

        let Some(opponent) = opponent else {
            continue;
        };

        if opponent.is_empty() || opponent.eq_ignore_ascii_case(username) {
            continue;
        }

        if seen.insert(opponent.to_ascii_lowercase()) {
            opponents.push(opponent.to_string());
        }
    }

    Ok(opponents)
}

/*
 * Expand the graph.
 *
 * Seed source:
 *
 *     currently-online bots
 *
 * Then:
 *
 *     bot -> recent opponents
 *     opponent -> their recent opponents
 *     opponent -> their recent opponents
 *     ...
 *
 * The database makes this persistent across restarts.
 */
pub(super) async fn expand_player_graph(
    client: &LichessClient,
    our_username: &str,
    graph_db: &Arc<Mutex<Connection>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Expanding persistent Lichess player graph...");

    /*
     * Seed the graph with both online bots and online humans. Human users
     * discovered here are expanded through the same recent-game API, so a
     * human found in /player can discover additional players on later scans.
     */
    let online_humans = fetch_online_player_usernames().await?;

    let bot_api = client.bot();

    let mut online_bots = bot_api.online(None).await?;

    let mut human_seed_count = 0usize;

    let mut bot_seed_count = 0usize;

    {
        let db = graph_db.lock().await;

        for username in online_humans {
            if username.eq_ignore_ascii_case(our_username) {
                continue;
            }

            graph_insert_player(&db, &username, None, "online_human")?;

            human_seed_count += 1;
        }

        while let Some(bot_result) = online_bots.next().await {
            let bot = match bot_result {
                Ok(bot) => bot,

                Err(error) => {
                    eprintln!("Graph bot-list error: {}", error);

                    continue;
                }
            };

            if bot.username.eq_ignore_ascii_case(our_username) {
                continue;
            }

            graph_insert_player(&db, &bot.username, Some(true), "online_bot")?;

            bot_seed_count += 1;
        }
    }

    println!(
        "Graph: seeded {} currently-online humans and {} bots.",
        human_seed_count, bot_seed_count,
    );

    /*
     * Now take a bounded number of graph nodes and expand them.
     *
     * We intentionally do this sequentially. This makes the graph
     * traversal predictable and avoids hammering the public game API.
     */
    let expansion_candidates = {
        let db = graph_db.lock().await;

        graph_get_expansion_candidates(&db, MAX_GRAPH_PLAYERS_PER_SCAN)?
    };

    println!(
        "Graph: {} players selected for expansion.",
        expansion_candidates.len()
    );

    let mut expanded = 0usize;

    let mut discovered = 0usize;

    for username in expansion_candidates {
        println!("Graph: expanding {}...", username);

        let opponents = match fetch_player_opponents(&username).await {
            Ok(opponents) => opponents,

            Err(error) => {
                eprintln!("Graph: failed to fetch games for {}: {}", username, error);

                continue;
            }
        };

        {
            let db = graph_db.lock().await;

            for opponent in &opponents {
                if opponent.eq_ignore_ascii_case(our_username) {
                    continue;
                }

                let existed: bool = db.query_row(
                    r#"
                        SELECT EXISTS(
                            SELECT 1
                            FROM players
                            WHERE username = ?1
                        )
                        "#,
                    params![opponent],
                    |row| row.get(0),
                )?;

                graph_insert_player(&db, opponent, None, &format!("game: {}", username))?;

                graph_insert_edge(&db, &username, opponent)?;

                if !existed {
                    discovered += 1;
                }
            }

            graph_mark_expanded(&db, &username)?;
        }

        expanded += 1;

        println!(
            "Graph: {} produced {} unique recent opponents.",
            username,
            opponents.len()
        );
    }

    let pruned = {
        let db = graph_db.lock().await;

        graph_prune_players(&db)?
    };

    if pruned > 0 {
        println!(
            "Graph: pruned {} players to stay within the {}-player limit.",
            pruned, MAX_GRAPH_PLAYERS
        );
    }

    let graph_size: usize = {
        let db = graph_db.lock().await;

        db.query_row("SELECT COUNT(*) FROM players", [], |row| row.get(0))?
    };

    let edge_count: usize = {
        let db = graph_db.lock().await;

        db.query_row("SELECT COUNT(*) FROM player_edges", [], |row| row.get(0))?
    };

    println!(
        "Player graph expansion complete: \
         {} players expanded, \
         {} newly discovered players, \
         {} total players, \
         {} total edges.",
        expanded, discovered, graph_size, edge_count
    );

    Ok(())
}

/*
 * Return persisted graph users who are currently online.
 *
 * This intentionally uses the status endpoint instead of intersecting with
 * /player: /player is only a discovery source and may not contain everybody
 * already known by the graph.
 */
pub(super) async fn graph_online_candidates(
    client: &LichessClient,
    graph_db: &Arc<Mutex<Connection>>,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let graph_usernames: Vec<String> = {
        let db = graph_db.lock().await;

        let mut statement = db.prepare(
            r#"
                    SELECT players.username
                    FROM players
                    LEFT JOIN player_expansions
                        ON player_expansions.player_id = players.id
                    WHERE players.is_bot IS NULL OR players.is_bot = 0
                    ORDER BY players.last_seen_online_at DESC,
                             player_expansions.last_expanded_at ASC
                    LIMIT ?1
                    "#,
        )?;

        let rows = statement.query_map(params![MAX_ONLINE_HUMAN_CANDIDATES as i64], |row| {
            row.get::<_, String>(0)
        })?;

        let mut result = Vec::new();

        for row in rows {
            result.push(row?);
        }

        result
    };

    if graph_usernames.is_empty() {
        return Ok(Vec::new());
    }

    let candidate_ids = graph_usernames
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    let statuses = client
        .users()
        .statuses(&candidate_ids, Some(false), Some(false), Some(false))
        .await?;

    let mut result = Vec::new();

    let db = graph_db.lock().await;

    for status in statuses {
        let username = status.user.name;

        let is_bot = matches!(status.user.title, Some(LichessTitle::Bot));

        graph_insert_player(&db, &username, Some(is_bot), "graph_status")?;

        if status.online == Some(true) {
            graph_mark_online(&db, &username)?;

            if !is_bot {
                result.push(username);
            }
        }
    }

    Ok(result)
}

/*
 * Persistent human challenge cooldown. This survives process restarts because
 * the last successful challenge time is stored in the matchmaking database.
 */
pub(super) fn graph_human_challenge_is_on_cooldown(
    db: &Connection,
    username: &str,
) -> Result<Option<i64>, rusqlite::Error> {
    let last_challenged_at = db
        .query_row(
            r#"
            SELECT human_challenges.last_challenged_at
            FROM human_challenges
            JOIN players
                ON players.id = human_challenges.player_id
            WHERE players.username = ?1
            "#,
            params![username],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;

    let Some(last_challenged_at) = last_challenged_at else {
        return Ok(None);
    };

    let cooldown_seconds = HUMAN_CHALLENGE_COOLDOWN_DAYS.saturating_mul(24 * 60 * 60);

    let age = unix_time_now().saturating_sub(last_challenged_at);

    if age < cooldown_seconds {
        Ok(Some(cooldown_seconds.saturating_sub(age)))
    } else {
        Ok(None)
    }
}

pub(super) fn graph_mark_human_challenged(
    db: &Connection,
    username: &str,
) -> Result<(), rusqlite::Error> {
    db.execute(
        r#"
        INSERT INTO human_challenges (
            player_id,
            last_challenged_at
        )
        SELECT id, ?2
        FROM players
        WHERE username = ?1
        ON CONFLICT(player_id) DO UPDATE SET
            last_challenged_at = excluded.last_challenged_at
        "#,
        params![username, unix_time_now()],
    )?;

    Ok(())
}

/*
 * ============================================================================
 * HUMAN MATCHMAKING
 * ============================================================================
 *
 * The original /player discovery remains the first source.
 *
 * The graph database is an additional source.
 */
pub(super) async fn challenge_online_humans(
    client: &LichessClient,
    our_username: &str,
    pending_challenges: &mut HashSet<String>,
    api_request_lock: &Arc<Mutex<()>>,
    graph_db: &Arc<Mutex<Connection>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Scanning currently-online Lichess humans...");

    let our_perfs = current_perfs(client).await?;

    let our_rating = match our_perfs
        .as_ref()
        .and_then(|perfs| perfs.blitz.as_ref().map(|blitz| blitz.rating))
    {
        Some(rating) => rating,

        None => {
            println!(
                "Cannot run human matchmaking: \
                     our account has no Blitz rating."
            );

            return Ok(());
        }
    };

    println!("Our current Blitz rating: {}", our_rating);

    /*
     * ORIGINAL SOURCE:
     *
     * /player
     */
    let mut online_usernames = fetch_online_player_usernames().await?;

    /*
     * NEW SOURCE:
     *
     * persistent graph source. Its status is queried live below, so we do
     * not assume a graph user is still online merely because they were
     * online when discovered.
     */
    let graph_candidates = graph_online_candidates(client, graph_db).await?;

    let mut seen: HashSet<String> = online_usernames
        .iter()
        .map(|username| username.to_ascii_lowercase())
        .collect();

    let mut graph_added = 0usize;

    for username in graph_candidates {
        if seen.insert(username.to_ascii_lowercase()) {
            online_usernames.push(username);

            graph_added += 1;
        }
    }

    println!(
        "Graph source added {} online \
         candidates to the direct /player source.",
        graph_added
    );

    println!(
        "Combined online-player source yielded {} \
         unique usernames.",
        online_usernames.len()
    );

    if online_usernames.is_empty() {
        println!("No online-player usernames discovered.");

        return Ok(());
    }

    let candidate_usernames = online_usernames
        .into_iter()
        .filter(|username| !username.eq_ignore_ascii_case(our_username))
        .take(MAX_ONLINE_HUMAN_CANDIDATES)
        .collect::<Vec<_>>();

    let stale_pending = {
        let db = graph_db.lock().await;
        pending_challenges
            .iter()
            .filter_map(|username| match graph_player_exists(&db, username) {
                Ok(true) => None,
                Ok(false) => Some(Ok(username.clone())),
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    for username in stale_pending {
        pending_challenges.remove(&username);
    }

    if candidate_usernames.is_empty() {
        println!(
            "No online human candidates after \
             excluding ourselves."
        );

        return Ok(());
    }

    let candidate_ids = candidate_usernames
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    let users = client
        .users()
        .get_many(&candidate_ids, Some(false), Some(false))
        .await?;

    let mut eligible_count = 0usize;

    let mut challenge_count = 0usize;

    for user in users {
        if challenge_count >= MAX_HUMAN_CHALLENGES_PER_SCAN {
            println!(
                "Reached human matchmaking limit of {} \
                 challenge attempts this scan.",
                MAX_HUMAN_CHALLENGES_PER_SCAN
            );

            break;
        }

        let username = user.username.clone();

        if username.eq_ignore_ascii_case(our_username) {
            continue;
        }

        /*
         * Keep the graph database's BOT classification current.
         */
        {
            let db = graph_db.lock().await;

            let is_bot = matches!(user.title, Some(LichessTitle::Bot));

            graph_insert_player(&db, &username, Some(is_bot), "online_profile")?;
        }

        if matches!(user.title, Some(LichessTitle::Bot)) {
            println!("Skipping {}: account is a BOT.", username);

            continue;
        }

        let candidate_rating = match user
            .perfs
            .as_ref()
            .and_then(|perfs| perfs.blitz.as_ref().map(|blitz| blitz.rating))
        {
            Some(rating) => rating,

            None => {
                println!("Skipping {}: no Blitz rating.", username);

                continue;
            }
        };

        let rating_difference = candidate_rating as i32 - our_rating as i32;

        eligible_count += 1;

        if pending_challenges.contains(&username) {
            println!(
                "Skipping {}: outgoing human challenge \
                 already pending.",
                username
            );

            continue;
        }

        let cooldown_remaining = {
            let db = graph_db.lock().await;

            graph_human_challenge_is_on_cooldown(&db, &username)?
        };

        if let Some(remaining_seconds) = cooldown_remaining {
            println!(
                "Skipping {}: persistent human challenge \
                 cooldown active ({}s remaining; cooldown={} days).",
                username, remaining_seconds, HUMAN_CHALLENGE_COOLDOWN_DAYS
            );

            continue;
        }

        pending_challenges.insert(username.clone());

        challenge_count += 1;

        let challenge_color = challenge_color_for_bot(&username);

        let time_control = random_time_control();
        let (initial_seconds, increment_seconds) = time_control.clock();
        let rated = HUMAN_CHALLENGE_RATED && time_control.has_rating(our_perfs.as_ref());

        println!(
            "Challenging online human {} \
             (Blitz {}, ours {}, difference {:+}) \
             as {:?} with {:?} ({}+{} {}, {}/{})",
            username,
            candidate_rating,
            our_rating,
            rating_difference,
            challenge_color,
            time_control,
            initial_seconds / 60,
            increment_seconds,
            if rated { "rated" } else { "casual" },
            challenge_count,
            MAX_HUMAN_CHALLENGES_PER_SCAN
        );

        let result = {
            let _api_guard = api_request_lock.lock().await;

            client
                .challenges()
                .challenge(&username)
                .color(challenge_color)
                .rated(rated)
                .clock(initial_seconds, increment_seconds)
                .send()
                .await
        };

        match result {
            Ok(challenge) => {
                println!(
                    "Human challenge sent to {}: \
                     id={}, status={:?}",
                    username, challenge.id, challenge.status
                );

                if matches!(challenge.status, LichessChallengeStatus::Created) {
                    let db = graph_db.lock().await;

                    graph_mark_human_challenged(&db, &username)?;
                } else {
                    pending_challenges.remove(&username);
                }
            }

            Err(error) => {
                eprintln!("Failed to challenge human {}: {}", username, error);

                pending_challenges.remove(&username);
            }
        }

        if challenge_count < MAX_HUMAN_CHALLENGES_PER_SCAN {
            sleep(Duration::from_millis(HUMAN_CHALLENGE_SEND_DELAY_MS)).await;
        }
    }

    println!(
        "Online human scan complete: \
         {} eligible humans found, \
         {} challenge attempts made, \
         {} outgoing human challenges currently pending.",
        eligible_count,
        challenge_count,
        pending_challenges.len()
    );

    Ok(())
}

pub(super) async fn current_perfs(
    client: &LichessClient,
) -> Result<Option<LichessPerfs>, Box<dyn std::error::Error + Send + Sync>> {
    let me = client.account().profile().await?;

    Ok(me.user.perfs)
}

pub(super) async fn fetch_online_player_usernames(
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let response = reqwest::Client::new()
        .get(ONLINE_HUMAN_PAGE_URL)
        .header(
            reqwest::header::USER_AGENT,
            "TrueZero Lichess matchmaking",
        )
        .send()
        .await?
        .error_for_status()?;

    let html = response.text().await?;

    let mut usernames = Vec::new();

    let mut seen = HashSet::new();

    let marker = "href=\"/@/";

    let mut search_start = 0usize;

    while let Some(relative_start) = html[search_start..].find(marker) {
        let start = search_start + relative_start + marker.len();

        let remainder = &html[start..];

        let end = match remainder.find('"') {
            Some(end) => end,

            None => break,
        };

        let username = &remainder[..end];

        if !username.is_empty()
            && !username.contains('/')
            && seen.insert(username.to_ascii_lowercase())
        {
            usernames.push(username.to_string());
        }

        search_start = start + end + 1;

        if usernames.len() >= MAX_ONLINE_HUMAN_CANDIDATES {
            break;
        }
    }

    Ok(usernames)
}

pub(super) fn challenge_color_for_bot(username: &str) -> LichessChallengeColor {
    let hash = username.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u64)
    });

    if hash % 2 == 0 {
        LichessChallengeColor::White
    } else {
        LichessChallengeColor::Black
    }
}
