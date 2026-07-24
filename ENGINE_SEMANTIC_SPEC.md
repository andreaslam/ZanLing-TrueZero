# ZanLing-TrueZero Engine Semantic Specification

**Status:** authoritative semantic contract for compatible engines, replay producers, replay consumers, networks, MCTS implementations, and training systems.

This document defines meaning, coordinate systems, transformations, and required invariants. It deliberately does **not** prescribe program structure, threading, storage implementation, search optimisation, or model architecture beyond observable input/output semantics.

The terms **White** and **Black** refer to physical chess colours. **Us** and **opponent** always mean the side to move at the position currently being represented.

---

# Board representation

## Physical board

A physical board is an ordinary chess position in standard chess coordinates. Its squares retain their normal names (`a1` through `h8`) and it records which physical colour is to move.

Physical moves are applied only in this physical coordinate system.

## Canonical network/replay board

Every replay input and every network input represents the position in a player-to-move canonical frame.

For chess, canonicalisation has two independent parts:

1. **Piece ownership order:** the first six piece planes represent **us**; the next six represent the opponent.
2. **Square orientation:**
   - If White is to move, squares retain their physical locations.
   - If Black is to move, the board planes are rotated by $180^\circ$.

For a flattened square index $i = 8r + f$, Black canonicalisation maps:

$$
i \mapsto 63-i
$$

Equivalently:

$$
(r,f) \mapsto (7-r,7-f).
$$

The en-passant plane uses the same square transformation. Castling-right fields are ordered **us first, opponent second**.

The canonical chess input has 21 planes of size $8\times8$:

- 8 scalar channels, broadcast across the board;
- 12 piece planes, ordered as us then opponent, each ordered by the shared piece-type order;
- 1 en-passant plane.

The scalar channels retain enough absolute information for a canonical tensor to be self-describing, including separate White-to-move and Black-to-move indicators.

### Board representation invariants

1. **Canonical ownership invariant:** the first piece group always means the player to move, never “White”.

   **Example:** In a Black-to-move position, black knights are in the “us knight” plane and white knights are in the opponent-knight plane.

2. **Black rotation invariant:** every Black-to-move board square in tensor/replay board planes is rotated by $180^\circ$.

   **Example:** A black piece physically on `a7` is represented at canonical square `h2`.

3. **White identity invariant:** White-to-move board planes preserve physical square locations.

   **Example:** A white piece physically on `c3` is represented at `c3`.

4. **En-passant consistency invariant:** the en-passant square uses the same orientation as pieces.

   **Example:** If an en-passant target is physically `a6` for Black to move, it is represented at `h3`.

5. **No double-canonicalisation invariant:** a replay consumer must copy the stored board planes into the network tensor without applying another colour swap, mirror, rotation, or value sign change.

   **Example:** A Black-to-move replay tensor already has black as “us” and rotated squares; rotating it again is invalid.

---

# Player representation

## Physical colour

Physical colour is White or Black and is part of the chess state. It determines legal moves and the conventional absolute outcome.

## Relative player

At every position, `us` is the physical side to move and `opponent` is the other colour. All externally visible value targets are relative to `us`.

## Absolute value representation

An implementation may use a White-absolute representation internally, provided it converts correctly at every boundary where a player-relative value is required.

A White-absolute scalar value means:

$$
+1 = \text{White win},\qquad 0 = \text{draw},\qquad -1 = \text{Black win}.
$$

A White-absolute WDL vector is ordered:

$$
[\Pr(\text{White win}),\Pr(\text{draw}),\Pr(\text{Black win})].
$$

## Conversion between absolute and relative values

For White to move, absolute and relative values are identical.

For Black to move:

$$
v_{\mathrm{POV}}=-v_{\mathrm{abs}}
$$

and:

$$
[w,d,l]_{\mathrm{POV}}=[l,d,w]_{\mathrm{abs}}.
$$

Moves-left is not a player value and is unchanged by this conversion.

### Player representation invariants

1. **Relative-target invariant:** all replay `final_*`, `zero_*`, and `net_*` value/WDL targets are from the stored side-to-move perspective.

   **Example:** If White ultimately wins, a White-to-move record has `final_v = +1` and final WDL `[1,0,0]`; a Black-to-move record has `final_v = -1` and final WDL `[0,0,1]`.

2. **Perspective-switch invariant:** changing the side to move negates scalar value and swaps WDL win/loss entries.

   **Example:** White-absolute value `+0.4` becomes Black-relative value `-0.4`; White-absolute WDL `[0.7,0.2,0.1]` becomes Black-relative `[0.1,0.2,0.7]`.

3. **Moves-left invariance:** changing perspective never negates, swaps, mirrors, or otherwise changes moves-left.

   **Example:** A position estimated to be 12 plies from completion remains 12 moves-left from either player’s perspective.

---

# Replay semantics

## Dataset identity

A replay dataset is a sequence of complete simulations. Each simulation contains one non-final record for every played move and exactly one appended final record.

For a game with $L$ played plies:

- non-final records have `pos_index` $0,1,\ldots,L-1$;
- the final record has `pos_index = L`;
- every record has `game_length = L`.

## Record fields

A record contains:

- game identity and position identity;
- board tensor data in the canonical frame;
- a sparse legal-move policy;
- a played-move index;
- final outcome targets (`final_*`, also called $z$);
- search targets (`zero_*`, also called $q$);
- network targets (`net_*`, also called $v$);
- metadata and terminal flags.

All scalar fields are `float32`, including booleans and count-like metadata. Sparse policy indices are signed 32-bit integers. Boolean planes are packed little-bit-first.

## Final record

The appended final record represents the board after the last physical move, or the board at truncation.

It has:

- `is_final_position = true`;
- `pos_index = game_length`;
- `available_mv_count = 0`;
- empty policy index and value arrays;
- `played_mv = -1`;
- `zero_visits = 0`;
- unavailable (`NaN`) `zero_*` and `net_*` targets;
- player-relative `final_*` targets;
- `final_moves_left = 0`.

A final record can represent either a true terminal state or a move-limit truncation. `is_terminal` distinguishes them. `hit_move_limit` is true exactly for the latter.

A truncated game has no winner. Its final target is therefore a draw target unless an explicitly different terminal-result policy is introduced and documented.

## Replay sampling

Ordinary training samples non-final records by default. If a training mode requests the final input for a non-final record, it must select the appended record at:

$$
\text{final record index} = \text{current record index} + \texttt{final\_moves\_left}.
$$

### Replay semantics invariants

1. **Pre-move record invariant:** every non-final record represents the board before its `played_mv` is applied.

   **Example:** If a record’s played move is `e2e4`, the record board contains a white pawn on `e2`, not `e4`.

2. **Position-count invariant:** a game of $L$ plies has $L+1$ replay records when final records are enabled.

   **Example:** A 73-ply game has positions `0..73`, totalling 74 records.

3. **Final-index invariant:** `is_final_position` is true if and only if `pos_index == game_length`.

   **Example:** In a 40-ply game, position 39 is non-final and position 40 is final.

4. **Final moves-left invariant:**

   $$
   \texttt{final\_moves\_left} = \texttt{game\_length} - \texttt{pos\_index}.
   $$

   **Example:** In a 60-ply game, record 17 has `final_moves_left = 43`; the appended final record has `final_moves_left = 0`.

5. **Final-record policy invariant:** a final record has no available move and no policy mass.

   **Example:** `available_mv_count = 0`, `played_mv = -1`, and both sparse policy arrays have length zero.

6. **Terminal-status invariant:** `is_terminal` means the final board is terminal under the game rules; `hit_move_limit` means the game ended only because of the configured length bound.

   **Example:** A checkmate record has `is_terminal = true`, `hit_move_limit = false`; a forced length cutoff has `is_terminal = false`, `hit_move_limit = true`.

7. **Exact-decode invariant:** a consumer must consume precisely the bytes belonging to one record, with no trailing or missing bytes.

   **Example:** A declared 20-move policy consumes exactly 20 indices and exactly 20 values.

8. **Batch-padding invariant:** padding exists only in batch tensors, not in replay records. Padding is `(policy_index=0, policy_value=-1)`.

   **Example:** A record with 3 legal moves in a batch padded to 5 has two trailing `(0,-1)` pairs.

---

# Value semantics

Every value family comprises a scalar value, WDL probabilities, and moves-left:

$$
(v, w, d, l, m).
$$

WDL order is always `[win, draw, loss]` from the stored position’s side-to-move perspective.

## Final values: $z$

`final_*` is the exact eventual game result from the current player’s perspective.

- `final_v` is exactly one of $-1$, $0$, $+1$.
- `final_wdl` is one-hot.
- `final_moves_left` is factual remaining ply count, not a prediction.

## Search values: $q$

`zero_*` is the MCTS-improved root estimate from the current player’s perspective.

- `zero_v` is the mean root search value.
- `zero_wdl` is the corresponding search WDL estimate.
- `zero_moves_left` is the search estimate of remaining plies.

## Network values: $v$

`net_*` is the raw network evaluation of the position, from the current player’s perspective, before search improvement.

## Missing values

`zero_*` and `net_*` are unavailable for final records and must be represented by all-`NaN` value/WDL/moves-left targets. Consumers must not train ordinary value or policy losses from unavailable targets.

### Value semantics invariants

1. **Final outcome invariant:** final scalar value and final WDL agree.

   $$
   \texttt{final\_v}=\texttt{final\_wdl}_w-\texttt{final\_wdl}_l.
   $$

   **Example:** `[0,1,0]` implies `final_v = 0`; `[0,0,1]` implies `final_v = -1`.

2. **WDL-normalisation invariant:** every available WDL vector sums to one.

   $$
   w+d+l=1.
   $$

   **Example:** `[0.55,0.30,0.15]` is valid; `[0.55,0.30,0.25]` is invalid.

3. **Final exactness invariant:** `final_*` reflects the realised outcome, not a network or search prediction.

   **Example:** If Black checkmates White, every earlier White-to-move record has final target loss even if the network previously predicted a White win.

4. **Search-mean invariant:** `zero_v`, `zero_wdl`, and `zero_moves_left` are averages of search outcomes, never unnormalised backup totals.

   **Example:** Two root evaluations $+1$ and $-1$ yield `zero_v = 0`, not 2 or -2.

5. **Final availability invariant:** final values are available at every record, including the appended final record; search and raw-network targets are unavailable on the appended final record.

   **Example:** A final record may have `final_v=-1` but `zero_v=NaN` and `net_v=NaN`.

---

# Policy semantics

## Policy space

Chess uses one fixed flat policy space of size 1880. A policy index denotes a move in this shared space, not a board-relative “from/to” pair stored separately per position.

## Sparse legal policy

A replay policy contains only legal actions:

$$
\{(a_j,\pi_j)\}_{j=1}^{K}
$$

where $K$ is `available_mv_count`, every $a_j$ is legal in the represented position, and $\pi_j$ is the normalised MCTS root-visit probability for that action.

The order of the pairs is the game’s legal-move-generator order. Values remain in that order whenever indices are transformed.

## Chess move coordinates

Policy indices use **White-absolute move-index coordinates** regardless of side to move.

- For White to move, a physical move is indexed directly.
- For Black to move, the physical move is rank-flipped before looking up its index:

$$
(r,f)\mapsto(7-r,f).
$$

This is a rank mirror, not the 180-degree board-plane rotation. Board orientation and policy-index coordinates are distinct semantic spaces and must not be conflated.

The stored `played_mv` uses exactly the same mapping as policy indices.

When an engine uses a policy index to choose a physical Black move, it must apply the inverse rank flip before making the move on the physical board.

### Policy semantics invariants

1. **Legal-support invariant:** the sparse policy contains exactly the legal moves of the represented non-final position.

   **Example:** If a position has 31 legal moves, it has 31 policy indices and 31 policy values.

2. **Policy-normalisation invariant:** non-empty policy mass sums to one.

   $$
   \sum_{j=1}^{K}\pi_j=1.
   $$

   **Example:** visit counts `[30,10,10]` become policy values `[0.6,0.2,0.2]`.

3. **Policy-index-range invariant:** each chess policy index satisfies:

   $$
   0\le a_j<1880.
   $$

   **Example:** index 1879 is valid; index 1880 is invalid.

4. **Played-move invariant:** every non-final `played_mv` is one of that record’s policy indices and is legal on that record’s physical board after inverse coordinate conversion where needed.

   **Example:** If Black physically plays `e7e5`, replay stores the index for rank-flipped move `e2e4`; that index must also occur in the policy-index list.

5. **Black-policy transform invariant:** Black policy indexing uses rank flip only.

   **Example:** Black physical `a7a5` maps to policy move `a2a4`, not `h2h4`.

6. **Policy/board coherence invariant:** the canonical board tensor, white-absolute policy index, and encoded played move describe the same legal action.

   **Example:** For Black to move, a rotated board tensor and the policy index for rank-flipped `e7e5` must identify the same underlying physical move.

---

# MCTS semantics

MCTS searches the physical game tree. Each tree node represents one physical board position and its physical side to move. Board canonicalisation is performed only at network-evaluation boundaries.

## Priors and expansion

For a non-terminal leaf, the network supplies a scalar/WDL/moves-left evaluation and a dense 1880-logit policy. The search extracts logits for legal moves, applying the chess policy-coordinate mapping, and normalises across legal moves to obtain priors.

At the root during self-play, priors may be temperature-adjusted and mixed with Dirichlet noise before selection. This changes exploration only; it does not change the semantic meaning of stored visit policy.

## Selection

Selection uses a PUCT score:

$$
\operatorname{PUCT}(s,a)=Q(s,a)+U(s,a)+M(s,a),
$$

where:

- $Q$ is the mean action value from the parent player’s perspective;
- $U$ is the prior-based exploration term;
- $M$ is an optional moves-left preference term.

An unvisited action uses the configured first-play urgency value instead of an empirical $Q$.

## Backup

A search implementation may accumulate values in White-absolute form internally. If it does, it adds one absolute leaf result to each ancestor and converts to the selecting parent’s POV when computing $Q$.

Equivalently, a purely POV implementation may negate scalar values and swap WDL win/loss once across each player transition. Both representations are valid only if they produce the same parent-relative $Q$ and player-relative replay outputs.

Moves-left never changes sign. A leaf estimate of $m$ remaining plies contributes:

$$
m+d
$$

to an ancestor at distance $d$ plies from that leaf.

## Root outputs

A completed search returns:

- a chosen physical move, normally the root child with maximum visit count;
- root visit count;
- root visit policy, normalised from child visits;
- mean root search evaluation $q$;
- root raw network evaluation $v$.

A search must perform enough simulations for at least one root child to receive a visit before emitting a visit policy.

### MCTS semantics invariants

1. **Parent-$Q$ invariant:** $Q(s,a)$ is evaluated from the perspective of the player who chooses $a$ at $s$, not the child’s side to move.

   **Example:** If White chooses an action leading to a position favourable to White, the action’s White-parent $Q$ is positive even though the child is Black to move.

2. **Backup-perspective invariant:** an internal absolute backup needs no sign change per edge; an internal POV backup needs exactly one value negation and WDL win/loss swap per edge. Mixing these schemes is invalid.

   **Example:** A White-absolute terminal value `+1` remains `+1` at all ancestors in an absolute ledger; it is read as `-1` only when queried from Black POV.

3. **Terminal-leaf invariant:** terminal moves-left is zero at the terminal node.

   **Example:** Checkmate at the selected leaf contributes 0 moves-left to that leaf, 1 to its parent, and 2 to its grandparent.

4. **Root-policy invariant:** stored search policy is root-child visit count divided by total root-child visits.

   **Example:** child visits `[9,3,0]` produce `[0.75,0.25,0.0]`.

5. **Root-value invariant:** replay `zero_*` is a mean root search evaluation, not a sum over simulations.

   **Example:** 100 identical root values of `0.2` produce `zero_v=0.2`, not `20`.

6. **Search-output POV invariant:** before writing replay, root $q$ and root $v$ are converted to the root side-to-move POV.

   **Example:** a White-absolute root search value of `+0.35` becomes `zero_v=-0.35` when Black is to move.

---

# Network semantics

## Input

The network input is the 21-plane canonical chess tensor described in [Board representation](#board-representation). It always describes the side to move as us.

## Output

The network outputs:

1. one raw scalar value logit;
2. three WDL logits ordered `[win, draw, loss]` from the input position’s side-to-move perspective;
3. one raw moves-left output;
4. 1880 policy logits in White-absolute chess move-index space.

Output interpretation is:

$$
v=\tanh(s_0),
$$

$$
[w,d,l]=\operatorname{softmax}(s_{1:4}),
$$

$$
m=\max(0,s_4).
$$

The policy logits become a legal policy only after legal moves are selected using the fixed coordinate mapping and softmax is taken over the intended action set.

### Network semantics invariants

1. **Value-head range invariant:** interpreted scalar value lies in $[-1,1]$.

   **Example:** a raw scalar logit of 0 gives value 0; a large positive logit approaches $+1$.

2. **WDL-order invariant:** output WDL classes are `[current-player win, draw, current-player loss]`.

   **Example:** a network assigning 90% to its first WDL class predicts a 90% chance that the input side to move wins.

3. **Moves-left non-negativity invariant:** interpreted moves-left is non-negative.

   **Example:** a raw moves-left output of `-3` is interpreted as 0.

4. **Policy-space invariant:** output coordinate $k$ always refers to fixed chess move index $k$, independent of the physical side to move.

   **Example:** index meaning does not change because an input board is Black to move.

5. **Legal-policy invariant:** a network policy used by MCTS or masked policy training is normalised only after restricting to the relevant legal replay/move indices.

   **Example:** illegal move logits do not receive probability in a legal-only MCTS prior.

---

# Training semantics

## Replay-to-tensor transformation

Training decodes replay records exactly, broadcasts the 8 input scalars spatially, and concatenates them before the 13 stored Boolean planes. It must not rotate, mirror, negate, recolour, or otherwise reinterpret a record based on side to move.

Optional symmetry augmentation transforms board Boolean planes, policy indices, and `played_mv` together. It leaves values, WDL, moves-left, and policy probabilities unchanged. It may be used only when input scalar features are invariant under the chosen symmetry.

## Scalar targets

A training configuration chooses a mixing coefficient $\lambda\in[0,1]$:

$$
y_v=\lambda z_v+(1-\lambda)q_v,
$$

$$
y_{\mathrm{wdl}}=\lambda z_{\mathrm{wdl}}+(1-\lambda)q_{\mathrm{wdl}}.
$$

- $\lambda=1$ means pure final-outcome targets.
- $\lambda=0$ means pure search targets.

Moves-left training always uses factual `final_moves_left`, not `zero_moves_left` or `net_moves_left`.

## Losses

For a batch of non-final positions:

$$
L_v=\operatorname{MSE}(\tanh(s_0),y_v),
$$

$$
L_{\mathrm{wdl}}=-\sum_c y_{\mathrm{wdl},c}\log\operatorname{softmax}(s_{1:4})_c,
$$

$$
L_m=\operatorname{Huber}(\max(0,s_4),\texttt{final\_moves\_left}),
$$

and the policy loss is cross-entropy against the sparse replay visit policy.

The weighted total loss is:

$$
L=w_vL_v+w_{\mathrm{wdl}}L_{\mathrm{wdl}}+w_mL_m+w_\pi L_\pi,
$$

plus any explicitly configured representation-similarity loss for unrolled dynamics training.

Gradients are computed from this total loss, optionally clipped, then applied by the optimiser.

## Policy masking

When legal-move masking is enabled, the policy softmax is over replay-listed legal indices only. When it is disabled, the softmax is over the whole 1880-space and replay-listed moves have target mass while all other moves implicitly have target zero.

Padding entries with value $-1$ are excluded from all policy-loss calculations.

### Training semantics invariants

1. **No-colour-correction invariant:** training performs no side-to-move sign flip or WDL swap.

   **Example:** a Black-to-move replay target `final_v=-1` is passed to MSE as `-1`, not changed to `+1`.

2. **Scalar-broadcast invariant:** each scalar input channel is constant across all 64 squares of that sample.

   **Example:** a halfmove-clock scalar of 17 produces an $8\times8$ plane filled with 17.

3. **Target-mix invariant:** scalar and WDL final/search mixing uses the same coefficient $\lambda$.

   **Example:** with $\lambda=0.25$, a final value $1$ and search value $0.2$ give target $0.4$.

4. **Moves-left-target invariant:** the moves-left loss target is always factual `final_moves_left`.

   **Example:** if search predicts 15 plies but the game ends in 7, the training target is 7.

5. **Padding-mask invariant:** a padded policy value of $-1$ contributes zero loss and does not expose policy index 0 as a target move.

   **Example:** padding pair `(0,-1)` is ignored even though policy logit 0 exists.

6. **Policy-mass validation invariant:** every non-empty replay policy target has total mass one within tolerance; an empty final-policy target has mass zero.

   **Example:** `[0.5,0.3,0.2]` is valid; `[0.6,0.6]` is invalid.

7. **Terminal-metric invariant:** terminal-only logged losses average only records marked terminal.

   **Example:** a batch with terminal losses 2 and 4 reports terminal mean 3, regardless of non-terminal losses.

8. **Final-input lookup invariant:** when an unrolled/sample mode requests a sample’s final board input, it selects the appended final record exactly `final_moves_left` records ahead.

   **Example:** a record with `final_moves_left=5` uses record index `current+5`, not `current+4`.

---

# Terminal reward semantics

A terminal state’s reward is defined relative to the player to move **at that terminal state**.

- If the state is checkmate, the side to move has been mated and receives loss.
- The player who made the immediately preceding mating move receives win from the parent position’s perspective.
- A draw is neutral for both players.

Thus, a terminal value viewed at the terminal node is:

$$
\begin{cases}
-1 & \text{if side to move is checkmated},\\
0 & \text{if drawn}.
\end{cases}
$$

When viewed one ply earlier from the mating player’s parent position, the same outcome is $+1$.

For a move-limit cutoff, there is no winner; the final target is draw/neutral and the record is marked `hit_move_limit=true`, `is_terminal=false`.

### Terminal reward invariants

1. **Mated-side invariant:** a checkmated terminal node is a loss for its side to move.

   **Example:** White delivers mate; the terminal board has Black to move and its player-relative terminal value is `-1`.

2. **Mating-parent invariant:** the parent action that delivers checkmate has value `+1` from the parent player’s perspective.

   **Example:** The White-to-move parent before `Qh7#` evaluates that action as win.

3. **Draw-neutrality invariant:** any terminal draw has scalar value 0 and WDL `[0,1,0]` from either perspective.

   **Example:** Stalemate yields `final_v=0` for both the final Black-to-move state and earlier White-to-move records.

4. **Truncation-neutrality invariant:** a length-limited non-terminal game has no winner and uses a neutral final target.

   **Example:** A game stopped after 200 plies without terminal status has final WDL `[0,1,0]`, `is_terminal=false`, and `hit_move_limit=true`.

---

# Perspective rules

The following table is mandatory whenever a quantity crosses between physical, canonical, absolute, and player-relative representations.

| Quantity | White to move | Black to move | On player switch |
|---|---|---|---|
| Physical board | standard coordinates | standard coordinates | apply physical move only |
| Canonical board planes | identity | rotate $180^\circ$ | re-canonicalise from physical board |
| Piece-plane ownership | White then Black | Black then White | swap us/opponent groups |
| Scalar/WDL value, absolute → POV | unchanged | negate scalar, swap W/L | negate and swap W/L |
| Moves-left | unchanged | unchanged | unchanged |
| Policy index, physical → white-absolute | direct | rank-flip move | use inverse mapping when applying move |
| Played move, physical → replay | direct | rank-flip move | same policy mapping |
| Replay tensor → training tensor | copy | copy | never transform again |

### Perspective-rule invariants

1. **Board/value separation invariant:** board rotation and value sign conversion are separate operations.

   **Example:** Canonicalising a Black board rotates its planes; converting a White-absolute value to Black POV negates the scalar. Neither operation substitutes for the other.

2. **Board/policy separation invariant:** Black board planes use a $180^\circ$ rotation, while Black policy moves use a rank flip.

   **Example:** physical Black square `a7` appears at canonical `h2`, but physical Black move `a7a5` is indexed as `a2a4`.

3. **Replay-boundary invariant:** replay is always player-relative for values and fixed white-absolute for chess move indices.

   **Example:** A Black-to-move record can store `zero_v=+0.3` for a Black-favourable position while its `played_mv` is an index for rank-flipped White-coordinate move notation.

4. **Training-boundary invariant:** training consumes replay semantics directly; it never reconstructs physical colour to repair targets.

   **Example:** no training branch should say “if Black, negate target”.

---

# Invariants

The following global invariants must hold simultaneously. A system is compatible only when all apply.

1. **Single semantic frame per external record:** board input is current-player canonical, values are current-player POV, and chess policy/played-move indices use the fixed white-absolute 1880-space.

   **Example:** A Black-to-move record has black-first rotated board planes, a Black-relative value, and rank-flipped move indices.

2. **Outcome consistency:** final scalar, final WDL, winner, and player to move all describe the same outcome.

   **Example:** If Black wins, a White-to-move record has `final_v=-1` and `[w,d,l]=[0,0,1]`.

3. **Search consistency:** `zero_*` corresponds to the same represented position and same current-player frame as the board and policy.

   **Example:** search values from a White-to-move root are not reused unchanged for the following Black-to-move record.

4. **Network consistency:** `net_*` uses the same current-player frame as the input presented to the network.

   **Example:** a network input with Black as us must have Black-relative raw value targets.

5. **Action consistency:** every sparse policy index, policy probability, and played-move index refers to the same action under the prescribed coordinate mapping.

   **Example:** reordering legal moves requires reordering policy values identically; changing only indices is invalid.

6. **Probability consistency:** WDL vectors and non-empty policies are normalised probability distributions.

   **Example:** WDL `[0.2,0.5,0.3]` and policy `[0.7,0.3]` are valid; sums other than one are invalid.

7. **Unavailable-data consistency:** unavailable final-record search/network values are all-`NaN`; unavailable batch-policy slots are all `$-1$` values.

   **Example:** a final record cannot use `zero_v=0` to mean unavailable; it must use `NaN`.

8. **Coordinate inversion consistency:** any move transformed for policy lookup must be inversely transformed before being applied to the physical board.

   **Example:** a Black indexed move `e2e4` is converted back to physical `e7e5` before game-state transition.

9. **Simulation-boundary consistency:** records from different games never form one unrolled transition chain.

   **Example:** record 52 of game A must not use record 53 of game B as its successor, even if their global file offsets are adjacent.

10. **Final-record lookup consistency:** a requested final input is the appended final board of the same simulation, not the last non-final board.

   **Example:** for `final_moves_left=1`, final input is the next record, not the current record.

11. **Transformation-pair consistency:** every allowed geometric symmetry maps board Boolean planes, policy indices, and played moves together; values/WDL/moves-left remain untouched.

   **Example:** applying a horizontal reflection to the board without reflecting its move indices creates an invalid training sample.

12. **No implicit absolute target invariant:** no replay or training target may silently change meaning based on physical colour.

   **Example:** `+1` always means “the side to move wins” in replay value fields, never “White wins”.
