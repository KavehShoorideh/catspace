# repo_structure.md — where things go, and why

This is the instruction doc for the layout adopted on 2026-08-03. If you are adding code,
data, an experiment or a doc, the answer to "where does this go" is here. `AGENTS.md`
points at this file and states the rules that are actually enforceable.

Anything dated before 2026-08-03 (JOURNAL.md, MILESTONES.md, `research/docs/archive/`,
`research/docs/writing/`, `research/weekly_report/`) describes the OLD layout and was
deliberately not rewritten — see [Historical documents](#historical-documents).

---

## The one-paragraph version

`catspace/` is the wrapper package. It owns the engine backbone and the plugin Protocols,
and it wires together four research components — **encoder, planner, search, memory** —
each of which holds several competing **approaches**. A named **end-to-end approach**
(`catspace/approaches/<name>/`) says which approach fills each slot. `catspace/deployment/`
consumes an end-to-end approach and never reaches into a component itself. Everything is
`import catspace.*`; there are no other top-level packages.

---

## The tree

```
/                                repo root
  repo_structure.md              this file
  AGENTS.md                      the short, enforceable version
  pyproject.toml                 packaging, entry points, pytest config
  config.json                    fastchess tournament config
  scripts/                       symlinks to the entry points a newcomer needs
  tests/                         cross-component and integration tests
  contrib/                       third-party code, unchanged
  data/                          generated datasets, DVC-tracked
  assets/                        THIRD-PARTY only: tablebases, engines, corpora, weights
  artifacts/                     experiment records + curated results
  mlruns/, mlflow.db             the one tracking store
  README.md JOURNAL.md MILESTONES.md human-written-alt-arch.md    (historical, banner-noted)

  catspace/                      THE WRAPPER PACKAGE
    interfaces.py                the Protocols an implementation must satisfy
    engine_core.py               LayeredEngine, the composer
    registry.py                  resolve "component:approach" -> module
    orchestrator.py introspection.py watchlist.py priors.py fields.py values.py
    io/paths.py                  THE PATH REGISTRY (see Data)
    incubator/                   explicit holding pen: work with no home yet
      analyst/discovery/         the analyst's discovery code; the wake/sleep loop (plan 78.14-78.17):
        wake.py sleep.py         self-play logging (play_self(..., starts=None): game g starts from starts[g % n], a position string the kernel accepts, None = the game's own; `game` rows carry `start`; read_starts(path); CLI --starts); sleep (surprises, cards, lemmas; rank sources, one shape `rank(variant, moves, cap=None, start=None)` (start passed only when not None): parse_multipv (+ parse_multipv_lines: the whole principal line per rank), disagreement, break_ties, plan_tools (one source or a list: the human source first when coarse), source_of (mcts | human[:path] | gnn[:path]), UciTeacher (gap = 100, start-aware; `grade(score)` (mate exact, else score // gap); `lines(variant, moves, cap=None, start=None) -> {move: principal line}` from the same cached analysis as `rank`; CLI --teacher-ms, repeatable --teacher-option NAME=VALUE); Settings.regret; plan 78.23: Settings.judge_by ("store" | "source") + judge_positions, CLI --judge-by / --judge-positions; read_log keeps `start`); plan 78.24: Settings.vocabulary (a NAMED lemma vocabulary: spec/vocabulary, known lemmas, meta.vocabulary and publish all use it), UciTeacher.mate_bucket (plies, default 1: a mate's distance floored to a multiple), CLI --vocabulary NAME, repeatable --gate NAME=VALUE (dev and held-out gates) / --held-gate NAME=VALUE (held-out only), --teacher-mate-bucket N; plan 78.25: play_self(..., reason=False) and CLI --reason pass the reasoning chain through to solve_batch (as --ties does); plan 78.26: play_self(..., wdl=False, wdl_table=None) and CLI --wdl / --wdl-table PATH pass the win / draw / loss path (and its results table) through to solve_batch, as --reason does; tests/test_plan78_22.py, tests/test_plan78_24.py, tests/test_plan78_25_reason.py
        detectors.py             plan 78.22: surprise detectors as modules (Context, Hindsight (no oracle), Disagreement(source), Regret(source) (the move played against the source's ranking, by the source's own `gap`, else top-k order), confirm (shared deeper-card confirmation: confirmed / tied (OPEN) / unconfirmed / cut), run, detector_of(name, source=None), REGISTRY); sleep calls them; Context.continuations (plan 78.23); `ask` takes cap; curriculum ChessCodec.kinds (optional codec method); tests/test_plan78_22.py
        judges.py                plan 78.23: a rank source as the JUDGE of the cards (optional source methods `grade(score) -> int`, `lines(...)`; default_grade, grader, describe, revalue(cards, positions, source, variant, indices=None, *, cap=None, say=None, context=None) -> (CardSet via CardSet.revalued, report incl. the optimal share), regret_seeds (regret confirmed directly on the revalued card; rep["regret"]["confirmed_rows"] and meta.why = the source's lines after the played / best move), consequence(variant, start, moves, line, codec=None) (replay a line: plies, outcome, units, first change), say_consequence; describe also reports mate_bucket; plan 78.24: Cached(source, path) = any rank source remembered in a JSON file keyed (variant, start, moves, cap), so the sleep, the audit and a re-run read one verdict per position; the wrapped source's own attributes show through); tests/test_plan78_23.py, tests/test_plan78_24_loop.py
        curriculum.py            plan 78.22, generic: the material curriculum (PositionCodec protocol, ChessCodec (the one place chess appears; optional `kinds`), CODECS by game name, required(game) found by test, starts(game, size, limit=, seed=), survey, CLI -> a --starts file under paths.derived); tests/test_plan78_22.py
        apply_audit.py          plan 78.24 B: the audit of application (generic): render(lemma) / mined(game) (a lemma read as IF body THEN best move removes/adds, with stats, judge, meta.why); audit(game, log, *, snapshot=None, variant=None, source=None, vocabulary=None, limit=None, say=print) -> per decision where a lemma fired: the moves it proposed (recovered with the ENGINE's matcher: CardSet.revalued one-hot per move + CardSet.eval, since the log row holds only the `fired` fact and the count), played, followed, optimal by the rank source; per-lemma and overall summaries as OBS lines; CLI --log --snapshot --teacher --out; tests/test_plan78_24_loop.py
        initialize.py           plan 78.24 C: the loop for one size on a SANDBOX store (GS.LEARNED_ROOT / SNAPSHOT_ROOT swapped in and restored; vocabulary copied from the real rules/learned, never written back): run_size(game, size, *, source, store_root=None, judge=None, games_per_start=4, ..., sleep_settings=None, vocabulary=None, tag=None, wdl=False, cycles=1, wdl_judge=None, wdl_settings=None) = starts -> wake a -> sleep (source is the judge) -> wake b -> audit -> before/after compare (a mechanism check); settings_pairs / settings_from (Settings fields typed from the dataclass); CLI --size --teacher --store-root --set NAME=VALUE --vocabulary --gate NAME=VALUE --held-gate NAME=VALUE --teacher-mate-bucket N --teacher-cache FILE; plan 78.27: wdl=True plays every wake with the estimate and the sandbox table `<root>/learned/<game>/wdl_table.json`, prints `OBS init gaps`, and after the lemma sleep runs wdl_sleep.sleep_wdl (wdl_judge: a judge of readings, wdl_settings: its keywords) carrying the table forward; cycles=k repeats sleep then wake (report gains `cycles`; wdl=False, cycles=1 is as before); CLI --wdl --cycles K --wdl-judge-depth, --wdl-judge-cache, --wdl-set NAME=VALUE D (the --teacher engine also judges); tests/test_plan78_24_loop.py, tests/test_plan78_27_wdl_sleep.py
        diagnose.py              game-free diagnostics over a wake log (ties, surprises by kind, cost, depth, gap share); tests/test_plan78_17.py
        rank_bench.py            plan 78.18: a rank source's agreement with a reference on logged positions, and a match of two rank sources (positions carry their start); tests/test_plan78_18.py
        wdl_table.py             plan 78.26 THE WIN / DRAW / LOSS RESULTS TABLE: build(records, variant, min_exact=2, roles=None, provenance="own games", *, settled=None, deep_rows=None, prior_table=None) [plan 78.27: `settled` unfinished games read to soft results {start, moves, mover, wdl}, `deep_rows` {key: [w, d, l]} never dropped by min_exact, `prior_table` (dict or path) carried forward and summed in; stats gain settled {games, positions}, deep {rows, mass}, carried_rows; provenance names each own source], load(path), flip(mass) -> {format "wdl-table-1", game, provenance, rows {key: [mover wins, draws, mover losses]}, stats} from (start | None, moves, outcome per role 0..100 | None) records (every position along a finished game adds its result from the mover's view under each key of cs_engine.wdl_keys_along; an outcome off 0 / 50 / 100 is split like the rust `split`; unfinished games skipped and counted; `exact:` rows under `min_exact` games dropped), split, family, records_from_wake_logs(paths), write(table, path), default_path(game) (paths.derived), CLI `wdl_table <game> --log LOG [--log ...] [--out PATH] [--min-exact N] [--provenance TEXT]` printing OBS lines; never reads the human-games index; tests/test_plan78_26_wdl.py
        wdl_sleep.py             plan 78.27 THE SLEEP CYCLE OF THE ESTIMATE (game-free; the UCI judge class is the one chess-library place): gaps(log | [logs], pick_margin=0.02, surprise=0.5) -> {decisions, counts {no_table, coarse_table, unknown, thin_pick, no_lemma, outcome_surprise}, modes, table, mass_median, with_lemma, picked, worklist [{log, run, game, ply, start, moves, kinds}]} (reading_of(row) from the row's structured facts / steps; decision_readings(logs)); WdlJudge (training_only = True; wdl(variant, moves, start=None) -> (win, draw, loss) of the side to move; close()), UciWdlJudge(path, depth=12, cache=None, options=None) (python-chess, UCI_ShowWDL, disk cache by position, save / close); ReadSetting(name, seconds, max_playout=8, rollout="careful"), default_menu(deep_seconds) (wake, long, far, end), read(setting, variant, start, moves, judge_kw, seed, table); choose_reading(positions, judge, menu, *, variant, judge_kw, sample, seed) -> {setting, reason, scores, sample} (no judge: long); settle(unfinished_games, setting, *, variant, ...) -> [{start, moves, mover, wdl, settled}]; deepen(worklist, setting, *, variant, max_positions, ...) -> {rows, touched, read, mass} (DEEP_EXACT_MASS 200.0 on the exact key, COARSE_VOTE 1.0 on coarser keys; spread over games by sleep.spread); calibrate(positions, judge, *, variant, table, seconds, judge_kw, ...) -> {positions, mae, corr, calls, table_hits, unknown}; sleep_wdl(game, logs, *, variant, prior_table, judge, judge_kw, out, deep_seconds, max_deep, sample, seed, max_settle, setting, say) -> report (gaps, choice, deep, stats, table, calibration held_out / touched x before / after), judge_kw_of_snapshot(path), unfinished_of(logs), default_out(game) (paths.derived); no outside number enters the table; CLI `wdl_sleep <game> --log LOG [--log ...] [--prior-table P] [--judge-engine UCI] [--judge-depth 12] [--judge-cache FILE] [--store SNAPSHOT] [--out P] [--deep-seconds S] [--max-deep N] [--sample N]`; tests/test_plan78_27_wdl_sleep.py
        playout_sleep.py         plan 78.28 PLAYOUTS TO A REAL RESULT, DIAGNOSED AT SLEEP (game-free: the game is the variant string, the outside engine a judge object with wdl(variant, moves, start=None); training only; nothing of the judge enters a snapshot or the wake): verdict_class((w, d, l)) -> "win" / "draw" / "loss" (largest share, ties to the lower), rank_of, flip, outcome_class(outcome, role), Verdicts(judge, variant) (of(start, moves), after(start, moves, role), seat(start, moves), calls; memoised); playout_games(positions, *, variant, store, spec, n_per_position=1, seed=1, rollout="rules", max_plies=120, worth=None, valuation="") -> [{start, prefix, moves, outcome, capped, named, rollout, seed}] (cs_engine.playout_game); match_share(games, judge, variant, *, verdicts=None) -> {games, finished, capped, matched, worse, better, share, calls}; turning(game, judge, variant, *, verdicts=None, max_scan=60) -> None | {ply, start, moves, played, before, after, start_class, result_class, role}; extra_positions_of(turnings, tag="playout") -> [{id, start, moves, played, group}] (the seeds' route: sleep.sleep(extra_positions=...)); sample_positions(logs, n, *, seed=1) (wdl_sleep.gaps worklist, else the logged decisions); check(problems, *, variant, store, spec, judge, seconds, wake_kw=None, threads=0, verdicts=None) -> [{move, keeps, before, after, proved, steps, error}] (the wake's own solve_batch call, ties=True, then the judge); load_problems(path) / save_problems(path, rows) / key_of(row) / merge_problems(rows, found, cycle) -> (rows, new) / record(rows, results, cycle, phase) / problem_counts(rows, cycle, new) -> {total, new, solved_before, solved_after, still_open, came_back} / say_counts(counts, say) (problem file format "playout-problems-1"); sleep_playouts(game, logs, *, variant, store, judge, teacher=None, spec=None, rollout="rules", positions=20, per_position=2, seed=1, max_plies=120, max_turning=20, max_scan=60, sleep_kw=None, say=print) -> {positions, games, match, turnings, extra_positions, [sleep]} printing `OBS playout_sleep ...`; CLI `playout_sleep <game> --log LOG --store SNAPSHOT --judge-engine UCI [--judge-depth 12] [--judge-cache FILE] [--vocabulary NAME] [--positions N] [--per-position N] [--rollout both|careful|rules] [--max-plies N] [--max-turning N] [--seed S]`; tests/test_plan78_28_playout_sleep.py
        targeted.py              plan 78.29 THE TARGETED MINER: one problem at a time, the simplest rule in the store's lemma format, drilled against counter-positions, admitted only when the problem's own check passes (game-free: the variant string, the store's fact vocabulary, a judge object with wdl(variant, moves, start=None) -- training only; no verdict or number of the judge enters a lemma or snapshot): Reading(pos, label, group, before, keep, .good, .bad); Reader(variant, spec, vocabulary, verdicts, *, depth=1, fuse, threads, worth, ours_first) (legal(p), read(p, label="", group="") -> Reading | None, cards(items)); Drill(reader, readings) (cards whose move values are the judge's keep (0) / lose (1); read(lemma) / read_many(lemmas) -> {fires, wrong, right, trivial, wrong_at, right_at, trivial_at, truncated}: WRONG = a move the rule names loses the verdict, RIGHT = it names moves and all keep, TRIVIAL = every move keeps); head_matches(head, removed, added), lift_head(head, keep=frozenset()), head_text(head), heads_for(card, keep, *, max_lits=2, limit=8, variety=True) -> [{head, matched, lits, consts, names}], body_candidates(facts, names, have), lemma_of(body, head, name), mine_body(drill, seed_ci, facts, head, names, *, max_rounds=8, say, tag) -> Mined(ok, lemma, reading, rounds, why) (the repair loop, `OBS targeted ... round`), drill_positions(reader, problem, reading, *, others, log_pool, seed=1, before_plies=3, playouts=3, playout_plies=6, neighbours=16, similar=24, cap=120, profile=None, exclude=None) -> [(position, label)], trial_snapshot(game, lemma, directory, vocabulary=None) -> path (the snapshot the store would publish with the lemma, same id), provenance(...) (meta: origin "targeted", mined_from "problem"; drill counts only), sandbox(root) (context: store paths into the sandbox, restored after), family_key(card, played), SETTINGS (min_right, min_fresh, verify_rounds, verify_cap, verify_similar, verify_neighbours, max_rounds, max_heads, drill_cap, similar, neighbours, seed, pool), solve_problems(game, problems_path | rows, *, variant=None, store_root=None, judge, teacher=None, logs=(), seconds=0.2, max_rounds=8, vocabulary=None, spec=None, wake_kw=None, settings=None, say=print) -> {total, attempted, solved, still_open, admitted: [{name, text, problem, solved_others, snapshot, ...}], rows, snapshot, already_kept} printing `OBS targeted problems: total N, attempted N, solved N, still open N, rules admitted N`; diagnose_problems(game, problems_path, *, variant, store_root, judge, deep_judge=None, vocabulary, spec, seconds, wake_kw, ledger_depth=6, say) (`OBS targeted diagnose ...`: what the decision does with a proposal, an ORACLE rule, the deeper judge); diagnose_sleep(game, log, problems_path, *, store_root, teacher, variant, settings, say) (what the statistical lemma sleep did with each problem); ground_oracle(card, keep); CLI `targeted <game> --store-root DIR --judge-engine UCI [--judge-depth 12] [--judge-cache F] [--judge-option NAME=VALUE ...] [--vocabulary NAME] [--log LOG ...] [--diagnose] [--diagnose-sleep --teacher UCI] [--solve] [--deep-depth D --deep-option NAME=VALUE ...] [--set NAME=VALUE ...]`; initialize.run_size(..., targeted=False, targeted_settings=None) / CLI `--targeted [--targeted-set NAME=VALUE ...]` runs solve_problems after the lemma sleep and before the problems are checked again (needs playouts); tests/test_plan78_29_targeted.py; PLAN 78.31 THE RESULT STANDARD (a mistake is a change of result; optional, default `score` unchanged): sleep.UciTeacher(..., grade_mode="score"|"result", decisive=None) (result: grade 1 win / 0 draw / -1 loss, a mate or >= `decisive` (300 cp, read off Stockfish with SyzygyPath at three units) is a win; `regret_by_grade`; the disk cache keeps RAW scores, so re-grading never re-asks); abstraction.Gates.all_named (cards.eval / eval_many gain `all_named=False`; Rust memory/cards.rs `eval_with`, `Eval.trivial`, `chance_all`: a firing HOLDS only if every named move is best, a card where every move is best is trivial and not a firing); sleep.Settings.standard, sleep.apply_standard(s, teacher) (refuses a source that grades the other way), sleep.standard_text; detectors.Regret under result = a lower class; apply_audit.audit(..., all_named=None) (claim held / refuted / trivial); initialize.run_size(standard="score") / CLI `--standard score|result [--decisive N]` switches sleep, forced seeds, evidence, audit and the regret of `OBS init compare` together; first OBS line of each tool prints the standard (`OBS sleep standard`, `OBS audit standard`, `OBS init standard`); tests/test_plan78_31_result_standard.py; PLAN 78.30 EVIDENCE ON DEMAND (evidence.py: Evidence(corpus, *, variant, spec, vocab, teacher, is_dev, per_candidate=60, total=600, cap=30000, known=None).gather(lemma) -> Gathered(cards, dev, held, corpus, binds, asked, new_asks, keys) | None; corpus positions where a forced candidate's body binds and the rule names a move, the rank source asked at a bounded number spread over groups, valued by the source's grades; sleep.sleep(..., corpus=None, prefer=None) re-applies the SAME gates on the old cards plus these, a refutation is a counter-example (re-climb once on the extended development cards, else it dies); Settings.evidence_per_candidate / evidence_total / evidence_corpus_cap; prints `OBS sleep evidence bounds|<seed>|total`; `prefer(lemma, x)` picks, among candidates passing the gates, those whose named moves at the seed problem all keep the verdict (`OBS sleep forced rule choice`); initialize.corpus_positions(logs, playout_games) -> [{start, moves, group}] (every log position, every playout position; playout_before returns `games`); targeted.keep_at_problem(variant, spec, vocab_json, judge, positions) and targeted.say_forced_rules(...) (`OBS sleep forced rule check`: fires at its seed problem, names, decision plays)); SECOND ROUTE, FORCED SEEDS: sleep.sleep(extra_positions=[{..., "force": True}]) puts the position's group in the development half only, always shows it to the rank source (detectors._sampled, Context.force_ids), raises a regret whenever the source grades the move played below its best (detectors.Regret), climbs its seeds first (not cut by max_seeds), prints `OBS sleep forced seeds: given N, confirmed N, climbed N, supported on dev N, verified held N` and per failed forced seed `OBS sleep forced seed ID died: STAGE; best candidate ...` (report key `forced`); initialize.forced_positions(rows, turning_extras) -> (forced, rest); run_size(force_problems=True) / CLI `--no-force-problems` hands every open problem (carried too) to each lemma sleep; targeted.is_counter(fact) keeps a move-counter fact (relation(N)) out of body candidates; THE WILD DRILL (reconciles the two standards, printed as `OBS targeted standard`: the local drill's right = every named move keeps the judge's class; the audit's claim held and the sleep's gates = a named move is among the rank source's best by CardSet.eval on cards revalued by the source): Wild(reader, logs, teacher, *, n=1200, settings=None, say) (a stride over all games of the wake logs, split by game dev / held-out as sleep.is_dev; .held(lemma) -> {ev, summary, named, share, fails, ok} with the lemma sleep's own held gates, abstraction.Gates; say_wild(tag, w, say)); solve_problems needs `teacher` (rank source, its disk cache reused) and `logs` unless settings wild_n=0; admission also needs the held-out wild gates, the repair reads the wild development half; `OBS targeted rule ...` ends with wild held-out fires / right / precision / chance / excess
        exact.py                 plan 78.32 THE EXACT LABELLER (game-free interface; TRAINING ONLY, the wake never reads it): NotCovered(LookupError); ExactLabeller (result(variant, start, moves) -> "win"/"draw"/"loss" for the mover or None when not covered or over; moves_keep(variant, start, moves) -> {legal move: keeps} or None; wdl(variant, moves, start=None) -> one-hot (w, d, l), the call shape of the judge objects playout_sleep.Verdicts takes, raises NotCovered; save(); close(); training_only = True; a subclass writes `_class`); SyzygyChess(directory=None, cache=None) = THE CHESS INSTANCE over python-chess `chess.syzygy` (tables from paths.syzygy_dir(); CURSED WINS and BLESSED LOSSES count as DRAWS; castling-rights positions read through their successors; JSON disk cache by EPD); LABELLERS (registry by game name), labeller_of(game, **kw); agree_with_engine(labeller, game, size, *, variant, engine, depth=20, options=None, n=200, seed=1, say) -> {agree, total, disagree} printing `OBS exact agree: N of M`
        corpus.py                plan 78.32 THE EXACT CORPUS AND THE FULL DRILL (game-free; training data: the wake, game_store and the Rust sources never read it): Corpus(rows, groups, ...) (rows {start, moves, group, before, keep {move: bool}}; groups {group: "dev"|"held"}; part(name) / non_trivial(i) / excluding(keys) / cards(spec, vocab_json, depth=1, threads=0) (built ONCE, valued 1 keep / 0 lose) / idx(part) / save(path)); load_corpus(path); default_path(game, size, n, seed); split_groups(groups, seed, dev_share); build_corpus(game, size, *, variant, n=30000, seed=1, labeller, sources=("starts","random","playouts"), logs=(), starts=None, max_starts=400, walk_plies=24, playout_plies=40, dev_share=0.5, path=None, say) -> Corpus (distinct by the kernel's position key, groups = the starts, `OBS corpus: ...`); ensure_corpus(...) (read the file or build it); drill(rule, corpus, *, part="dev", examples=5, max_bindings=256) -> {binds, names_a_move, right, wrong, trivial, truncated, wrong_examples}; drill_many(rules, corpus, *, part, indices=None, max_bindings) / drill_cases (with right_at / wrong_at card indices); file format "exact-corpus-1" Plan 78.34 adds `Corpus.extend(rows)` (append labelled rows whose groups exist; when the cards are built only the NEW rows' cards are built and merged after the old ones, so nothing is rebuilt: building all 30000 cards takes about a minute) and `Corpus.position_keys()` (the kernel's keys of the rows, kept up to date).
        exam.py                  plan 78.32 THE EXAM (training data in paths.derived, graded only, never used to admit): build_exam(game, size, *, n=300, seed=1, corpus, path=None, say) (non-trivial positions of the corpus's HELD-OUT groups, stratified by result class and group); load_exam / exam_keys / ensure_exam / default_path; grade(exam, *, variant=None, store, spec, seconds, wake_kw=None, labeller, threads=0) -> {kept, total, by_class, decisions} (the wake's own decision through playout_sleep.check, the labeller says whether it keeps); compare(prev, cur) -> {gained, lost, changed, same}; say_exam(label, g, cmp, say) (`OBS exam <label>: kept N of M (win-keeping ..., draw-keeping ...), gained N, lost N`); grade_twice(exam, *, say, **grade_kw) (`OBS exam spread: A vs B, decisions differing N`); file format "exam-1"
        board_structure.py       plan 78.33 THE BOARD'S STRUCTURE FROM THE DESCRIPTION (game-free): the vocabulary file's `axes` ([{sort, succ, goal?}], `distance_sort`) -> a rules-delta of ground facts with `signature` lines, composed after the concept deltas by game_store.spec_texts: edge_gap_<sort> ?c ?d (steps to the nearest end of the axis) and, with `goal`, goal_gap_<sort> ?x.. ?c ?d (absolute steps to the side's last coordinate); Structure(text, axes, distance_sort) (.tables, .order, .coordinate_sorts, .signatures, .sorts, .delta(), .relations(), .edge_gap, .goal_gap), from_vocabulary(text, vocabulary_file), structure_delta(text, vocabulary_file) ("" without axes), relation_names(vocabulary_file) (game_store.vocabulary lists them in `derived`, not `delta`), signatures_of(text), sorts_of(text), edge_name/goal_name/axes_of; tests/test_plan78_33_tiers.py
        induce.py                plan 78.32 INDUCTION FROM EXACT LABELS (game-free, deterministic): SETTINGS (max_literals 4, beam 8, min_right 10 on the development part, min_held 5 on the held-out part, max_heads 4, finalists 3, max_bindings 256, stride 3); search_body(corpus, facts, names, head, *, max_literals, beam, min_right, finalists, max_bindings, part="dev", stride=1) (beam search over lifted literals true at the problem, scored zero wrong first, then most non-trivial right, then fewest literals; no move-counter literal: targeted.is_counter); induce(problem, corpus, *, reader, ..., accept=None, say, tag) -> {status "found"|"open", lemma, head, dev, held, why, stage, heads_tried, candidates}; provenance(...) (origin "induced", mined_from "problem"; no verdict, no number); trial_snapshot(game, lemma, directory, vocabulary=None, judge=None) (the snapshot the store would publish with the lemma, carrying the valuation record `judge` the wake plays by: targeted.trial_snapshot leaves it out; the lemma sleep's own publish leaves it out too, so run_size re-publishes with it before induction and the exam); solve_problems(game, problems, corpus, *, variant, labeller, vocabulary, spec, seconds, wake_kw, judge=None, settings, say) -> {total, attempted, admitted, solved, still_open, reasons, rows, snapshot} (admission: zero wrong + minimum right on dev AND held, the problem's own check on a trial snapshot, no solved problem broken; `OBS induce problem i/N: ...`, `OBS induce total: ...`); initialize.run_size(..., exact=False, induce=False, induce_settings=None, exam=None, exam_size=None, corpus_n=30000) / CLI `--exact --induce [--induce-set NAME=VALUE] [--corpus-n N] [--exam PATH] [--exam-size N]` (exact: the labeller judges playouts and problems; induce: after the lemma sleep on the open problems; exam: graded twice before cycle 1 and after every cycle, `OBS exam curve`); tests/test_plan78_32_exam_drill_induce.py Also `drill_admit(lemmas, corpus, spec, vocab_json)` (the same full drill as the admission test of rules from another miner or from outside the loop: zero wrong firings on both parts). Plan 78.33 adds: TIERS, SETTINGS["tiers"], tiered_candidates(facts, names, structure) -> {relative, structure, named} (an argument is a COORDINATE when its declared sort is an axis sort: RELATIVE has none as a constant, STRUCTURE adds edge_gap/goal_gap literals read from the card's structure facts on coordinate variables, NAMED keeps a coordinate constant), tier_passes(facts, names, structure, tiers) (the narrowest-first searches; no axes: one "untiered" pass = targeted.body_candidates), search_body(..., cands=None, needs=None) (a structure literal joins a body only where its variable is in the head or in a non-structure literal), induce(..., structure=None, tiers=...) (-> also tier, tiers {tier: {stage, near, why}}), provenance(..., tier=None) (meta.tier); `OBS induce total` gains `admitted per tier` and `near misses per tier`. Plan 78.34 adds: SETTINGS["max_problems"] (None = all; else solve_problems attempts at most that many open problems, the oldest first (`first_seen`), prints `OBS induce deferred: max_problems M, open at the start O, attempted A, left for the next cycle L`, returns `left`), SETTINGS["head_tiers"] (default relative, named; `--induce-set head_tiers=relative,named`), HEAD_TIERS, head_coordinates(head, structure) (the coordinate constants of a head, by the declared sorts), relative_form(h, card, keep, structure) (a named head with its coordinates lifted, None if it names a losing move), tiered_heads(card, keep, structure, *, max_heads, head_tiers) (RELATIVE heads before NAMED ones, each with `head_tier`; no axes: targeted.heads_for unchanged, no tier; targeted.heads_for itself is not changed), induce(..., head_tiers=...) (-> also head_tier), provenance(..., head_tier=None) (meta.head_tier); tests/test_plan78_34_corpus_problems.py.
                                 (sleep.py gains `sleep(..., extra_positions=None)`: positions {id, start, moves, played, group} that join the corpus as cards beside the log's; absent, the sleep is unchanged. initialize.py gains run_size(..., playouts=False, playout_judge=None, playout_settings=None) + helpers playout_before / playout_after + CLI --playouts, --playout-set NAME=VALUE (positions, per_position, max_plies, max_turning, max_scan, check_seconds); the problem file `<root>/learned/<game>/playout_problems.json` is training data the wake never reads.)
        corpus_problems.py       plan 78.34 THE PROBLEMS THE GAMES DO NOT REACH, AND A CORPUS THAT GROWS WITH THE GAMES (game-free, TRAINING ONLY: the wake, game_store and the Rust sources never read it): sample_dev(corpus, n, seed, cycle) -> row indices (non-trivial positions of the DEVELOPMENT part, round-robin over the groups, Random(seed + cycle)); corpus_problems(corpus, rows, *, cycle, seed, n, variant, store, spec, judge, seconds, wake_kw=None, say) -> {asked, wrong, new, known, rows, new_rows} (the wake's own decision through playout_sleep.check; a move the corpus labels `loses` is a problem of source "corpus"; `OBS corpus problems cycle C: asked N, wrong W, new problems P (already known K)`); source_of(row) ("playout" when the row has no `source`) / source_counts(rows) / say_sources(rows, cycle, say) (`OBS problems by source cycle C: playout total T open O, corpus total T open O, withdrawn total T open O`); fired_positions(log) -> [{start, moves}] (the `fired` facts of a wake log); part_of_new_start(start) (a start with no group gets a new group `grown<k>`, dev for an even SHA-1 of its text, else held); grow(corpus, positions, labeller, *, exclude=frozenset(), say) -> {added, dev, held, known, exam, over, uncovered, new_groups, rows} (exact labels; a position already there by the kernel's key, an exam position, a finished game or one the labeller does not cover is not added; group = its start's group); drill_rules(corpus, rules); grow_and_withdraw(game, corpus, log, *, variant, labeller, spec, vocabulary=None, judge=None, seconds, wake_kw=None, problems_file=None, cycle=1, exclude, grown_path=None, max_check=10, say) -> {cycle, positions, added, dev, held, drilled, withdrawn [{name, wrong, dev, held, example, asked, wrong_without, new_problems, recorded}], snapshot, grown_path, size} (every rule of the store drilled on both parts of the grown corpus; a rule with any wrong firing is withdrawn with game_store.remove_lemma and a snapshot is published with the valuation record; each position where it was wrong, at most `max_check` per rule, is asked of the wake's decision without it: wrong too -> a problem of source "withdrawn", else only recorded; `OBS corpus grow cycle C: positions with a firing F, added A (dev D, held H), rules drilled R, withdrawn W [names]`, `OBS corpus grow detail cycle C: ...`, `OBS corpus withdrawn cycle C: rule N, wrong firings W (dev D, held H), e.g. <position> (losing moves [..]); the decision without the rule is wrong at X of the Y positions asked, new problems P`). initialize.run_size(..., corpus_problems=0, grow_corpus=False) / CLI `--corpus-problems N --grow-corpus` (both need --exact; corpus problems need --playouts): corpus problems are checked before the lemma sleep and joined to the problem rows after it, before induction; the growth and withdrawal run after the cycle's wake and BEFORE the cycle's exam, so the exam grades the surviving snapshot (`wake_b.survivor`, the snapshot the next cycle's problem checks use); the grown corpus is saved to `<root>/learned/<game>/corpus_grown.json`, never over the cached file. playout_sleep.merge_problems keeps a `source` of a found row; game_store.remove_lemma(game, name) removes a lemma's file from the store it points at. tests/test_plan78_34_corpus_problems.py
        human_index.py           plan 78.19 stage E, TRAINING ONLY: the human-games index; HumanSource.rank/detail take start= (answered by the kernel key, [] for a source keyed otherwise) (sqlite: position key -> move n, mover's points; build, lichess_records_iter, kernel_key, signed, points_of, default_path, main CLI)
        fact_graph.py          plan 78.20, generic: the typed fact graph (graph_of(facts, moves, vocabulary, bindings) -> HeteroData: term / fact / move / claim nodes, arg<i> / removes / adds / binds edges + reverses; Vocab (UNK = 0), encode, edge_types(max_arity, max_parts)) and graph_hash (Weisfeiler-Lehman abstraction key of 78.19 reading ii); the claim type is built but nothing supplies bindings yet. Plan 78.21: from_engine(CE.graph dict, values, lost) -> HeteroData, the graphs the net trains on are BUILT IN RUST (decision/src/graph.rs); tests/test_plan78_21.py
        graph_prior.py         plan 78.20 / 78.21: GraphNet (torch; HeteroConv SAGE over arg<i>/part<i>/removes/adds/binds edges, scores the move nodes), train(cards, out, log=...) (graphs built by CE.graph from the card's position, cross-entropy against the optimal set, dev/held by group, OBS gnn lines incl. the torch-vs-Rust parity line), export(checkpoint, weights.json) (the Rust weights file), GraphSource(weights_path, specs=, vocabulary=, game=) (rank source `gnn`, inference kit, ALL IN RUST through CE.GraphModel, no torch needed; .ms, .timing()), build-cards / train CLI; tests/test_plan78_20.py, tests/test_plan78_21.py
                                 and HumanSource (rank/detail/__call__/close; source "human"); read by sleep's training kit only; tests/test_plan78_19e.py
    approaches.md                registry of end-to-end approaches
    approaches/<name>/           config.py + src/ + experiments/
    research/
      components/<component>/    encoder | planner | search | memory
        approaches.md            registry of this component's approaches
        approaches/<name>/
          src/                   SHIPPING code
          experiments/           lab scripts (not packaged, no __init__.py)
          tests/                 unit tests for this approach
          logs/ artifacts/ data/ non-shipping output
      tools/                     by TYPE: probes figures embeddings ablations viz
                                 stats_eval training_infra chess_specific
      docs/                      research docs + archive/ + writing/
      infra/                     preemption, checkpointing, observability, cloud
      weekly_report/             phase reports
      assets/ logs/              (assets live at the repo root; see Deviations)
    deployment/
      server/                    assistant_server, uci_engine, banksync
      web/                       board UI
      docker/                    Dockerfile, compose, nginx, prometheus, grafana
```

---

## Imports

**No new top-level packages. Everything is `catspace.*`.** `experiments/`, `tools/` and
`infra/` are gone as top-level names; the restructure deleted ~294 `sys.path.insert` lines
that existed only to make bare imports of them work.

- An approach's `src/` is a regular package and gets an `__init__.py` that re-exports the
  approach's public names, so callers can write
  `from ...approaches.reachability_field.src import ReachabilityField`.
- `experiments/`, `logs/`, `artifacts/` and `data/` **must not** have an `__init__.py`.
  That is what keeps them out of `setuptools.packages.find()` and out of the wheel.
- They are still importable — a directory without `__init__.py` inside a regular package is
  a PEP 420 namespace portion — so research scripts can import each other the way a lab
  notebook does. This is deliberate, not an accident of packaging.
- End-to-end configs name approaches **by string** (`"search:puct_mcts"`), never by deep
  import path. Only `catspace/registry.py` does deep imports.
- Do not re-export a function whose name collides with its own module. `subgoal_cascade`
  does not re-export `decompose`, because `from ...src import decompose` must keep meaning
  the module.

---

## approaches.md — the schema

Every component directory and `catspace/approaches/` has an `approaches.md`. Every folder
must have an entry and every entry must have a folder;
`scripts/check_approaches.py` enforces both directions and runs in P13 / CI.

Each entry is a level-2 heading (`## <folder_name>`) followed by:

| field | meaning |
|---|---|
| **folder** | path, relative to the registry |
| **status** | `active` \| `parked` \| `superseded-by:<name>` |
| **hypothesis** | one line: what this approach claims |
| **definition of done** | the measurement that would settle it |
| **results** | link to where the numbers are, or `—` if it has not reported |
| **added** | date |
| **owner** | owner of record |

Optional: **wiring** (end-to-end approaches only), **notes**, **why parked**.

Write `—` for results when an approach has not reported. Do not invent a link: an entry
that overclaims is worse than one that admits it has no numbers yet.

---

## The component contract

`catspace/interfaces.py` holds the Protocols. Anything satisfying one can be injected into
`LayeredEngine`:

```python
class ValueModel(Protocol):        # leaf evaluation, white-POV, in [-1, 1]
    def values(self, boards: list) -> np.ndarray: ...

class MovePrior(Protocol):         # board -> {move: prob}
    def priors(self, board: chess.Board) -> dict: ...

class SubgoalSelector(Protocol):   # propose the current Region, or None
    def select(self, board: chess.Board) -> Region | None: ...
```

plus the `Region` (goal-as-region) and `SearchOutcome` dataclasses.

The rule these encode, and it is the important one: **subgoals enter through the prior,
never through the value.** The value is the global objective. See DECISIONS §8.

> **Known gap.** `catspace/registry.py` also defines a uniform constructor —
> `registry.build("search:puct_mcts", **kwargs)` expects the approach's `src` package to
> expose `build(**kwargs)`. **No approach implements it yet** (0 of 22). `registry.load()`
> works and is what everything currently uses. New approaches should provide `build()`;
> retrofitting the existing 22 needs a real decision per approach about what its
> constructor arguments are, and was not guessed at during the restructure.

---

## Production vs research

`catspace/research/components/<c>/approaches/<name>/src/` is **SHIPPING CODE** despite
living under `research/`. It goes in the wheel and runs in the container.

Non-shipping, under any approach: `experiments/`, `logs/`, `artifacts/`, `data/`.

The practical consequence, and the one that actually bit during the restructure: **shipping
code must never import from an `experiments/` directory.** `bootstrap_mate/src/engine.py`
was lazily importing three model classes (`DTMTok`, `DTMNet`, `PlanNet`) out of training
scripts that the Docker image does not copy. Model architectures the engine loads at play
time belong in `src/`; the trainer imports them from there, which also stops the training-
time and play-time featurization from drifting apart.

---

## Tests

- **Cross-component and integration** → root `tests/`.
- **Unit tests for one approach** → `approaches/<name>/tests/`.

`pytest` `testpaths` covers both. `*/experiments` is excluded from collection: lab scripts
with `test_`-shaped names (`test_field_fullgame.py` — "test the field on full games") are
not tests and must not be imported at collection time.

---

## Data

**Generated data** lives under `data/`, DVC-tracked, and is **always** reached through the
path registry — never a literal relative path.

```python
from catspace.io import paths

paths.derived("dtm_endgame.npz")     # -> /abs/.../data/derived/dtm_endgame.npz
paths.sep("lichess_mc2.pt")
paths.experiment("krrkbp_test_n200.json")
paths.figure("reach_curve.png")
paths.syzygy_dir(), paths.engine("maia/maia-1900.pb.gz")
```

Why this is a rule and not a preference: `"data/derived/foo.npz"` only resolves when the
process happens to have been launched from the repo root. That is why cross-module artifact
handoff kept breaking, and why the container silently pointed at nothing. The restructure
replaced 901 such literals across 294 files.

- The **directory** is declared once, in `catspace/io/paths.py`. The **filename** stays at
  the call site, where it is meaningful.
- File accessors return `str`, so they drop into every place a literal used to sit.
- `REPO_ROOT` is found by walking up for `pyproject.toml`/`.git`, not by counting
  `parents[N]`, so moving a module deeper cannot break it.
- **Third-party** downloads (tablebases, engine binaries, corpora, Maia weights) are not
  data: they go under `assets/`, never `data/`.

---

## Historical documents

**Never rewrite a dated or archived document to match the new layout.** JOURNAL.md,
MILESTONES.md, README.md, `research/docs/archive/*`, `research/docs/writing/*` and
`research/weekly_report/*` are the research record; paths in them were correct when
written. They carry a banner pointing here instead. The migration tooling skips `.md`
entirely for this reason.

This does not apply to Python docstrings that name their own file path — those are code
comments and were updated.

---

## Deviations from the refactor plan

Recorded so the difference is a decision rather than a discrepancy:

1. **`assets/` and `data/` stayed at the repo root** rather than moving under
   `research/`. Deployment needs `assets/engines` (compose mounts it), so they are not
   research-only; the path registry, written in P1, already resolved them at the root.
2. **The path registry is `catspace/io/paths.py`**, not `research/tools/io/paths.py`. It
   is imported by shipping code and by deployment, so it belongs at wrapper level.
3. **Directory names use underscores**, not hyphens (`gauntlet_harness`,
   `weekly_report`) — they have to be importable.
4. **`uci-engine` and `http-assistant` did not become end-to-end approach folders.** Their
   shared glue did: it is `catspace/approaches/bootstrap_mate/`, which both deployment
   shells build from. Splitting the 949-line `assistant_server.py` into analysis logic and
   HTTP shell was judged a separate change, not part of a move.
5. **Generated data was not physically scattered** into per-approach `data/` folders. The
   registry is the indirection that makes location a one-line change; relocating ~100
   DVC-tracked outputs is a separate, riskier operation and the plan's access rule (always
   via the registry) is satisfied without it.

## Rust engine (rust/engine, a Cargo workspace; game-free layer crates)

Plan 78's end-to-end system (docs/ANALYST_PLAN.md 78). One crate per layer; every layer crate reads only the `Game` trait and names no chess object
(tests/test_plan78.py's generic test scans their sources). The build is offline: `cargo test --release --offline --workspace --exclude cs_engine`;
the Python extension `cs_engine` is built from `py/` with `maturin develop --release --offline` (the venv's python).

```
rust/engine/
  gdl/                           cs_gdl: the GDL reader and spec evaluator (compose a variant's delta onto a base description); plan 78.19 A: `Program::run_for(wanted)`
                                 (only the dependency cone's strata; `cone`, `cone_rules`, `run_in` for the audit's wrong cone; `run` stays the oracle), statics shared by
                                 `Rc` with prebuilt indexes (`Db` layered over them), indexes kept current on `add`; tests/demand.rs (run == run_for on the cone, negative control)
  game/                          cs_game: the Game trait (roles, mover, moves, apply, outcome, key, base_facts), GdlGame (any description),
    src/kernel.rs                  KernelGame over shakmaty's variants -- THE ONE CHESS MODULE (rules ingestion); src/outside.rs the outside engines
    src/certify.rs                 co-play certification of a kernel against its description (states, moves, facts); src/rng.rs the seeded RNG
    src/gdl_game.rs                FactView: the vocabulary's derived relations evaluated over base facts; `with_demand(bool)` (off = full `run`, the oracle), `rows` (derived
                                   rows interned, no strings), `cone_size`, string<->term memos (plan 78.19 A)
    src/fact_audit.rs              plan 78.19 A: `positions` (seeded random play of any Game) and `audit` (demand view vs full view: equal/unequal, ms each); py `facts_audit`
    tests/demand.rs                demand == full facts over random play, the cone's size, a negative control (a wrong cone is caught), rows vs facts, statics' copy cost
  operator/src/lib.rs            cs_operator: THE SHARED INTERFACE -- Goal (+ its Mode; plan 78.25 adds Goal::Reason { role }), Budget, Outcome (status, move, value, candidates, evidence, line, events,
                                 measured cost in parts, facts), Operator, Controller, drive; Kit (Inference | Training) is a property of each Operator and
                                 drive = drive_in(Kit::Inference, ...): an inference drive refuses a training operator in the type (plan 78.17)
  tactics/                       cs_tactics: forcing (proof), the ledger search (events along the line, a Judge turns them into a value: Count or Worth), census;
                                 src/ledger.rs also `impl Judge for &J` (an operator can hold a judge the decision lends it)
                                 src/operators.rs their Operator wrappers (ForcingOp, LedgerOp, CensusOp, plan 78.25 ReasonOp)
    src/reason.rs                  plan 78.25 THE REASON OPERATOR, game-free: `reason(g, s, judge, depth, fuse) -> Report` (+ `reason_within(.., seconds)`: ONE deadline for the whole chain, through `census::census_within(.., seconds)`; a reading the clock cut or left no time for is counted in `cut`, the chain is then a GAP) = the threats against the side to move by stake (the census of the other side after a pass, no defect; no pass: `must_answer`, every move read as an answer), each threat's FRAME (the base facts its line removes or adds, the facts every legal move changes left out), the exact NET of every move of ours one ply deeper (one ledger search, `root_values`), the ANSWERS (net beats `-stake`; the question asked of each is whether they can still carry the threat out: the threat's first move, if still legal, is read exactly (`Search::gain` is pub for this) and its line searched for the targets -- Parry = they cannot, Counter = they can and our net if they do (`if_executed`, with `executed_line`) is above 0 (a greater threat), Cover = it is 0 (the taker is taken back or matched), Lessen = it is below 0 but better than passing; a goal threat has parries only; a threat's line is cut at its last ledger event so the frame holds its own facts only), the IRRELEVANT units (the facts of ours a move removes, owner read off the roles' names in the facts; a unit none of whose moves answers and none of whose facts is in the frame; moves left unread by a cut are reported unread, never irrelevant), with no threat OUR tactics (census, one defect) as the candidates; `explain(&Report) -> Vec<String>` = the chain as plain `reason:` lines; named REASON_DEPTH / REASON_FUSE / ANSWER_FUSE_FACTOR. An order and an explanation, never a verdict; py `reason`
  planning/                      cs_planning: keep, the density prior; src/mcts.rs UCT Monte Carlo tree search (Rollout pluggable; MctsOp answers Goal::Estimate);
                                 tests/mcts.rs on tic-tac-toe in GDL
    src/rollout.rs                 the stronger rank source as pluggable pieces: Leaf (Draw, Balance: value of a cut playout), Greedy rollout, Careful rollout (plan 78.26: a move's gain less the best gain it leaves the next mover; a move that ends the game counts by its outcome, SURE), Prior + Weighted rollout,
                                   GainPrior (one notion of gain for Greedy and Weighted)
    tests/rollout.rs               each piece alone, on a toy units-only Game
    src/wdl.rs                     plan 78.26 WIN / DRAW / LOSS, game-free: Wdl {win, draw, loss} mass (n, p, score, sharpness, add, add_mass, scaled, flip), split(value) -> [w, d, l] (the score kept exactly), consts SHARP, MIN_MASS, TABLE_MIN, TABLE_CAP, WDL_PLAYOUT, WDL_SCALE, TOP_MOVES, SCORE_SCALE, sharp(&Wdl) -> Option<bool>, word, Table {name, rows} (keys(g, s): `exact:<hex>` then `units:<kinds per role>:<mover>`; lookup(g, s, role) -> Option<(key, Wdl)>, a plain map the caller fills: no serde here), leaf_scale(g, s, judge) (WDL_SCALE x the smallest positive worth at the root), WdlOp (name "wdl", Kit::Inference; Goal::Estimate = playouts pooled with a table hit, Goal::Prefer = the probabilities pick among `among`; facts wdl, sharpness, playouts, table_hit, move_wdl); mcts::Search also carries root_wdl and move_wdl tallies
    tests/wdl.rs                   split, the tallies, the table back-off and flip, sharp, Prefer picks the win, on tic-tac-toe in GDL and a toy units game
  memory/                        cs_memory: content-addressed snapshots, the lemma language, the one matcher, cards (the corpus lemmas are verified on), LemmaOp; src/op.rs also `delta` and plan 78.24's `Effects` (the one reader of a move's effect, state or derived delta relations)
  decision/                      cs_decision: Params (plan 78.25: `reason`, default off; plan 78.26: `wdl` default off and `wdl_table: Option<Arc<Table>>`; WDL_SHARE, WDL_EPS, WDL_SEED, WdlMode, wdl_mode), the Fixed controller (lemma, proof, [plan 78.26 with `wdl`: the estimate (WDL_SHARE of the clock; Careful playouts cut at WDL_PLAYOUT = 8 plies, Balance leaf), then sharp or unknown: reason and the ledger; quiet: the ledger with ties, then wdl Prefer among its candidates; one `wdl: <mode> (...)` trace line], [reason: REASON_SHARE of the clock, its candidates join the lemmas' as the ledger's `prefer`], ledger, prior), batch (JSON Lines log), cards (the card builder)
    src/lemma_prior.rs             LemmaPrior: a snapshot's bound lemmas as a playout Prior (cs_planning::rollout), matched as LemmaOp does
    src/lemma_prior.rs (plan 78.28)  RulesFirst<P: Prior, R: Rollout>::new(prior, fallback, eps, seed): a playout policy that plays uniformly among the moves the prior weighs above 1.0 (the lemmas' named moves) when any, else the fallback's pick, a uniform move with probability eps; `named` / `picks` counters; unit tests in the file
    tests/lemma_prior.rs           the lemma prior on a small GDL game; an empty snapshot gives uniform weights
    src/graph.rs                   plan 78.21: GraphSpec::new(description text, graph vocabulary JSON), .graph(g, s) -> Graph: the typed fact graph of a position, generic (terms typed by sort,
                                   product sorts fused into objects with part<i> edges, always-present constants, static / state / derived fact nodes, move nodes with removes / adds, an empty claim type, reverses;
                                   constant term / relation vocabularies read from the description's sorts and signatures)
    src/graph_net.rs               plan 78.21: GraphNet::load / from_json / validate(spec) / forward(&Graph) -> one score a move: the torch GraphNet's forward (SAGEConv per edge type, sum, residual LayerNorm, MLP head) from the JSON weights file, plain-loop matrix products
    src/graph_prior.rs             plan 78.21: GraphPrior {spec, net}: scores(g, s) -> (move names, scores), a cs_planning Prior (softmax x n; uniform on error), timing counters (facts / graph / forward)
    tests/graph.rs                 graph.rs on a made-up GDL game (sorts, an object, a scale, a static / state / derived relation): typing, fusion, constants, identical statics, edgeless constants, move effects; the forward pass on a hand-computed 1-layer dim-2 net; the prior
  learn/                         cs_learn: THE TRAINING KIT -- watching a teacher (src/watch.rs; Teacher is training-only in the type), the valuation miner,
    src/advice.rs                  Advisor (an outside source's ordering of the legal moves), TeacherOp (Kit::Training), ScriptedAdvisor for tests.
                                 No decision-path crate depends on cs_learn (tests/test_plan78_14.py reads the manifests)
  py/src/lib.rs                  cs_engine: the one Python entry (best_move, plan 78.28: `playout_game(variant, fen=None, moves=[], seed=1, max_plies=400, rollout="careful", store=None, spec=None, worth={}, valuation="", eps=0.1, boost=2.0) -> {moves, outcome (per role or None), capped, named, plies}` (one playout to a real result; "rules" = cs_decision::RulesFirst over the snapshot's lemma prior, else the careful look; needs store + spec), solve_batch (both take `reason=False, wdl=False, wdl_table=None`; the table is a file path), plan 78.26: `wdl(variant, fen=None, moves=[], seconds=0.2, max_nodes=10_000_000, worth={}, ours_first=False, valuation="", table=None, among=None, seed=1, max_playout=8, rollout="careful") -> dict` (`rollout`: "careful" or "greedy") (win, draw, loss, n, sources, score, sharpness, sharp, table_key, table_counts, playouts, cut, nodes, seconds, evidence, move, moves [{move, win, draw, loss, n, score, visits}]; `among` asks the probabilities to pick), `wdl_keys(variant, fen=None, moves=[]) -> [str]`, `wdl_keys_along(variant, fen=None, moves=[]) -> [[str]]`; table file format {"format": "wdl-table-1", "game", "provenance", "rows"}, plan 78.25: `reason(variant, fen=None, moves=[], depth=2, fuse=200000, worth={}, ours_first=False, valuation="") -> dict` (threats, answers, irrelevant, unread, leaving, ours, candidates, candidate_moves, evidence, readings, cut, nodes), build_cards, CardSet (card, eval, eval_many, save, load, merge, plan 78.23: revalued(values) -> CardSet: per-card {move: value}, a missing move cuts the card, None keeps it), estimate, facts, snapshots, certify, watch, mine, plan 78.21: graph, graph_vocab, GraphModel (rank / scores / timing))
```

The Python side of the loop is `catspace/incubator/analyst/discovery/` (wake, sleep, diagnose, abstraction, game_store);
plan 78.24 A: `game_store` named vocabularies (`vocabulary_file/vocabulary/spec_texts(game, name=None)` read `vocabulary.<name>.json`, whose optional `concepts` is a rules-delta path or list composed in order; `read_lemmas(game, vocabulary=None)` returns the lemmas of that vocabulary by `meta.vocabulary` (None = default, `GS.ANY` = all); `build/publish(game, ..., vocabulary=None)`); `abstraction` gains the step DROP A RESPONSE LITERAL (`drop_response(lemma, side, i)`; `size` counts response literals; `Gates.override(["NAME=VALUE"])`); a delta relation may be DERIVED: `cs_memory::op::Effects::at(view, relations, base).of(after_base)` reads a move's effect (base facts when every delta relation is a state relation, else the full fact views before and after), used by card_of, LemmaOp, LemmaPrior and GraphSpec::graph; tests/test_plan78_24.py (+ rust/engine/decision/tests/lemmas.rs); the learned knowledge is
`rules/learned/<game>/` (vocabulary, lemmas) and its immutable snapshots under `data/derived/snapshots/<game>/`.

Plan 78.21, the human concept vocabulary as rules: `rules/concepts/chess.gdl` is a DELTA (additive only) on `rules/chess.gdl` holding the concepts (defended, hanging,
pinned, fork, open_file, half_open_file, passed_pawn, isolated_pawn, doubled_pawn, absent, bishop_pair) and the static `diagonal_of` (26 named diagonals);
`rules/learned/chess/graph_vocabulary.json` names what a graph reader sees (state, derived, static relations, `objects`, `scales`, the `concepts` delta), read by
`game_store.graph_vocabulary(game)` / `graph_spec_texts(game)`; each concept is tested against a python-chess oracle in `tests/test_plan78_21_concepts.py`.

Plan 78.24 B, relations between units: `rules/concepts/chess_units.gdl` is a second DELTA composed after `rules/concepts/chess.gdl` (additive only): the sort `distance` (d0..d7, never the rank
numerals) with static `file_gap` / `rank_gap` / `dist_max` / `closer` / `reach`, and `apart` (king-step distance between a king and another unit), `flight`, `opposition`,
`diagonal_opposition`, `to_promote`, `king_ahead`, `in_square`. `rules/learned/chess/vocabulary.relational.json` is the second lemma vocabulary (the keys of `vocabulary.json` plus
`concepts`, a list of deltas composed in order); each relation is tested against a python-chess oracle in `tests/test_plan78_24_concepts.py`.

Plan 78.32, the hand step (key squares): `rules/concepts/chess_keys.gdl` is a third DELTA composed after `chess_units.gdl` (additive only): `key_square`, `key_gap` (our king's distance to the
nearest key square of our pawn), `key_guard` (the same for the opponent's king), `king_in_front`; tested against a python-chess oracle in `tests/test_plan78_32_keys.py`.
`rules/learned/chess/vocabulary.keys.json` is the third lemma vocabulary (the relational one plus these words). `rules/concepts/chess_keys.proposed.json` holds rules PROPOSED by hand
(not lemmas of the store): `initialize --seed-rules FILE` files each into the SANDBOX store only with zero wrong firings on the exact corpus (`induce.drill_admit`);
`initialize --drill-sleep` puts the lemma sleep's rules through the same test. `game_store.head_judge(game)` is the valuation record of the HEAD snapshot (a snapshot built on the head
carries it); `apply_audit` counts a firing that names no move as `silent`, not refuted.
