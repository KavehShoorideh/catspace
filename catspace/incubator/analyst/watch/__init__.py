"""THE SURPRISE WATCHER (Kaveh 2026-10-05). The tactical layer watches games -- anybody against anybody, Lichess
records on disk -- and after every move forecasts, with probabilities, (short) the material balance once the
exchanges settle, as the sum of its FRAMES' contributions (a frame is a square and the pieces contesting it), and
(long) the game's result. Surprise is -log2 of the probability given to what then happened. Knowledge is a tree of
rules over frame attributes (a Ripple-Down tree: an exception refines its parent); a rule's forecast is the count of
what happened when it applied, smoothed toward its parent. The rule base starts empty (roots only); Claude, in
session, reads the largest surprises after each batch of games and writes the missing rule or exception, with the
position as its cornerstone. The loop is judged by the moving-average surprise on the games that follow.
docs/WATCH.md."""
