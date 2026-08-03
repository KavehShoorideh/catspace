This is an attempt to write an alternative arch without the AI directly writing.

A good World Model for chess will compress the useless info out of the embedding, things like random moves at unimportant times. What matters are the critical moves, moves where one wrong step costs us the game. I.e., most of the success probability is concentrated in one move, or maybe two.

So, let's train a JEPA style encoder for chess. What we would want to predict are three kinds of futures: immediate, medium term (to subgoal), and long term (to goal). The job of the immediate JEPA is to predict an embedding representing the probability distribution of how the next few moves will unfold, conditioned on everything we know about the game, i.e. clock, board state, player strength, player history with each other, and maybe even info from their online profiles.
- a probability distribution over short sequences of tokens, each token a legal move

For the next layer, each sequence will become a token.

The medium term embedding will return the subgoals, which should be places where success probability bottlenecks, and their distance out.
This should be:
- a probability of occurence,
- a sequence of moves 1-k,
- an embedding representing the policy at the kth move

the long term 

I need the following components
1) Chess encoder
  - Board state
  - player clocks
  - player strengths
  - (later) info from players' online profiles, game history
2) Futures predictor(s)
  - short horizon (next few plies)
  - medium horizon (to subgoal)
  - long horizon (to goal)
3) Search Executor
4) Search cache memory
5) Opening memory (book)
6) Endgame memory (tablebase) <=7 w/ 149 piece combinations, 3 end states, each end state multiple types
7) Central RL agent (planner)

We jointly train the Chess encoder and Futures predictors JEPA style to predict interesting MCTS results:
1) 