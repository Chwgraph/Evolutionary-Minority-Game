# Evolutionary-Minority-Game
Simulation for variants of evolutionary minority game.

# Introduction

## Definition 
Agent-based models of complex adaptive systems provide valuable insights into the emergent properties and large-scale effects of locally interacting components (Holland, 1995). One model that combines the general properties of such systems is the Minority Game (MG) (Challet and Zhang, 1997) inspired by the El Farol bar-attendance problem (Arthur, 1994). In the MG, a population of agents with bounded rationality compete for a limited resource. The traditional game involves N players (agents) forced to make a binary decision: 0 (eg. go to the bar, or take route A) or 1 (eg. do not go to the bar, or take route B). The agents share a common “memory” or look-up table, containing the outcomes from the m most recent occurrences. The resulting 2m possible histories are then used to generate strategies for
making an appropriate binary choice. At each time step, an agent receives one point if their decision places them in the minority and loses a point otherwise. The game evolves as the agents modify their behaviours (or strategies) based on the previous experiences. (Michael 2006) After an agent's score falls down the threshold, he would change his strategy by picking some of the best agent's strategies and some randomly chosen new strategies to form his new strategy pool.

In the studies of evolutionary minority games, there are many different settings for the threshold and the rule of mutation. In this project, we simulate different variants of EMG and compare their systematic behaviors. The table below displays the versions of EMG and MG we studied in this project:

|Version|Mutation|Threshold|Time to Eliminate|Elimination Rule|
|---|---|---|---|---|
|Basic|None|None|None|None|
|EMG(V1)|Best+Random|Dynamic|Fixed Time-steps|Worst|
|EMG(V2)|Random|Fixed|Anytime|Below Threshold|
|EMG(V3)|Best+Random|Fixed|Fixed Time-steps|Below Threshold|
|EMG(V4)|Best+Random|Dynamic|Fixed Time-steps|Poverty Level|

## Model Formulation

In this project, we always consider the case in which each agent has two strategies in their strategy pools. Let us consider a population of N (odd) players, each has some finite number of strategies $\textit{S}$. At each time step, everybody has to choose to be in side A or side B. The payoff of the game is to declare that after everybody has chosen side independently, those who are in the minority side win. In the simpliest version, all winners collect a point. The players make decisions based on the common knowledge of the past record. We further limit the record to contain only yes and no, e.g. side A is the winning side or not, without the actual attendance number. Thus, the system's signal can be represented by a binary sequence, meaning A is the winning side (1) or not (0). Let us assume that our players are quite limited in their analysing power, they can only retain last M bits of the system's signal and make their next decision basing only on these M bits. We call $M$ the momery size and the binary sequence memory. All possible predictions based on the memory form the whole strategy pool. Each player has a finite set of strategies. A strategy is defined to be the next action (to be in A or B) given a specific signal's M bits. We randomly draw S strategies for each player, and some strategies maybe by chance-shared. (Challet and Zhang, 1997) S is called the strategy pool size. For each certain player, he would mutate his strategy by picking one of the strategies owned by the best player, and randomly choosing a new strategy from the whole strategy pool. 

## Parameters
The common parameters are as shown below:

|Parameter|Value|
|---|---|
|Rounds|4000|
|Rounds Before Recording|200|
|Memory Size|5|
|Sample Size|10|
|Strategy Pool Size|2|


# Detailed Introduction to Variants of EMG

## EMG(V1)

This version of EMG was introduced by Challet and Zhang (Challet and Zhang, 1997). Initially, each agent draws randomly one from his strategy pool and use it to predict the outcome. After each round, each strategy gains $\textit{virtual}$ $\textit{score}$ if its prediction matches the outcome. Agents choose the strategy with the highest virtual score in his strategy pool to predict for each round then. If the prediction matches the outcome, the agent gains a $\textit{real}$ $\textit{score}$. After a fixed number of rounds, the worst agent - the agent with the lowest real score will be eliminated and a new agent will join. This new agent copies one strategy from the strategy pool of the best agent and randomly chooses another strategy from the whole strategy pool. 

## EMG(V2)

This version of EMG was introduced by T.S. Lo and his co-authors(T.S. Lo et. al, 2000)

This variant of EMG consists of an odd number N of agents repeatedly choosing to be in room 0 ~e.g., sell! or room 1 ~e.g., buy!. The winners are those in the minority room. A single binary digit denotes the winning minority room. The agents have a common memory look-up table containing the outcomes from the most recent occurrences of all 2m bit strings of length m. Faced with a given bit string of recent occurrences, each agent chooses the outcome in the memory with probability p, which we refer to as the agent’s gene value, and chooses the opposite action with probability 1-p. To incorporate evolution, +1(-1) point is assigned to every agent in the minority ~majority! room at each timestep. If an agent’s score falls below a value d (d<0), a new p value is chosen randomly within a range R centered on the old p. We impose reflective boundary conditions to ensure that 0<p<1, although the conclusions do not depend on this particular choice of boundary conditions.

## EMG(V3)

For comparison purpose, we proposed a new version of EMG combining a fixed threshold with an elimination per certain time-steps. In this project, it is referred to as EMG(V3) hereafter.

## EMG(V4)

This version of EMG was proposed by Li et al. (Li et al., 2000) 

In this version of EMG, elimination occurs after fixed number of rounds, called $\textit{generations}$. Agents in the last group determined by $\textit{poverty}$ $\textit{level}$ will change their strageties in the way we introduced in EMG(V1).

# Acknowledgement

Joint work with Haoxian Li, HKUST DDM alumni.


# References
1. Holland, J. (1995). Hidden Order: How adaptation build complexity. Addison-Wesley, Reading.
2. Challet, D. and Zhang, Y.C. (1997). Emergence of cooperation and organisation in an evolutionary game. Physica A, 246:407.
3. Arthur, W. B. (1994). Bounded Rationality and Inductive Behavior (the El Farol Problem). American Economic Review, 84:406-411.
4. Michael, Mickey (2006). Evolutionary minority games with small-world interactions. Physica A, 521-528.
5. Johnson, N.F., Hui, P.M., Jonson, R. and Lo, T.S. (1999). Self-Organized Segregation within an Evolving Population. Physical Review Letters, 82(16):3360.
6. Lo, T. S., P. M. Hui, and N. F. Johnson. "Theory of the evolutionary minority game." Physical Review E 62.3 (2000): 4393.
7. Y. Li, R. Riolo, & R. Savit, Evolution in Minority Games(I): Games With a Fixed Strategy Space, Physica A, 276, 234-264, 2000.
