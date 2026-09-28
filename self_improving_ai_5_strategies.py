"""
=======================================================================
SELF-IMPROVING AI: 5-STRATEGY REINFORCEMENT-LEARNING TOY MODEL
=======================================================================

Purpose
-------
This program demonstrates a simple mathematical mechanism for recursive
AI self-improvement.

An AI system has a capability level:

    A_t in [0, 1]

At every research generation, the AI can choose ONE of five research
strategies:

    1. Prompt / scaffold improvement
    2. Search / planning improvement
    3. Synthetic data generation
    4. Fine-tuning
    5. Inference optimization

The AI DOES NOT initially know which strategy is best.

It learns from experimentation using a simple reinforcement-learning
algorithm: epsilon-greedy action selection + Q-value learning.

The important recursive feedback loop is:

    better AI capability
            ↓
    better probability of successful AI research
            ↓
    successful research improves the AI
            ↓
    the next AI researcher is more capable
            ↓
        repeat

The reinforcement-learning loop is:

    choose research strategy
            ↓
    run experiment
            ↓
    observe improvement / reward
            ↓
    update estimated value of strategy
            ↓
    make better research choices in the future


This is NOT intended to be a realistic model of frontier AI research.

It is a small mathematical model intended to illustrate the mechanism
behind recursive self-improvement in a transparent way.


Mathematical model
------------------

For research strategy i:

    p_i(A_t)
        = sigmoid(
            logit(base_success_i)
            + sensitivity_i * (A_t - A_0)
          )

This means that a more capable AI becomes better at executing research.

If the experiment succeeds:

    ΔA_t = gain_i * A_t * (1 - A_t)

Otherwise:

    ΔA_t = 0

Therefore:

    A_(t+1) = A_t + ΔA_t


The A_t term means:

    better AI -> better research

The (1 - A_t) term introduces diminishing returns:

    as capability approaches 1,
    further improvement becomes harder.


Reinforcement learning
----------------------

The AI maintains an estimated value Q_i for every strategy.

It chooses strategies using epsilon-greedy exploration:

    with probability epsilon:
        explore a random strategy

    otherwise:
        exploit the strategy with the highest Q-value


After observing reward r_t = ΔA_t:

    Q_i <- Q_i + alpha * (r_t - Q_i)

Because the environment changes as A changes, we use a constant
learning rate alpha rather than a simple historical average.

=======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. GLOBAL CONFIGURATION
# =====================================================================

# Random seed makes the stochastic simulation reproducible.
RANDOM_SEED = 42

# Number of recursive research / self-improvement cycles.
N_GENERATIONS = 200

# Initial AI capability.
#
# A = 0 means essentially no capability.
# A = 1 represents the theoretical upper bound in this toy model.
INITIAL_CAPABILITY = 0.40


# ---------------------------------------------------------------------
# Reinforcement-learning parameters
# ---------------------------------------------------------------------

# Q-learning update rate.
#
# Higher alpha:
#     agent reacts faster to recent results
#
# Lower alpha:
#     agent changes its beliefs more slowly
#
# We intentionally use a constant alpha because the reward distribution
# changes as AI capability changes.
ALPHA = 0.15


# Initial probability of random exploration.
EPSILON_START = 0.35

# Minimum exploration probability.
#
# Even after learning, we continue experimenting occasionally.
EPSILON_MIN = 0.05

# Controls how quickly epsilon decreases.
EPSILON_DECAY = 0.985


# Initial Q-value assigned to every strategy.
#
# The units are expected capability improvement per experiment.
INITIAL_Q_VALUE = 0.010


# =====================================================================
# 2. DEFINE THE FIVE AI R&D STRATEGIES
# =====================================================================

"""
Each strategy contains three hidden properties.

The RL agent DOES NOT directly observe these parameters.

base_success:
    Probability that the strategy succeeds when A = INITIAL_CAPABILITY.

gain:
    How much capability improvement the strategy can produce when
    successful.

sensitivity:
    Determines how much the strategy benefits from having a more
    capable AI researcher.

This creates an interesting trade-off.

Some strategies are:

    - easy but produce modest gains
    - difficult but potentially produce large gains

Therefore a strategy that is unattractive for a weak AI may become
very useful once the AI becomes more capable.
"""

STRATEGIES = [
    {
        "name": "Prompt / scaffold",
        "base_success": 0.80,
        "gain": 0.050,
        "sensitivity": 0.40,
    },
    {
        "name": "Search / planning",
        "base_success": 0.65,
        "gain": 0.065,
        "sensitivity": 0.80,
    },
    {
        "name": "Synthetic data",
        "base_success": 0.38,
        "gain": 0.100,
        "sensitivity": 2.00,
    },
    {
        "name": "Fine-tuning",
        "base_success": 0.20,
        "gain": 0.150,
        "sensitivity": 4.00,
    },
    {
        "name": "Inference optimization",
        "base_success": 0.70,
        "gain": 0.055,
        "sensitivity": 0.60,
    },
]


# =====================================================================
# 3. MATHEMATICAL HELPER FUNCTIONS
# =====================================================================

def sigmoid(x):
    """
    Standard logistic sigmoid.

    Converts any real number into a value between 0 and 1.

        sigmoid(x) = 1 / (1 + exp(-x))

    We use it to transform research capability into a success
    probability.
    """

    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    """
    Inverse of the sigmoid function.

        logit(p) = log(p / (1-p))

    Allows us to start with an explicit probability such as 0.80,
    modify it in log-odds space, and convert it back with sigmoid().
    """

    return np.log(p / (1.0 - p))


def research_success_probability(strategy, capability):
    """
    Compute probability that a particular R&D strategy succeeds.

    Mathematical equation:

        p_i(A)
        =
        sigmoid(
            logit(base_success_i)
            +
            sensitivity_i * (A - A_initial)
        )

    Interpretation
    --------------

    If capability = INITIAL_CAPABILITY:

        p_i(A) = base_success_i

    As capability A increases:

        p_i(A) increases.

    Therefore a better AI researcher becomes more effective at
    conducting AI research.

    This is one of the mechanisms that creates recursive improvement.
    """

    base_probability = strategy["base_success"]

    sensitivity = strategy["sensitivity"]

    capability_difference = capability - INITIAL_CAPABILITY

    log_odds = (
        logit(base_probability)
        + sensitivity * capability_difference
    )

    probability = sigmoid(log_odds)

    return probability


def capability_gain(strategy, capability):
    """
    Calculate improvement IF the experiment succeeds.

    Mathematical equation:

        ΔA = gain_i * A * (1 - A)

    Components
    ----------

    gain_i:
        inherent improvement potential of the research strategy

    A:
        more capable AI researchers can exploit discoveries better

    (1 - A):
        diminishing returns

    The product

        A * (1 - A)

    creates an interesting shape.

    When A is low:
        the researcher itself is weak

    When A is moderate:
        improvement can happen quickly

    When A approaches 1:
        diminishing returns dominate
    """

    gain = strategy["gain"]

    delta_A = gain * capability * (1.0 - capability)

    return delta_A


def expected_reward(strategy, capability):
    """
    Calculate the TRUE expected immediate reward of a strategy.

    The agent does NOT know this function.

    It exists so that WE can inspect the simulated environment.

    Expected reward:

        E[r | strategy i, A]

        = P(success | i, A)
          *
          improvement_if_success

    Therefore:

        E[r]
        =
        p_i(A)
        *
        gain_i
        *
        A
        *
        (1 - A)
    """

    probability = research_success_probability(
        strategy,
        capability,
    )

    gain = capability_gain(
        strategy,
        capability,
    )

    return probability * gain


# =====================================================================
# 4. EPSILON-GREEDY RL POLICY
# =====================================================================

def choose_strategy(q_values, epsilon, rng):
    """
    Choose an R&D strategy using epsilon-greedy exploration.

    With probability epsilon:

        EXPLORE
        choose a random research strategy

    Otherwise:

        EXPLOIT
        choose the strategy with the highest estimated Q-value


    Returns
    -------

    action : int
        Index of selected strategy.

    mode : str
        Either "explore" or "exploit".
    """

    random_number = rng.random()

    # ---------------------------------------------------------------
    # Exploration
    # ---------------------------------------------------------------

    if random_number < epsilon:

        action = rng.integers(
            low=0,
            high=len(q_values),
        )

        mode = "explore"

        return action, mode

    # ---------------------------------------------------------------
    # Exploitation
    # ---------------------------------------------------------------

    # There can occasionally be ties between Q-values.
    #
    # Instead of always selecting the first strategy, randomly
    # choose among all strategies that have the maximum Q-value.

    max_q = np.max(q_values)

    candidate_actions = np.flatnonzero(
        np.isclose(q_values, max_q)
    )

    action = rng.choice(candidate_actions)

    mode = "exploit"

    return action, mode


# =====================================================================
# 5. RUN ONE R&D EXPERIMENT
# =====================================================================

def run_research_experiment(
    strategy,
    capability,
    rng,
):
    """
    Simulate one research experiment.

    Steps
    -----

    1. Calculate probability that research succeeds.

    2. Randomly determine whether experiment succeeds.

    3. If successful:
           improve AI capability.

       Otherwise:
           no capability improvement.

    4. Return reward = capability improvement.


    This creates stochastic feedback.

    The same research strategy can succeed in one generation and fail
    in another.
    """

    # ---------------------------------------------------------------
    # Probability that research succeeds
    # ---------------------------------------------------------------

    probability_success = research_success_probability(
        strategy,
        capability,
    )

    # ---------------------------------------------------------------
    # Draw random Bernoulli outcome
    # ---------------------------------------------------------------

    random_number = rng.random()

    success = random_number < probability_success

    # ---------------------------------------------------------------
    # Calculate actual improvement
    # ---------------------------------------------------------------

    if success:

        improvement = capability_gain(
            strategy,
            capability,
        )

    else:

        improvement = 0.0

    # Reward is simply improvement in capability.
    reward = improvement

    return success, probability_success, improvement, reward


# =====================================================================
# 6. MAIN SELF-IMPROVEMENT SIMULATION
# =====================================================================

def run_simulation():
    """
    Run the complete recursive self-improvement process.

    At each generation:

        1. Current AI has capability A_t

        2. RL policy chooses one R&D strategy

        3. AI executes research

        4. Experiment succeeds or fails

        5. Successful experiment changes A_t -> A_(t+1)

        6. RL algorithm observes reward

        7. Q-value of chosen strategy is updated

        8. Improved AI performs the next research cycle

    Returns
    -------

    results_df : pandas.DataFrame
        One row per research generation.

    q_history_df : pandas.DataFrame
        Evolution of learned Q-values.
    """

    # ---------------------------------------------------------------
    # Initialize random generator
    # ---------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    number_of_strategies = len(STRATEGIES)

    # ---------------------------------------------------------------
    # Initial AI capability
    # ---------------------------------------------------------------

    capability = INITIAL_CAPABILITY

    # ---------------------------------------------------------------
    # Initialize Q-values
    # ---------------------------------------------------------------

    q_values = np.full(
        number_of_strategies,
        INITIAL_Q_VALUE,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Initial epsilon
    # ---------------------------------------------------------------

    epsilon = EPSILON_START

    # ---------------------------------------------------------------
    # Storage for simulation history
    # ---------------------------------------------------------------

    results = []

    q_history = []

    # Store generation-zero Q-values.
    initial_q_row = {
        "generation": 0,
    }

    for i, strategy in enumerate(STRATEGIES):

        initial_q_row[strategy["name"]] = q_values[i]

    q_history.append(initial_q_row)

    # ===============================================================
    # Recursive R&D loop
    # ===============================================================

    for generation in range(
        1,
        N_GENERATIONS + 1,
    ):

        # -----------------------------------------------------------
        # STEP 1:
        # AI enters this generation with capability A_t
        # -----------------------------------------------------------

        capability_before = capability

        # -----------------------------------------------------------
        # STEP 2:
        # RL algorithm chooses research strategy
        # -----------------------------------------------------------

        action, decision_mode = choose_strategy(
            q_values=q_values,
            epsilon=epsilon,
            rng=rng,
        )

        strategy = STRATEGIES[action]

        # -----------------------------------------------------------
        # STEP 3:
        # Run actual R&D experiment
        # -----------------------------------------------------------

        (
            success,
            probability_success,
            improvement,
            reward,
        ) = run_research_experiment(
            strategy=strategy,
            capability=capability_before,
            rng=rng,
        )

        # -----------------------------------------------------------
        # STEP 4:
        # Update AI capability
        # -----------------------------------------------------------

        capability = capability_before + improvement

        # Because capability represents a normalized quantity,
        # ensure numerical errors never push it above 1.
        capability = min(
            capability,
            0.999999,
        )

        # -----------------------------------------------------------
        # STEP 5:
        # Reinforcement-learning Q update
        # -----------------------------------------------------------

        old_q = q_values[action]

        # Temporal-difference error:
        #
        #     reward - current estimate
        #
        prediction_error = reward - old_q

        # Bandit Q update:
        #
        #     Q <- Q + alpha * (reward - Q)
        #
        q_values[action] = (
            old_q
            + ALPHA * prediction_error
        )

        new_q = q_values[action]

        # -----------------------------------------------------------
        # STEP 6:
        # Compute TRUE expected reward for analysis
        # -----------------------------------------------------------

        # The AI does not receive these values.
        #
        # They allow us to compare:
        #
        #     what is actually best
        #
        # versus
        #
        #     what the agent currently believes is best.

        true_expected_rewards = np.array(
            [
                expected_reward(
                    s,
                    capability_before,
                )
                for s in STRATEGIES
            ]
        )

        true_best_action = np.argmax(
            true_expected_rewards
        )

        true_best_strategy = (
            STRATEGIES[
                true_best_action
            ]["name"]
        )

        # -----------------------------------------------------------
        # STEP 7:
        # Save everything from this generation
        # -----------------------------------------------------------

        result_row = {
            "generation": generation,

            "capability_before": capability_before,
            "capability_after": capability,

            "strategy": strategy["name"],
            "action_index": action,

            "decision_mode": decision_mode,

            "epsilon": epsilon,

            "probability_success": probability_success,

            "success": int(success),

            "reward_delta_A": reward,

            "Q_before": old_q,
            "Q_after": new_q,

            "prediction_error": prediction_error,

            "true_expected_reward_selected":
                true_expected_rewards[action],

            "true_best_strategy":
                true_best_strategy,

            "selected_true_best":
                int(action == true_best_action),
        }

        results.append(result_row)

        # -----------------------------------------------------------
        # Save current Q-values
        # -----------------------------------------------------------

        q_row = {
            "generation": generation,
        }

        for i, s in enumerate(STRATEGIES):

            q_row[s["name"]] = q_values[i]

        q_history.append(q_row)

        # -----------------------------------------------------------
        # STEP 8:
        # Reduce exploration gradually
        # -----------------------------------------------------------

        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY,
        )

    # ---------------------------------------------------------------
    # Convert results into pandas DataFrames
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    q_history_df = pd.DataFrame(
        q_history
    )

    return results_df, q_history_df


# =====================================================================
# 7. PRINT SIMULATION SUMMARY
# =====================================================================

def print_summary(results_df):
    """
    Print important statistics from the simulation.
    """

    print("\n")
    print("=" * 72)
    print("SELF-IMPROVING AI SIMULATION")
    print("=" * 72)

    initial_capability = INITIAL_CAPABILITY

    final_capability = results_df[
        "capability_after"
    ].iloc[-1]

    total_improvement = (
        final_capability
        - initial_capability
    )

    number_successes = results_df[
        "success"
    ].sum()

    success_rate = results_df[
        "success"
    ].mean()

    exploration_rate = (
        results_df["decision_mode"]
        == "explore"
    ).mean()

    optimal_strategy_rate = results_df[
        "selected_true_best"
    ].mean()

    print(
        f"\nInitial AI capability : "
        f"{initial_capability:.4f}"
    )

    print(
        f"Final AI capability   : "
        f"{final_capability:.4f}"
    )

    print(
        f"Total improvement     : "
        f"{total_improvement:.4f}"
    )

    print(
        f"\nSuccessful experiments: "
        f"{number_successes}"
    )

    print(
        f"Overall success rate  : "
        f"{success_rate:.2%}"
    )

    print(
        f"Exploration rate      : "
        f"{exploration_rate:.2%}"
    )

    print(
        f"True-best strategy selected: "
        f"{optimal_strategy_rate:.2%}"
    )

    # ----------------------------------------------------------------
    # Strategy-level statistics
    # ----------------------------------------------------------------

    strategy_summary = (
        results_df
        .groupby("strategy")
        .agg(
            experiments=("strategy", "size"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            average_reward=("reward_delta_A", "mean"),
            total_reward=("reward_delta_A", "sum"),
        )
        .sort_values(
            "total_reward",
            ascending=False,
        )
    )

    print("\n")
    print("=" * 72)
    print("RESULTS BY RESEARCH STRATEGY")
    print("=" * 72)

    print(
        strategy_summary.round(5)
    )

    print("\n")


# =====================================================================
# 8. PLOT 1: AI CAPABILITY OVER TIME
# =====================================================================

def plot_capability(results_df):
    """
    Show the main recursive self-improvement curve.
    """

    generations = np.concatenate(
        (
            [0],
            results_df["generation"].values,
        )
    )

    capability = np.concatenate(
        (
            [INITIAL_CAPABILITY],
            results_df[
                "capability_after"
            ].values,
        )
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        generations,
        capability,
        linewidth=2,
    )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "AI capability A"
    )

    plt.title(
        "Plot 1: Recursive growth of AI capability"
    )

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 9. PLOT 2: CAPABILITY IMPROVEMENT PER GENERATION
# =====================================================================

def plot_improvement_per_generation(results_df):
    """
    Show individual research breakthroughs.

    Many experiments fail and produce zero improvement.

    Successful experiments create discrete jumps.
    """

    rolling_reward = (
        results_df[
            "reward_delta_A"
        ]
        .rolling(
            window=15,
            min_periods=1,
        )
        .mean()
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        results_df["generation"],
        results_df["reward_delta_A"],
        alpha=0.45,
        label="Actual ΔA",
    )

    plt.plot(
        results_df["generation"],
        rolling_reward,
        linewidth=2.5,
        label="15-generation average ΔA",
    )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "Capability improvement ΔA"
    )

    plt.title(
        "Plot 2: Improvement produced by each research experiment"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 10. PLOT 3: LEARNED Q-VALUES
# =====================================================================

def plot_q_values(q_history_df):
    """
    Show what the RL agent believes each research strategy is worth.

    Q-values are learned entirely from observed experimental rewards.

    Higher Q-value:
        agent expects greater immediate capability improvement.
    """

    plt.figure(
        figsize=(12, 7)
    )

    for strategy in STRATEGIES:

        name = strategy["name"]

        plt.plot(
            q_history_df["generation"],
            q_history_df[name],
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "Learned Q-value"
    )

    plt.title(
        "Plot 3: RL agent learns the value of research strategies"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 11. PLOT 4: CUMULATIVE STRATEGY SELECTION SHARE
# =====================================================================

def plot_strategy_selection_share(results_df):
    """
    Show how the agent changes its research allocation.

    Example:

        if Fine-tuning becomes more attractive,
        its cumulative selection share should eventually increase.
    """

    plt.figure(
        figsize=(12, 7)
    )

    generations = results_df[
        "generation"
    ].values

    for strategy in STRATEGIES:

        name = strategy["name"]

        selected = (
            results_df["strategy"]
            == name
        ).astype(int)

        cumulative_count = (
            selected.cumsum()
        )

        cumulative_share = (
            cumulative_count
            / generations
        )

        plt.plot(
            generations,
            cumulative_share,
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "Cumulative share of experiments"
    )

    plt.title(
        "Plot 4: How RL reallocates research effort"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 12. PLOT 5: NUMBER OF TIMES EACH STRATEGY WAS SELECTED
# =====================================================================

def plot_strategy_counts(results_df):
    """
    Simple bar chart showing total research allocation.
    """

    counts = (
        results_df["strategy"]
        .value_counts()
        .reindex(
            [
                s["name"]
                for s in STRATEGIES
            ]
        )
    )

    plt.figure(
        figsize=(11, 6)
    )

    counts.plot(
        kind="bar"
    )

    plt.xlabel(
        "Research strategy"
    )

    plt.ylabel(
        "Number of experiments"
    )

    plt.title(
        "Plot 5: Total research experiments by strategy"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(
        True,
        axis="y",
    )

    plt.tight_layout()


# =====================================================================
# 13. PLOT 6: SUCCESS PROBABILITY OF CHOSEN RESEARCH
# =====================================================================

def plot_success_probability(results_df):
    """
    Compare predicted success probability with actual rolling success.

    As AI capability grows, research success probabilities tend to
    improve.
    """

    rolling_success = (
        results_df["success"]
        .rolling(
            window=20,
            min_periods=1,
        )
        .mean()
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        results_df["generation"],
        results_df["probability_success"],
        alpha=0.55,
        label="True P(success) for chosen strategy",
    )

    plt.plot(
        results_df["generation"],
        rolling_success,
        linewidth=2.5,
        label="20-generation observed success rate",
    )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "Probability / success rate"
    )

    plt.title(
        "Plot 6: Better AI becomes more effective at AI research"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 14. PLOT 7: TRUE EXPECTED REWARD LANDSCAPE
# =====================================================================

def plot_true_reward_landscape():
    """
    This plot exposes the hidden environment.

    The RL agent does NOT know these curves.

    They show the mathematically expected immediate reward of each
    strategy at different AI capability levels.

    This helps explain why the best R&D strategy can change as the
    AI becomes more capable.
    """

    capability_grid = np.linspace(
        INITIAL_CAPABILITY,
        0.99,
        300,
    )

    plt.figure(
        figsize=(12, 7)
    )

    for strategy in STRATEGIES:

        rewards = [
            expected_reward(
                strategy,
                capability,
            )
            for capability
            in capability_grid
        ]

        plt.plot(
            capability_grid,
            rewards,
            label=strategy["name"],
            linewidth=2,
        )

    plt.xlabel(
        "AI capability A"
    )

    plt.ylabel(
        "Expected ΔA per experiment"
    )

    plt.title(
        "Plot 7: Hidden expected payoff of each research strategy"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 15. PLOT 8: EXPLORATION VS EXPLOITATION
# =====================================================================

def plot_epsilon(results_df):
    """
    Show how random exploration decreases as the AI learns.

    The minimum exploration rate prevents the system from completely
    stopping experimentation.
    """

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        results_df["generation"],
        results_df["epsilon"],
        linewidth=2,
    )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "Exploration probability epsilon"
    )

    plt.title(
        "Plot 8: Exploration decreases as research knowledge accumulates"
    )

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 16. PLOT 9: FREQUENCY OF SELECTING TRUE BEST STRATEGY
# =====================================================================

def plot_optimal_strategy_rate(results_df):
    """
    Measure whether the RL researcher becomes better at choosing
    the currently best strategy.

    Because the best strategy may change as capability changes,
    this is a useful measure of research-policy quality.
    """

    rolling_optimal_rate = (
        results_df[
            "selected_true_best"
        ]
        .rolling(
            window=25,
            min_periods=1,
        )
        .mean()
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        results_df["generation"],
        rolling_optimal_rate,
        linewidth=2,
    )

    plt.xlabel(
        "Research generation"
    )

    plt.ylabel(
        "Fraction selecting true best strategy"
    )

    plt.title(
        "Plot 9: Is the AI learning to choose better research?"
    )

    plt.grid(True)

    plt.tight_layout()


# =====================================================================
# 17. SAVE DATA
# =====================================================================

def save_results(
    results_df,
    q_history_df,
):
    """
    Save detailed simulation results so they can be analyzed later
    with pandas, Excel, or another notebook.
    """

    results_df.to_csv(
        "self_improving_ai_results.csv",
        index=False,
    )

    q_history_df.to_csv(
        "self_improving_ai_q_values.csv",
        index=False,
    )

    print(
        "Saved:"
    )

    print(
        "  self_improving_ai_results.csv"
    )

    print(
        "  self_improving_ai_q_values.csv"
    )


# =====================================================================
# 18. CREATE ALL PLOTS
# =====================================================================

def create_all_plots(
    results_df,
    q_history_df,
):
    """
    Generate all explanatory plots.

    Each chart is deliberately kept in its own figure so that each
    mechanism can be inspected separately.
    """

    plot_capability(
        results_df
    )

    plot_improvement_per_generation(
        results_df
    )

    plot_q_values(
        q_history_df
    )

    plot_strategy_selection_share(
        results_df
    )

    plot_strategy_counts(
        results_df
    )

    plot_success_probability(
        results_df
    )

    plot_true_reward_landscape()

    plot_epsilon(
        results_df
    )

    plot_optimal_strategy_rate(
        results_df
    )

    # Display all figures.
    plt.show()


# =====================================================================
# 19. MAIN PROGRAM
# =====================================================================

def main():
    """
    Main execution function.
    """

    # ---------------------------------------------------------------
    # Run recursive AI R&D simulation
    # ---------------------------------------------------------------

    results_df, q_history_df = (
        run_simulation()
    )

    # ---------------------------------------------------------------
    # Print summary statistics
    # ---------------------------------------------------------------

    print_summary(
        results_df
    )

    # ---------------------------------------------------------------
    # Show first 15 generations
    # ---------------------------------------------------------------

    columns_to_display = [
        "generation",
        "capability_before",
        "strategy",
        "decision_mode",
        "probability_success",
        "success",
        "reward_delta_A",
        "capability_after",
        "Q_before",
        "Q_after",
    ]

    print(
        "=" * 72
    )

    print(
        "FIRST 15 RESEARCH GENERATIONS"
    )

    print(
        "=" * 72
    )

    print(
        results_df[
            columns_to_display
        ]
        .head(15)
        .round(5)
        .to_string(
            index=False
        )
    )

    print("\n")

    # ---------------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------------

    save_results(
        results_df,
        q_history_df,
    )

    # ---------------------------------------------------------------
    # Create explanatory plots
    # ---------------------------------------------------------------

    create_all_plots(
        results_df,
        q_history_df,
    )


# =====================================================================
# 20. EXECUTE PROGRAM
# =====================================================================

if __name__ == "__main__":
    main()