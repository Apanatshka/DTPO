from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax


@struct.dataclass
class MarkovDecisionProcessParams:
    """Encodes a discrete time, discrete action MDP using sparse successor lists.

    Instead of the dense `(state, next_state, action)` matrices, only the
    successors with non-zero transition probability are stored:

        next_states[s, a, k]   index of the k-th successor of (s, a)
        next_probs[s, a, k]    P(next_states[s, a, k] | s, a)
        next_rewards[s, a, k]  R(s, next_states[s, a, k], a)

    Every `(s, a)` row is padded out to `max_successors` (the largest number of
    successors over all state-action pairs) so the arrays keep static shapes and
    the environment stays jit/vmap-friendly. Padding entries point back at `s`
    and carry zero probability and zero reward, which makes them harmless if
    they are ever selected (e.g. for a row whose probabilities are all zero).

    Memory goes from O(n_states^2 * n_actions) to O(n_states * n_actions *
    max_successors), and a step is O(max_successors) instead of O(n_states).

    `absorbing` is precomputed at construction time: it marks the states that
    self-loop with probability one and zero reward for every action.
    """

    next_states: chex.Array  # (n_states, n_actions, max_successors), integer
    next_probs: chex.Array  # (n_states, n_actions, max_successors)
    next_rewards: chex.Array  # (n_states, n_actions, max_successors)
    initial_state_p: chex.Array  # (n_states,)
    observations: chex.Array  # (n_states, n_features)
    absorbing: chex.Array  # (n_states,), bool
    max_steps_in_episode: int = 1000


@struct.dataclass
class EnvState:
    state_index: int
    time: int


class MdpEnv(environment.Environment):
    @property
    def default_params(self) -> MarkovDecisionProcessParams:
        """Default environment parameters for Navigation3D."""
        return self.default_params_

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: MarkovDecisionProcessParams,
    ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        # Only the successors of (state, action) are touched, not all states.
        probs = params.next_probs[state.state_index, action]
        max_successors = probs.shape[0]

        # Inverse CDF sampling over the successor list. Sampling this way rather
        # than with jax.random.choice keeps a degenerate (all-zero) row from
        # producing NaNs: it then simply selects a zero-reward self-loop.
        cdf = jnp.cumsum(probs)
        u = jax.random.uniform(key) * cdf[-1]
        successor = jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, max_successors - 1)

        next_state_index = params.next_states[state.state_index, action, successor]
        reward = params.next_rewards[state.state_index, action, successor]

        next_state = EnvState(
            state_index=next_state_index,
            time=state.time + 1,
        )

        done = self.is_terminal(next_state, params)

        observation = params.observations[next_state_index]

        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(next_state),
            reward,
            done,
            {"discount": self.discount(next_state, params)},
        )

    def reset_env(self, key: chex.PRNGKey, params: MarkovDecisionProcessParams) -> tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        if params is None:
            params = self.default_params
        state_index = jax.random.choice(key, params.initial_state_p.shape[0], p=params.initial_state_p)
        observation = params.observations[state_index]
        state = EnvState(state_index=state_index, time=0)
        return observation, state

    def get_obs(self, state: EnvState) -> chex.Array:
        raise NotImplementedError()

    def is_terminal(self, state: EnvState, params: MarkovDecisionProcessParams) -> bool:
        # Whether a state only transitions to itself with zero reward is a
        # property of the MDP, so it is precomputed instead of scanned here.
        all_transitions_to_0_reward_state = params.absorbing[state.state_index]
        max_steps_reached = state.time == params.max_steps_in_episode

        return jnp.logical_or(all_transitions_to_0_reward_state, max_steps_reached)

    @property
    def name(self) -> str:
        """Environment name."""
        return self.name_

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_names)

    def action_space(self, params: MarkovDecisionProcessParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: MarkovDecisionProcessParams) -> spaces.Box:
        """Observation space of the environment."""
        low = params.observations.min(axis=0)
        high = params.observations.max(axis=0)
        return spaces.Box(low, high, shape=(params.observations.shape[1],), dtype=jnp.float32)

    def state_space(self, params: MarkovDecisionProcessParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "state_index": spaces.Discrete(params.observations.shape[0]),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def _absorbing_states(next_states: np.ndarray, next_probs: np.ndarray, next_rewards: np.ndarray) -> np.ndarray:
    """States that, for every action, transition to themselves with probability 1 and reward 0."""
    n_states = next_states.shape[0]

    self_loop = next_states == np.arange(n_states, dtype=next_states.dtype)[:, None, None]
    self_loop_p = np.where(self_loop, next_probs, 0).sum(axis=-1)
    # Padding entries also point at `s`, but they have zero probability so they
    # are excluded here.
    self_loop_has_reward = np.any(self_loop & (next_probs > 0) & (next_rewards != 0), axis=(1, 2))

    return np.all(np.isclose(self_loop_p, 1), axis=1) & ~self_loop_has_reward


def sparsify_mdp(
    trans_probs,
    rewards,
    initial_state_p,
    observations,
    max_steps_in_episode: int = 1000,
    dtype=jnp.float32,
    chunk_size: int | None = None,
) -> MarkovDecisionProcessParams:
    """Converts dense `(state, next_state, action)` matrices into a sparse MDP.

    The dense arrays are read in chunks of rows so that the conversion never
    needs a second copy of the full `n_states x n_states x n_actions` array, and
    they are never put on the accelerator.
    """
    trans_probs = np.asarray(trans_probs)
    rewards = np.asarray(rewards)

    assert trans_probs.ndim == 3, "trans_probs must have axes (state, next_state, action)"
    assert trans_probs.shape == rewards.shape
    n_states, n_next_states, n_actions = trans_probs.shape
    assert n_next_states == n_states

    if chunk_size is None:
        # Aim for ~4M elements per chunk so the temporary argsort stays small.
        chunk_size = max(1, min(n_states, 4_000_000 // max(1, n_states * n_actions)))

    # First pass: how wide does a successor list need to be?
    max_successors = 1
    for start in range(0, n_states, chunk_size):
        counts = (trans_probs[start : start + chunk_size] > 0).sum(axis=1)
        max_successors = max(max_successors, int(counts.max()))

    next_states = np.zeros((n_states, n_actions, max_successors), dtype=np.int32)
    next_probs = np.zeros((n_states, n_actions, max_successors), dtype=np.float64)
    next_rewards = np.zeros((n_states, n_actions, max_successors), dtype=np.float64)

    # Second pass: gather the non-zero successors of every (state, action).
    for start in range(0, n_states, chunk_size):
        stop = min(start + chunk_size, n_states)
        probs_block = trans_probs[start:stop]  # (chunk, n_states, n_actions)
        rewards_block = rewards[start:stop]

        nonzero = probs_block > 0
        # Stable argsort of the negated mask puts the non-zero successors first,
        # in ascending state order.
        order = np.argsort(~nonzero, axis=1, kind="stable")[:, :max_successors, :]
        valid = np.take_along_axis(nonzero, order, axis=1)

        # Padding points back at the source state with zero probability/reward.
        indices = np.where(valid, order, np.arange(start, stop, dtype=order.dtype)[:, None, None])
        probs = np.where(valid, np.take_along_axis(probs_block, order, axis=1), 0)
        step_rewards = np.where(valid, np.take_along_axis(rewards_block, order, axis=1), 0)

        # (chunk, max_successors, n_actions) -> (chunk, n_actions, max_successors)
        next_states[start:stop] = np.swapaxes(indices, 1, 2)
        next_probs[start:stop] = np.swapaxes(probs, 1, 2)
        next_rewards[start:stop] = np.swapaxes(step_rewards, 1, 2)

    absorbing = _absorbing_states(next_states, next_probs, next_rewards)

    return MarkovDecisionProcessParams(
        next_states=jnp.array(next_states, dtype=jnp.int32),
        next_probs=jnp.array(next_probs, dtype=dtype),
        next_rewards=jnp.array(next_rewards, dtype=dtype),
        initial_state_p=jnp.array(initial_state_p, dtype=dtype),
        observations=jnp.array(observations, dtype=dtype),
        absorbing=jnp.array(absorbing, dtype=bool),
        max_steps_in_episode=max_steps_in_episode,
    )


def check_mdp(mdp: MarkovDecisionProcessParams, check_jax_array_type=True):
    """
    Asserts that the shapes of the arrays inside the MDP are all valid and compatible.
    """
    if check_jax_array_type:
        assert isinstance(mdp.next_states, jax.numpy.ndarray)
        assert isinstance(mdp.next_probs, jax.numpy.ndarray)
        assert isinstance(mdp.next_rewards, jax.numpy.ndarray)
        assert isinstance(mdp.initial_state_p, jax.numpy.ndarray)
        assert isinstance(mdp.observations, jax.numpy.ndarray)
        assert isinstance(mdp.absorbing, jax.numpy.ndarray)

    # The successor arrays have axes: state, action, successor
    # observations has axes: state, feature
    assert len(mdp.next_states.shape) == 3
    assert mdp.next_probs.shape == mdp.next_states.shape
    assert mdp.next_rewards.shape == mdp.next_states.shape
    assert jnp.issubdtype(mdp.next_states.dtype, jnp.integer)

    n_states_ = mdp.next_states.shape[0]

    assert jnp.all((mdp.next_states >= 0) & (mdp.next_states < n_states_))
    assert jnp.all((mdp.next_probs >= 0) & (mdp.next_probs <= 1))
    row_sums = mdp.next_probs.sum(axis=-1)
    assert jnp.all(jnp.logical_or(jnp.isclose(row_sums, 1), jnp.isclose(row_sums, 0)))

    # The used entries of a successor list are packed at the front and strictly
    # increasing, so no successor is listed twice for the same (state, action).
    used = mdp.next_probs > 0
    assert jnp.all(used[..., 1:] <= used[..., :-1])
    both_used = used[..., 1:] & used[..., :-1]
    assert jnp.all(jnp.where(both_used, jnp.diff(mdp.next_states, axis=-1) > 0, True))

    assert mdp.initial_state_p.shape == (n_states_,)
    assert len(mdp.observations.shape) == 2
    assert mdp.observations.shape[0] == n_states_
    assert mdp.absorbing.shape == (n_states_,)


def remove_unreachable_states_mdp(mdp: MarkovDecisionProcessParams):
    """
    Returns a new MDP with unreachable states removed.

    Reachable states are determined using a depth first search to find all states
    reachable from the non-zero probability initial states.
    """
    next_states = np.asarray(mdp.next_states)
    next_probs = np.asarray(mdp.next_probs)
    n_states = next_states.shape[0]

    visited = set()
    stack = [int(x) for x in np.nonzero(np.asarray(mdp.initial_state_p))[0]]
    while stack:
        state = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        # Only the successor list of this state is scanned, not every state.
        for next_state in next_states[state][next_probs[state] > 0]:
            next_state = int(next_state)
            if next_state not in visited:
                stack.append(next_state)

    if len(visited) == n_states:
        return mdp

    print("Removed states:", n_states - len(visited))

    # Create a new MDP without all the unreachable states
    kept = np.array(sorted(visited))
    # Successors of kept states are reachable, so they all get a valid new index.
    remap = np.full(n_states, -1, dtype=np.int32)
    remap[kept] = np.arange(len(kept), dtype=np.int32)

    new_mdp = MarkovDecisionProcessParams(
        next_states=jnp.array(remap[next_states[kept]]),
        next_probs=mdp.next_probs[kept],
        next_rewards=mdp.next_rewards[kept],
        initial_state_p=mdp.initial_state_p[kept],
        observations=mdp.observations[kept],
        absorbing=mdp.absorbing[kept],
        max_steps_in_episode=mdp.max_steps_in_episode,
    )

    return new_mdp


class CustomMdp(MdpEnv):
    def __init__(self, mdp_file: str):
        self.name_ = Path(mdp_file).stem

        with np.load(mdp_file, allow_pickle=True) as data:
            ## Boswachter side:
            # observations=self._mdp.observations,
            # trans_probs=self._mdp.trans_probs,
            # rewards=self._mdp.rewards,
            # initial_state_probs=self._mdp.initial_state_probs,
            # terminal_states=np.array([self._mdp.terminal_states]),
            # feature_names=np.array(self.feature_names),
            # action_names=np.array(self.action_names),

            observations = data["observations"]
            trans_probs = data["trans_probs"]
            rewards = data["rewards"]
            initial_state_p = data["initial_state_probs"]

            # Boswachter has from, action, to; DTPO expects from, to, action
            trans_probs = np.swapaxes(trans_probs, 1, 2)
            rewards = np.swapaxes(rewards, 1, 2)

            # The dense matrices are converted straight to sparse successor
            # lists and dropped, so they never reach the device.
            self.default_params_ = sparsify_mdp(trans_probs, rewards, initial_state_p, observations)
            del trans_probs, rewards

            check_mdp(self.default_params_)
            self.feature_names = list(data["feature_names"])
            self.action_names = list(data["action_names"])

            self.obs_shape = (len(self.feature_names),)
