# simulation.py
import random
import numpy as np
from ethical_trust_cognitive_modeling.agents.emotional_agent import EmotionalAgent
from ethical_trust_cognitive_modeling.agents.constants import EMOTIONS
from ethical_trust_cognitive_modeling.simulation.utils import set_seed
# Assuming your plotting.py is in the plotting directory
from ethical_trust_cognitive_modeling.plots.plotting import (
    plot_emotional_data,
    plot_friendship_network,
    plot_opinions_and_trust_with_events,
    plot_emotion_heatmap_over_time,
    plot_trust_heatmap_at_iteration,
    plot_combined_networks,
    plot_resource_allocation_map
)

def create_emotional_agents(num_agents, num_friends=4, opinion_distribution='normal'):
    agents = []

    if opinion_distribution == 'normal':
        opinions = np.clip(np.random.normal(0.5, 0.15, num_agents), 0, 1)
    elif opinion_distribution == 'uniform':
        opinions = np.random.uniform(0, 1, num_agents)
    elif opinion_distribution == 'binomial':
        opinions = np.random.binomial(1, 0.5, num_agents)
    elif opinion_distribution == 'binary':
        opinions = np.random.choice([0, 1], num_agents)
    else:
        raise ValueError("Unsupported opinion distribution type.")

    for i in range(num_agents):
        trustworthiness = random.uniform(0.3, 0.7)
        reliability = random.uniform(0.4, 0.9)
        reputation = random.uniform(0.2, 0.8)
        opinion = opinions[i]
        agent = EmotionalAgent(i, trustworthiness, reliability, reputation, opinion)
        agents.append(agent)

    for agent in agents:
        agent.peers = [peer for peer in agents if peer.id != agent.id]
        agent.friends = random.sample(agent.peers, min(num_friends, len(agent.peers)))

    return agents

def simulate_emotional_interactions(agents, num_iterations, external_event_interval_news, external_event_interval_environment, influence_interval=1, friend_interaction_prob=0.6, external_event_interval=1000, resource_pool_size=1000, area_needs=None):
    opinions_over_time = []
    avg_trust_over_time = []
    emotions_over_time = {emotion: [] for emotion in EMOTIONS}
    allocation_records_over_time = []
    # Initialize the allocations record for each agent
    for agent in agents:
        agent.resource_allocation = {area: 0 for area in area_needs}

    for iteration in range(num_iterations):
        # Allocation of resources happens here for each iteration
        for agent in agents:
            agent.allocate_resources(resource_pool_size, area_needs)
            agent.record_trust()
        # Record the allocation state for this iteration
        allocation_records_over_time.append({agent.id: dict(agent.resource_allocation) for agent in agents})

        if random.random() < friend_interaction_prob:
            agent1 = random.choice(agents)
            agent2 = random.choice(agent1.friends)  # Agent selects a friend to interact with
        else:
            agent1, agent2 = random.sample(agents, 2)  # Random interaction

        agent1.interact(agent2)

        # Apply social influence after each interaction only if the iteration is a multiple of influence_interval
        if iteration % influence_interval == 0:
            for agent in agents:
                agent.apply_social_influence()

        # Trigger an external event at specific intervals
        if iteration == external_event_interval_news:
            trigger_external_event(agents, 'news', 0.2)  # Adjust news_impact

        if iteration == external_event_interval_environment:
            trigger_external_event(agents, 'environment_change', 0.2)  # Adjust environment_impact

        avg_opinion = sum([agent.opinion for agent in agents]) / len(agents)
        avg_trust = sum([sum(agent.trust.values()) / len(agent.trust) if agent.trust else 0 for agent in agents]) / len(agents)
        
        opinions_over_time.append(avg_opinion)
        avg_trust_over_time.append(avg_trust)
        
        for emotion in EMOTIONS:
            avg_emotion = sum([agent.emotions[emotion] for agent in agents]) / len(agents)
            emotions_over_time[emotion].append(avg_emotion)
    
    return opinions_over_time, avg_trust_over_time, emotions_over_time, agents, allocation_records_over_time  # agents are returned to access their allocations

def trigger_external_event(agents, event_type, event_impact):
    """ Trigger an external event that affects all agents. """
    for agent in agents:
        agent.receive_external_event(event_type, event_impact)

def run_simulation_with_strategy(agents, num_iterations, strategy, influence_interval=1, external_event_intervals=None, friend_interaction_prob=0.6):
    """ Run the simulation with a specific influence strategy. """
    strategy_results = {
        'opinions_over_time': [],
        'trust_over_time': []
    }

    initial_states = [agent.__dict__.copy() for agent in agents]

    for agent in agents:
        agent.apply_social_influence(strategy)

    for iteration in range(num_iterations):
        for agent in agents:
            if random.random() < friend_interaction_prob:
                if agent.friends:
                    friend = random.choice(agent.friends)
                    agent.interact(friend)
            else:
                other_agent = random.choice([other for other in agents if other != agent])
                agent.interact(other_agent)

            if iteration % influence_interval == 0:
                agent.apply_social_influence()

        if external_event_intervals:
            for event_type, interval in external_event_intervals.items():
                if iteration % interval == 0:
                    trigger_external_event(agents, event_type, event_impacts[event_type])

        avg_opinion = sum(agent.opinion for agent in agents) / len(agents)
        avg_trust = sum(sum(agent.trust.values()) / len(agent.trust) for agent in agents if agent.trust) / len(agents)
        
        strategy_results['opinions_over_time'].append(avg_opinion)
        strategy_results['trust_over_time'].append(avg_trust)

    for agent, state in zip(agents, initial_states):
        agent.__dict__.update(state)

    return strategy_results

def simulate_strategies(agents, num_iterations, strategies, influence_interval=1, external_event_intervals=None):
    """ Simulate each strategy and collect the average opinions and trust levels. """
    results = {}
    for strategy in strategies:
        results[strategy] = run_simulation_with_strategy(
            agents, num_iterations, strategy, influence_interval, external_event_intervals
        )
    return results

def reset_agents_to_initial_state(agents, initial_states):
    """ Reset agents to their initial state. """
    for agent, state in zip(agents, initial_states):
        agent.__dict__.update(state)

def generate_simulation_summary(results):
    """ Generate a summary of the simulation results. """
    summary = {
        'final_average_opinion': results['opinions_over_time'][-1],
        'final_average_trust': results['trust_over_time'][-1]
    }
    return summary

def setup_and_run_simulation(num_agents, num_iterations, strategies, opinion_distribution='normal'):
    """ Setup and run a new simulation. """
    agents = create_emotional_agents(num_agents, opinion_distribution=opinion_distribution)
    results = simulate_strategies(agents, num_iterations, strategies)
    summary = generate_simulation_summary(results)
    return agents, results, summary


def run_simulation():
    set_seed()
    num_agents = 10
    num_iterations = 5000
    influence_interval = 10
    opinion_distribution = 'normal'
    num_friends = 4  # Each agent will have this many friends
    friend_interaction_prob = 0.6
    area_needs = {i: random.randint(10, 100) for i in range(num_agents)}  # Example needs for each area
    resource_pool_size = 1000  # Example size of the total resource pool
    external_event_interval_news = 2500  # Iteration at which a news event happens
    external_event_interval_environment = 4000  # Iteration for environmental change

    agents = create_emotional_agents(num_agents, num_friends, opinion_distribution=opinion_distribution)
    opinions_over_time, avg_trust_over_time, emotions_over_time, _, allocation_records_over_time = simulate_emotional_interactions(
        agents, num_iterations, external_event_interval_news, external_event_interval_environment, influence_interval, friend_interaction_prob, resource_pool_size, area_needs = area_needs
    )

    # Plotting results
    plot_emotional_data(opinions_over_time, avg_trust_over_time, emotions_over_time)
    plot_friendship_network(agents)
    plot_opinions_and_trust_with_events(opinions_over_time, avg_trust_over_time, events={2500: 'News Event', 4000: 'Environmental Change'})
    plot_emotion_heatmap_over_time(emotions_over_time)
    plot_trust_heatmap_at_iteration(agents, specific_iteration=4000)  
    plot_combined_networks(agents)
    plot_resource_allocation_map(allocation_records_over_time, 4000, area_needs=area_needs)

# Run the simulation
run_simulation()