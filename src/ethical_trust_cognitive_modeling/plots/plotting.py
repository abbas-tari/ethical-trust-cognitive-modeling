# plotting.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from ethical_trust_cognitive_modeling.simulation.utils import set_seed

plt.switch_backend('TkAgg')

rc('text', usetex=False)
rc('font', family='serif')

def plot_emotional_data(opinions_over_time, avg_trust_over_time, emotions_over_time):
    set_seed()
    fig_size = (5.5, 4.5)
    fig1, ax1 = plt.subplots(figsize=fig_size)
    ax1.plot(opinions_over_time)
    ax1.set_title("Average Opinions Over Time")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Average Opinion")
    fig1.savefig("opinions_over_time.pdf")

    fig2, ax2 = plt.subplots(figsize=fig_size)
    ax2.plot(avg_trust_over_time, color='r')
    ax2.set_title("Average Trust Over Time")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Average Trust")
    fig2.savefig("avg_trust_over_time.pdf")

    fig3, ax3 = plt.subplots(figsize=fig_size)
    for emotion, values in emotions_over_time.items():
        ax3.plot(values, label=emotion)
    ax3.set_title("Average Emotions Over Time")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Average Emotion")
    ax3.legend(loc="upper right")
    fig3.savefig("emotions_over_time.pdf")
    plt.show()

def plot_friendship_network(agents):
    set_seed()
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id)
        for friend in agent.friends:
            G.add_edge(agent.id, friend.id)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(15, 12))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(agents)))
    node_colors = [colors[agent.id] for agent in agents]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    pos_labels = {node: (coords[0], coords[1] + 0.1) for node, coords in pos.items()}
    labels = {agent.id: "{}: {}".format(agent.id, ', '.join(str(friend.id) for friend in agent.friends)) for agent in agents}
    nx.draw_networkx_labels(G, pos_labels, labels=labels, font_size=9, font_weight='bold')
    plt.title("Friendship Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("friendship_network.pdf")
    plt.show()

def plot_opinions_and_trust_with_events(opinions_over_time, trust_over_time, events):
    set_seed()
    fig, ax1 = plt.subplots(figsize=(5.5, 4.5))
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average Opinion', color=color)
    ax1.plot(opinions_over_time, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Trust', color=color)
    ax2.plot(trust_over_time, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    for time_step, event in events.items():
        ax1.annotate(event, xy=(time_step, opinions_over_time[time_step]),
                     xytext=(time_step + 1000, opinions_over_time[time_step] - 0.05),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     fontsize=9, horizontalalignment='right', verticalalignment='bottom')
        ax2.plot(time_step, trust_over_time[time_step], 'ro')
    plt.title('Average Opinions and Trust Over Time with Event Annotations')
    fig.tight_layout()
    ax1.grid(True)
    plt.savefig("opinions_and_trust_with_events.pdf")
    plt.show()

def plot_emotion_heatmap_over_time(emotions_over_time):
    """
    Plot a heatmap showing the intensity of different emotions over time.

    :param emotions_over_time: A dictionary with emotion names as keys and lists of emotion intensities over time as values.
    """
    set_seed()
    # Extract the emotions and their values over time
    emotions = list(emotions_over_time.keys())
    emotion_values = np.array(list(emotions_over_time.values()))

    # Set up the figure
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    # Use imshow to plot the heatmap
    cax = ax.imshow(emotion_values, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set the tick marks for the y-axis
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels(emotions)
    
    # Set the x-axis to represent time
    time_steps = emotion_values.shape[1]
    tick_interval = max(1, time_steps // 10)  # Adjust this value as needed
    ax.set_xticks(np.arange(0, time_steps, step=tick_interval))
    ax.set_xticklabels(range(0, time_steps, tick_interval), rotation=45)  # Rotate labels for better visibility

    # Add a color bar to indicate the intensity scale
    fig.colorbar(cax, ax=ax, label='Intensity of Emotion')

    # Add titles and labels
    ax.set_title('Emotion Heatmap Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Emotions')
    
    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig("emotion_heatmap_over_time.pdf")

    # Show the plot
    plt.show()

def plot_trust_heatmap_at_iteration(agents, specific_iteration):
    set_seed()
    trust_matrix = np.zeros((len(agents), len(agents)))
    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            if i != j:
                trust_matrix[i, j] = agent_i.trust_history[specific_iteration].get(agent_j.id, 0)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(trust_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[agent.id for agent in agents],
                yticklabels=[agent.id for agent in agents],
                annot_kws={"size": 8})
    plt.title(f'Trust Dynamics at Iteration {specific_iteration}')
    plt.xlabel('Agent ID')
    plt.ylabel('Agent ID')
    plt.tight_layout()
    plt.savefig(f"trust_heatmap_iteration_{specific_iteration}.pdf")
    plt.show()

def plot_combined_networks(agents):

    """
    Plots two social network diagrams separately:
    - The first plot represents the friendship network with color-coded nodes.
    - The second plot represents the friendship and influence network with node sizes indicating influence.
    
    :param agents: A list of EmotionalAgent objects.
    """
    set_seed()
    # Create a single graph that will be used for both plots to ensure consistent node positions
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id)
        for friend in agent.friends:
            G.add_edge(agent.id, friend.id)

    # Calculate positions only once for consistency
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Figure size
    fig_size = (5.5, 4.5)

    # Plot 1: Friendship Network
    fig1, ax1 = plt.subplots(figsize=fig_size)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(agents)))
    node_colors = [colors[agent.id] for agent in agents]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    labels = {agent.id: "{}: {}".format(agent.id, ', '.join(str(friend.id) for friend in agent.friends)) for agent in agents}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    ax1.set_title("Friendship Network")
    plt.axis('off')
    plt.savefig("friendship_network.pdf")
    plt.show()

    # Plot 2: Influence Network
    fig2, ax2 = plt.subplots(figsize=fig_size)
    sizes = [agent.reputation * 1000 for agent in agents]  # Use reputation to determine the node size
    nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.8, node_color='blue')
    
    # Add weighted edges based on trust
    for agent in agents:
        for friend in agent.friends:
            avg_trust = (agent.trust.get(friend.id, 0) + friend.trust.get(agent.id, 0)) / 2
            G[agent.id][friend.id]['weight'] = avg_trust * 10  # Scale trust for visibility
    
    weights = nx.get_edge_attributes(G, 'weight').values()
    nx.draw_networkx_edges(G, pos, width=list(weights), alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    ax2.set_title('Social Network Diagram of Friendship and Influence')
    plt.axis('off')
    plt.savefig("influence_network.pdf")
    plt.show()

def plot_resource_allocation_map(allocation_records, iteration, area_needs, ax=None):
    """
    Visualizes the Resource Allocation Map for a given iteration.
    """
    set_seed()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure

    # Get the allocation records for the specified iteration
    allocations = allocation_records[iteration]
    
    # Convert the allocation records from agents to a 2D array
    allocation_array = np.array([[allocations[agent_id][area] for agent_id in allocations] for area in area_needs])

    cax = ax.matshow(allocation_array, cmap='viridis')
    fig.colorbar(cax, ax=ax)

    ax.set_xticks(range(len(allocations)))
    ax.set_xticklabels([agent_id for agent_id in allocations], rotation=90)
    ax.set_yticks(range(len(area_needs)))
    ax.set_yticklabels(['Area {}'.format(i) for i in area_needs])

    ax.set_title('Resource Allocation Map at Iteration {}'.format(iteration))
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Affected Area')

    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig("resource_allocation_map_iteration_{}.pdf".format(iteration))

    # Show the plot
    plt.show()