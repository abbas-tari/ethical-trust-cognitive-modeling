# emotional_agent.py
import random
from ethical_trust_cognitive_modeling.agents.constants import EMOTIONS, INFLUENCE_STRATEGIES, BIG_FIVE_TRAITS

class EmotionalAgent:
    def __init__(self, id, trustworthiness, reliability, reputation, opinion):
        self.id = id
        self.trustworthiness = trustworthiness
        self.reliability = reliability
        self.reputation = reputation
        self.opinion = opinion
        self.trust = {}
        self.emotions = {emotion: random.random() for emotion in EMOTIONS}
        self.bias = random.random() * 2 - 1  # Cognitive bias
        self.memory = {}  # Memory of past interactions
        self.peers = []  # List of influencing agents
        self.influence_strategy = 'average'  # Default strategy
        self.threshold = 0.5  # Threshold for certain behaviors
        self.cognitive_load = 0.5  # Cognitive load
        self.personality = {trait: random.random() for trait in BIG_FIVE_TRAITS}
        self.friends = []  # Friends list
        self.resource_allocation = {}
        self.trust_history = []

    def record_trust(self):
        """Record current trust values."""
        self.trust_history.append(self.trust.copy())

    def allocate_resources(self, resource_pool, area_needs):
        """
        Allocate resources to different areas based on the agent's opinion and the needs of the area.
        """
        for area, need in area_needs.items():
            if self.opinion > self.threshold and resource_pool > 0:
                allocation = min(need, resource_pool)  # Allocate as much as needed or as much as possible
                self.resource_allocation[area] = self.resource_allocation.get(area, 0) + allocation
                resource_pool -= allocation
            else:
                self.resource_allocation[area] = self.resource_allocation.get(area, 0)


    def consult_memory(self, other_agent):
        """Consult memory to adjust behavior based on past interactions."""
        interactions = self.memory.get(other_agent.id, [])
        trust_decreases = sum(1 for i in range(1, len(interactions)) if interactions[i][1] < interactions[i-1][1])
        trust_increases = sum(1 for i in range(1, len(interactions)) if interactions[i][1] > interactions[i-1][1])

        # Negative Feedback Loop
        if trust_decreases >= 3:
            self.bias *= 1.5  # Increase bias temporarily

        # Positive Feedback Loop
        if trust_increases > 2:
            consecutive_positive = min(trust_increases, 5)  # Limit the maximum consecutive count to avoid too large values
            self.trust[other_agent.id] = min(self.trust[other_agent.id] * (1.1 ** consecutive_positive), 1)

    def update_memory(self, other_agent):
        """Update memory after an interaction."""
        if other_agent.id not in self.memory:
            self.memory[other_agent.id] = []
        self.memory[other_agent.id].append((self.opinion, self.trust.get(other_agent.id, 0.5)))
        self.memory[other_agent.id] = self.memory[other_agent.id][-5:]

    def update_trust(self, other_agent):
        if other_agent.id not in self.trust:
            self.trust[other_agent.id] = 0.5

        trust_update = 0.1 * other_agent.reputation + self.bias * (1 + self.cognitive_load)
        emotional_factor = (self.emotions['trust'] - self.emotions['fear']) * (1 + self.cognitive_load)

        # Taking personality into account
        # Openness increases trust update while conscientiousness decreases it
        trust_update += trust_update * 0.5 * (self.personality['openness'] - self.personality['conscientiousness'])
        
        self.trust[other_agent.id] = self.trust[other_agent.id] + trust_update * emotional_factor
        self.trust[other_agent.id] = max(min(self.trust[other_agent.id], 1), 0)

        # Only apply negative feedback if there is memory of past interactions
        if other_agent.id in self.memory and self.memory[other_agent.id]:
            last_trust_value = self.memory[other_agent.id][-1][1]
            trust_change = self.trust[other_agent.id] - last_trust_value
            if trust_change < -0.1:  # If there's a significant decrease in trust
                consecutive_negative = sum(1 for i in range(1, len(self.memory[other_agent.id])) if self.memory[other_agent.id][i][1] < self.memory[other_agent.id][i-1][1])
                # Apply a stronger penalty for consecutive negative interactions
                self.trust[other_agent.id] = max(self.trust[other_agent.id] * (0.9 ** consecutive_negative), 0)

    def interact(self, other_agent):
        """Interact with another agent based on personality."""
        # Extraversion determines the likelihood of interaction
        if random.random() < other_agent.reliability * (0.5 + 0.5 * self.personality['extraversion']):
            self.consult_memory(other_agent)
            self.update_trust(other_agent)
            self.update_opinion(other_agent)
            self.update_emotions(other_agent)
            self.update_memory(other_agent)

    def update_opinion(self, other_agent):
        trust_factor = self.trust.get(other_agent.id, 0.5)
        emotional_factor = (self.emotions['joy'] - self.emotions['sadness']) * (1 + self.cognitive_load)
        opinion_update = trust_factor * other_agent.opinion + self.bias * (1 + self.cognitive_load)
        # Agreeableness increases alignment of opinion with others
        opinion_update *= 1 + 0.5 * self.personality['agreeableness']
        
        self.opinion = (1 - trust_factor) * self.opinion + opinion_update * emotional_factor
        self.opinion = max(min(self.opinion, 1), 0)

    def update_emotions(self, other_agent):
        # Previous emotion state to detect sudden changes
        prev_opinion = self.opinion
        prev_trust = self.trust.get(other_agent.id, 0.5)

        # Calculate the magnitude of trust change
        trust_change = self.trust.get(other_agent.id, 0.5) - prev_trust
        magnitude_of_trust_change = abs(trust_change)

        # Dynamic Emotional Change based on magnitude of trust change
        if trust_change > 0.05:
            self.emotions['joy'] += magnitude_of_trust_change
            self.emotions['trust'] += magnitude_of_trust_change
        elif trust_change < -0.05:
            self.emotions['sadness'] += magnitude_of_trust_change
            self.emotions['disgust'] += magnitude_of_trust_change * (1 + self.personality['neuroticism'])

        # Inter-Emotion Dynamics
        if self.emotions['joy'] > 0.7:
            self.emotions['sadness'] = max(self.emotions['sadness'] - 0.1, 0)

        # Agent's Own Reputation and Reliability Impact
        if self.reputation < 0.3:
            self.emotions['fear'] += 0.1

        # Contextual Emotional Influence
        if other_agent.reputation < 0.3:
            self.emotions['disgust'] += 0.1 * (1 + self.personality['neuroticism'])

        # Surprise if opinion or trust significantly changes
        if abs(prev_opinion - self.opinion) > 0.1 or magnitude_of_trust_change > 0.1:
            self.emotions['surprise'] += 0.1

        # Anticipation based on other agent's reputation
        if other_agent.reputation > 0.7:
            self.emotions['anticipation'] += 0.1
        elif other_agent.reputation < 0.3:
            self.emotions['anticipation'] -= 0.1

        # Sadness if trust decreases significantly
        if trust_change < -0.1:
            self.emotions['sadness'] += 0.1

        # Random Fluctuations for unpredictability
        for emotion in EMOTIONS:
            self.emotions[emotion] += (random.random() - 0.5) * 0.1

        # Normalize emotions to ensure they are between 0 and 1
        for emotion in EMOTIONS:
            self.emotions[emotion] = max(min(self.emotions[emotion], 1), 0)
        
        # Update cognitive load based on emotions
        # Increase cognitive load for strong negative emotions and decrease for strong positive emotions
        if self.emotions['anger'] > 0.7 or self.emotions['fear'] > 0.7:
            self.cognitive_load += 0.1
        elif self.emotions['joy'] > 0.7 or self.emotions['trust'] > 0.7:
            self.cognitive_load -= 0.1

        # Ensure cognitive load is between 0 and 1
        self.cognitive_load = max(min(self.cognitive_load, 1), 0)

    def apply_social_influence(self, strategy=None):
        if strategy and strategy in INFLUENCE_STRATEGIES:
            self.influence_strategy = strategy

        if self.influence_strategy == 'None':
            return
        if self.influence_strategy == 'average':
            self._apply_average_influence()
        elif self.influence_strategy == 'majority':
            self._apply_majority_rule()
        elif self.influence_strategy == 'threshold':
            self._apply_threshold_model()
        elif self.influence_strategy == 'weighted':
            self._apply_weighted_influence()
        elif self.influence_strategy == 'random':
            self._apply_random_influence()
        elif self.influence_strategy == 'reinforcement':
            self._apply_reinforcement()
        elif self.influence_strategy == 'contrarian':
            self._apply_contrarian()

    def _apply_average_influence(self):
        avg_peer_opinion = sum(peer.opinion for peer in self.peers) / len(self.peers)
        influence_strength = 0.1
        
        # Adjust for agreeableness, making the agent more likely to align with the average
        influence_strength += influence_strength * (self.personality['agreeableness'] - 0.5)
        
        # Openness makes the agent more receptive to influence
        influence_strength += influence_strength * 0.5 * self.personality['openness']
        
        self.opinion = (1 - influence_strength) * self.opinion + influence_strength * avg_peer_opinion
        self.opinion = max(min(self.opinion, 1), 0)


    def _apply_majority_rule(self):
        positive_opinions = sum(1 for peer in self.peers if peer.opinion > 0.5)
        if positive_opinions > len(self.peers) / 2 :
            self.opinion = 1
        else:
            self.opinion = 0

    def _apply_threshold_model(self):
        differing_opinions = sum(1 for peer in self.peers if abs(peer.opinion - self.opinion) > 0.5)
        if differing_opinions > self.threshold * len(self.peers):
            self.opinion = 1 - self.opinion

    def _apply_weighted_influence(self):
        total_weight = sum(peer.reputation for peer in self.peers)
        weighted_opinion = sum(peer.opinion * peer.reputation for peer in self.peers) / total_weight
        self.opinion = weighted_opinion

    def _apply_random_influence(self):
        peer = random.choice(self.peers)
        self.opinion = peer.opinion

    def _apply_reinforcement(self):
        similar_opinions = sum(1 for peer in self.peers if peer.opinion == self.opinion)
        self.opinion += 0.1 * (similar_opinions / len(self.peers) - 0.5)

    def _apply_contrarian(self):
        positive_opinions = sum(1 for peer in self.peers if peer.opinion > 0.5)
        if positive_opinions > len(self.peers) / 2:
            self.opinion = 0
        else:
            self.opinion = 1

    def receive_external_event(self, event_type, event_impact):
        """
        React to an external event that can affect trust, opinion, emotions, or other factors.
        """
        if event_type == 'news':
            # Example: A breaking news event might influence the opinion of the agents
            self.opinion += event_impact
            self.opinion = max(min(self.opinion, 1), 0)
        elif event_type == 'environment_change':
            # Example: A significant environmental change might affect emotions such as fear or surprise
            self.emotions['fear'] += event_impact
            self.emotions['surprise'] += event_impact
            self.emotions['fear'] = max(min(self.emotions['fear'], 1), 0)
            self.emotions['surprise'] = max(min(self.emotions['surprise'], 1), 0)

        # Normalize emotions to ensure they are between 0 and 1
        for emotion in EMOTIONS:
            self.emotions[emotion] = max(min(self.emotions[emotion], 1), 0)

