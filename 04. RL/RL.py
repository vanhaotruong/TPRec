

class RLAgent(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RLAgent, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Entity embeddings
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # Action space: all relations
        self.action_space = np.arange(num_relations)
        
        # State space: entity embeddings
        self.state_space = np.arange(num_entities)
        
        # Reward function
        self.reward_function = self.transE_reward
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def transE_reward(self, head, relation, tail):
        """TransE scoring function as reward"""
        head_emb = self.entity_embedding(head)
        relation_emb = self.relation_embedding(relation)
        tail_emb = self.entity_embedding(tail)
        
        score = torch.norm(head_emb + relation_emb - tail_emb, p=2)
        return -score  # Negative score for reward
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.greedy_action(state)
    
    def greedy_action(self, state):
        """Select greedy action"""
        head = state
        best_action = None
        best_reward = -np.inf
        
        for action in self.action_space:
            reward = self.reward_function(head, action, None)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action
    
    def train_agent(self, triples, num_epochs=100, learning_rate=0.01, epsilon=0.1):
        """Train the RL agent"""
        self.epsilon = epsilon
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_reward = 0
            
            # Shuffle triples for training
            np.random.shuffle(triples)
            
            for head, relation, tail in tqdm(triples, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Get state (head entity)
                state = torch.tensor([head], dtype=torch.long)
                
                # Select action (relation)
                action = self.select_action(state)
                
                # Calculate reward
                reward = self.reward_function(head, action, tail)
                
                # Update agent
                optimizer.zero_grad()
                reward.backward()
                optimizer.step()
                
                total_reward += reward.item()
            
            print(f"Epoch {epoch+1}: Average Reward = {total_reward / len(triples):.4f}")
    
    def save_embeddings(self, filepath):
        """Save entity and relation embeddings"""
        embeddings = {
            'entity_embedding': self.entity_embedding.weight.data.cpu().numpy(),
            'relation_embedding': self.relation_embedding.weight.data.cpu().numpy()
        }
        np.save(filepath, embeddings)
        print(f"Embeddings saved to {filepath}")

# Load dataset
triples = pd.read_csv('data/movie_triples.csv').values

# Initialize and train agent
num_entities = max(triples[:, 0].max(), triples[:, 2].max()) + 1
num_relations = triples[:, 1].max() + 1
embedding_dim = 50

agent = RLAgent(num_entities, num_relations, embedding_dim)
agent.train_agent(triples, num_epochs=100, learning_rate=0.01, epsilon=0.1)

# Save embeddings
agent.save_embeddings('embeddings/movie_rl_embeddings.npy')