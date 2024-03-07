import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        try:
            with open(f'{os.getcwd()}/{filename}.pickle', 'rb') as file:
                self.q = pickle.load(file)
            print("Loaded file: {}".format(filename+".pickle"))
        except:
            print("Could not open")

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        try:
            with open(f'{os.getcwd()}/{filename}.pickle', 'wb') as file:
                pickle.dump(self.q, file)
            print(f'Wrote to file: {os.getcwd()}/{filename}.pickle')
        except:
            print("Could not open")

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        
        if random.random() < self.epsilon:
            rand_val = random.random()
            num_actions = len(self.actions)
            action = sum([1 if rand_val > (float(i) / num_actions) else 0
                        for i in range(num_actions)]) - 1
            print("Num actions: {} | Random value: {} | Random action:{}".
                format(num_actions, rand_val, action))
            return action

        count = q.count(maxQ)
        if count > 1: 
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else: 
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:
            return action, q 
        return action

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)
        

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
