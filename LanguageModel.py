from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import random
import string

VOWELS = list("AEIOU")
CONSONANTS = list(set(string.ascii_uppercase) - set(VOWELS))

def compute_graph(model):
    agent_success = 0
    for agent in model.schedule.agents:
        agent_success += agent.communication_success
        # agent_success += random.random()
    N = model.num_agents
    return (agent_success / N)


class LanguageAgent(Agent):
    """ An agent that learns a communinty's vocabulary """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.meanings = []
        self.meaning2word = {}
        self.word2meaning = {}
        self.heading = [1,0]
        self.communication_success = 0

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False, # implements Von Neumann neighborhood
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.heading = [new_position[0] - self.pos[0], new_position[1] - self.pos[1]]
        self.model.grid.move_agent(self, new_position)

    def speak(self):
        # Speaks randomly to another agent on the same cell
        anticipated_meaning = None
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        # If other agents on the same cell
        if len(cellmates) > 1:
            hearer = self.random.choice(cellmates)
            meaning = self.random.choice(self.model.schedule.agents).unique_id

            # If the speaker is not acquainted with the meaning
            if meaning not in self.meanings:
                print("New meaning added to speaker")
                self.meanings.append(meaning)
                return 0.0

            # If the hearer is not acquainted with the meaning
            if meaning not in hearer.meanings:
                print("New meaning added to hearer")
                hearer.meanings.append(meaning)
                return 0.0

            # If the speaker has a word for the meaning
            if meaning in self.meaning2word:
                word = self.meaning2word[meaning]

                # If the hearer has a word for the meaning
                if word in hearer.word2meaning:
                    # If the hearer has no anticipated meaning
                    if not anticipated_meaning:
                        return 1.0
                    # If anticipated meaning different from hearer meaning
                    if (anticipated_meaning
                        and anticipated_meaning != hearer.word2meaning[word]):
                        hearer.word2meaning[word] = anticipated_meaning
                        hearer.meaning2word[anticipated_meaning] = word
                        hearer.meaning2word.pop(meaning, None)
                        return None
                    # If anticipated meaning same as hearer meaning
                    if (anticipated_meaning
                        and anticipated_meaning == hearer.word2meaning[word]):
                        return 1.0

                # If the hearer has no word for the meaning
                else:
                    # If anticipated meanig same as speaker meaning
                    if (anticipated_meaning
                        and word not in hearer.word2meaning
                        and anticipated_meaning not in hearer.meaning2word):
                        hearer.word2meaning[word] = anticipated_meaning
                        hearer.meaning2word[anticipated_meaning] = word
                    return 0.0

            # If the speaker has no word for the meaning
            if meaning not in self.meaning2word:
                if self.random.randrange(0, 101) <= 5:  # Probability of 5%
                    new_word = self.create_word()
                    print("New word:", new_word)
                    self.meaning2word[meaning] = new_word
                    self.word2meaning[new_word] = meaning
                return None

    # TODO: Complete function that creates new words
    def create_word(self):
        return self.random.choice(CONSONANTS) + self.random.choice(VOWELS)

    def step(self):
        self.move()
        self.speak()

class LanguageModel(Model):
    """ A model with variable number of agents who communicate. """
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False) # Last arg, if True makes grid toroidal
        self.schedule = RandomActivation(self)  # At each step, agents move in random order
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = LanguageAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        # TODO: Implement data collector and decide measures
        self.datacollector = DataCollector(
           model_reporters={"Graph": compute_graph},
           agent_reporters={"Meanings": "meanings"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# TODO: Add batch running to find overall patterns
