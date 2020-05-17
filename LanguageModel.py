from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner


class LanguageAgent(Agent):
    """ An agent that learns a communinty's vocabulary """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.vocabulary = []

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False, # implements Von Neumann neighborhood
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def speak(self):
        # TODO: Implement the language game between the agents
        # Speaks randomly to another agent on the same cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)

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

        # TODO: Impelement data collector and decide measures
        # self.datacollector = DataCollector(
        #    model_reporters={"Gini": compute_gini},
        #    agent_reporters={"Wealth": "wealth"}
        #)

    def step(self):
        # self.datacollector.collect(self)
        self.schedule.step()

# TODO: Add batch running to find overall patterns
