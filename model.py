from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from collections import namedtuple
import numpy as np
import random
import string
import math



VOWELS = list("AEIOU")
CONSONANTS = list(set(string.ascii_uppercase) - set(VOWELS))
R = 5 # word meaning pair change rate
ALPHA = 0.49 # sigmoid variable
BETA = 80 # sigmoid variable

Conversation = namedtuple("Conversation", ["word", "meaning", "success"])

def compute_graph(model):
    avg_success = np.mean([a.comm_success for a in model.schedule.agents])
    print("average success: ", avg_success)
    return avg_success


class LanguageAgent(Agent):
    dialog_count = 0
    """ An agent that learns a communinty's vocabulary """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.meanings = []
        self.meaning2word = {}
        self.word2meaning = {}
        self.wordsuccess = {}
        self.heading = self.random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.comm_success = 0
        self.number_of_dialogs = 0

    def create_link(self, word, meaning):
        """ Creates a coupling between word and meaning """
        print(str(self.unique_id) + " learned " + str(word) + " for " + str(meaning))
        self.meaning2word[meaning] = word
        self.word2meaning[word] = meaning
        self.wordsuccess[word] = []

        if meaning not in self.model.vocabulary:
            self.model.vocabulary[meaning] = {}

        # If word not in vocabulary, add it
        if word not in self.model.vocabulary[meaning]:
            self.model.vocabulary[meaning][word] = [self.unique_id]
        # Else append this agent to its users
        else:
            self.model.vocabulary[meaning][word].append(self.unique_id)


    def delete_link(self, word):
        """ Deletes a coupling between a word and a meaning """
        meaning = self.word2meaning[word]
        print(str(self.unique_id) + " forgot " + str(word) + " for " + str(meaning))
        del self.word2meaning[word]
        del self.meaning2word[meaning]
        del self.wordsuccess[word]

        # If the agent was the only one using the word, delete the word
        if len(self.model.vocabulary[meaning][word]) == 1:
            del self.model.vocabulary[meaning][word]
        # Else simply remove the agent
        else:
            self.model.vocabulary[meaning][word].remove(self.unique_id)

    def move(self):
        """ Implements the agents' movement """
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False, # implements Von Neumann neighborhood
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.heading = [new_position[0] - self.pos[0], new_position[1] - self.pos[1]]
        self.model.grid.move_agent(self, new_position)

    def speak(self):
        """ Implements the dialog between agents (Section 4.1.2 of the original paper) """
        # Speaks randomly to another agent on the same cell
        anticipated_meaning = None
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        # If other agents on the same cell
        if len(cellmates) > 1:
            hearer = self.random.choice(cellmates)

            while (hearer == self): # agents should not talk to themselves
                hearer = self.random.choice(cellmates)

            meaning = self.random.choice(self.model.schedule.agents).unique_id

            # If the speaker is not acquainted with the meaning
            if meaning not in self.meanings:
                print("New meaning added to speaker")
                self.meanings.append(meaning)
                return Conversation(word=None,meaning=None, success=0.0)

            # If the hearer is not acquainted with the meaning
            if meaning not in hearer.meanings:
                print("New meaning added to hearer")
                hearer.meanings.append(meaning)
                return Conversation(word=None, meaning=None, success=0.0)

            if self.random.random() <= 0.5: # 50% chance of having an anticipated meaning
                print("    " + str(self.unique_id) + " points at " + str(meaning))
                anticipated_meaning = meaning


            # If the speaker has a word for the meaning
            if meaning in self.meaning2word:
                word = self.meaning2word[meaning]

                # If the hearer has a word for the meaning
                if word in hearer.word2meaning:
                    # If the hearer has no anticipated meaning
                    if anticipated_meaning == None:
                        return Conversation(word=word, meaning=meaning, success=1.0)
                    # If anticipated meaning different from hearer meaning
                    if (anticipated_meaning != None
                        and anticipated_meaning != hearer.word2meaning[word]):
                        hearer.delete_link(word)
                        hearer.create_link(word, anticipated_meaning)
                        return None
                    # If anticipated meaning same as hearer meaning
                    if (anticipated_meaning != None
                        and anticipated_meaning == hearer.word2meaning[word]):
                        return Conversation(word=word, meaning=meaning, success=1.0)

                # If the hearer has no word for the meaning
                else:
                    # If anticipated meaning same as speaker meaning
                    if (anticipated_meaning != None
                        and word not in hearer.word2meaning
                        and anticipated_meaning not in hearer.meaning2word):
                        hearer.create_link(word, anticipated_meaning)
                    return Conversation(word=word, meaning=meaning, success=0.0)

            # If the speaker has no word for the meaning
            if meaning not in self.meaning2word:
                return Conversation(word=None, meaning=meaning, success=0.0)

    def do_change(self, success_array):
        """ Checks if a word-meaning coupling must be changed or not """
        avg = np.mean(success_array)
        # If average succes is zero drop the word
        if avg == 0:
            return True
        # If it is 1 keep it
        if avg == 1:
            return False
        # For cases in between we use the sigmoid function to decide
        probability = 1.0 / (1.0 + math.exp(4 * math.tan(math.radians(BETA*(avg - ALPHA)))))
        if self.random.random() < probability:
            return True
        else:
            return False

    def change_wordMeaning(self, conversation):
        """ After a conversation, implements the changes to relevant word - meaning coupling if needed """
        if conversation == None: return

        # If no word was used in the last conversation
        if conversation.word == None and conversation.meaning != None:
            if self.random.random() <= 0.05:  # Probability of 5%
                new_word = self.create_word()
                while new_word in self.wordsuccess: # cannot have one word with multiple meanings
                    new_word = self.create_word()
                print("New word:", new_word)
                self.create_link(new_word, conversation.meaning)

        # If a word was used in the last conversation
        elif conversation.word != None:
            self.wordsuccess[conversation.word].append(conversation.success)

            # if the word was used R times, there is a chance it will be dropped
            if len(self.wordsuccess[conversation.word]) >= R:
                if self.do_change(self.wordsuccess[conversation.word]):
                    self.delete_link(conversation.word) # forget word
                else:
                    self.wordsuccess[conversation.word] = [] # reset success

    def create_word(self):
        """ Creates a new word (1 consonant + 1 vowel) """
        return self.random.choice(CONSONANTS) + self.random.choice(VOWELS)

    def step(self):
        self.move()
        conversation = self.speak()
        # TODO: Refactor this out of the step method
        if conversation:
            self.number_of_dialogs += 1
            print(self.number_of_dialogs)
            self.comm_success = self.comm_success + (1/self.number_of_dialogs) * (conversation.success - self.comm_success)
        self.change_wordMeaning(conversation)


class LanguageModel(Model):
    """ A model with variable number of agents who communicate. """
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False) # Last arg, if True makes grid toroidal
        self.schedule = RandomActivation(self)  # At each step, agents move in random order
        self.running = True
        self.vocabulary = {}

        # Initialize a vocabulary. For each meaning it will collect the words used by the agents
        for e in range(N):
            self.vocabulary[e] = {}

        # Create agents
        for i in range(self.num_agents):
            a = LanguageAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
           model_reporters={"Average Communicative Success": compute_graph},
           agent_reporters={"Meanings": "meanings"}
        )

    def step(self):
        # TODO: Refactor the conditional out of this step method.
        self.datacollector.collect(self)
        total_dialogs = sum([a.number_of_dialogs for a in self.schedule.agents])
        # show global vocabulary every 50 iterations
        if self.schedule.time % 50 == 0:
            print("Dialog ", total_dialogs)
            self.showVocabulary()
        self.schedule.step()

    def showVocabulary(self):
        print("----------------")
        for e in self.vocabulary:
            text = str(e) + ": "
            for word in self.vocabulary[e]:
                text += str(word) + ":" + str(self.vocabulary[e][word]) + " "
            print(text)
        print("----------------")

# TODO: Add batch running to find overall patterns
