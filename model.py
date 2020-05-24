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
import itertools

VOWELS = list("AEIOU")
CONSONANTS = list(set(string.ascii_uppercase) - set(VOWELS))
STD_MEANINGS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
STD_WORDS = ['BA', 'CE', 'DI', 'FO', 'GU', 'HA', 'JE', 'KI', 'LO', 'MU']

Conversation = namedtuple("Conversation", ["word", "meaning", "success"])


def compute_graph(model):
    if len(model.success_array) == 0:
        return 0
    avg_success = np.mean(model.success_array)
    print("average success: ", avg_success)
    return avg_success


class LanguageAgent(Agent):
    dialog_count = 0
    """ An agent that learns a communinty's vocabulary """

    def __init__(self, unique_id, model, literate=False):
        super().__init__(unique_id, model)
        self.meanings = []
        self.meaning2word = {}
        self.word2meaning = {}
        self.wordsuccess = {}
        self.comm_success = 0
        # If the agent has initial language
        if literate:
            self.meanings = STD_MEANINGS
            for (meaning, word) in zip(STD_MEANINGS, STD_WORDS):
                self.meaning2word[meaning] = word
                self.word2meaning[word] = meaning
                self.wordsuccess[word] = [1.0] * 10
            self.comm_success = 1.0

        self.heading = self.random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.number_of_dialogs = 0

    def create_link(self, word, meaning):
        """ Creates a coupling between word and meaning """
        print(str(self.unique_id) + " learned " +
              str(word) + " for " + str(meaning))
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
        print(str(self.unique_id) + " forgot " +
              str(word) + " for " + str(meaning))
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
            moore=False,  # implements Von Neumann neighborhood
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.heading = [new_position[0] - self.pos[0],
                        new_position[1] - self.pos[1]]
        self.model.grid.move_agent(self, new_position)

    def speak(self):
        """ Implements the dialog between agents (Section 4.1.2 of the original paper) """
        # Speaks randomly to another agent on the same cell
        anticipated_meaning = None
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        # If other agents on the same cell
        if len(cellmates) > 1:
            hearer = self.random.choice(cellmates)

            while (hearer == self):  # agents should not talk to themselves
                hearer = self.random.choice(cellmates)

            meaning = self.random.choice(self.model.schedule.agents).unique_id

            # If the speaker is not acquainted with the meaning
            if meaning not in self.meanings:
                print("New meaning added to speaker")
                self.meanings.append(meaning)
                return Conversation(word=None, meaning=None, success=0.0)

            # If the hearer is not acquainted with the meaning
            if meaning not in hearer.meanings:
                print("New meaning added to hearer")
                hearer.meanings.append(meaning)
                return Conversation(word=None, meaning=None, success=0.0)

            # 50% chance of having an anticipated meaning default
            if self.random.random() <= self.model.antecipated_prob:
                print("    " + str(self.unique_id) +
                      " points at " + str(meaning))
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
        # If average success is zero drop the word
        if avg == 0:
            return True
        # If it is 1 keep it
        if avg == 1:
            return False
        # For cases in between we use the sigmoid function to decide
        probability = 1.0 / \
            (1.0 + math.exp(4 * math.tan(math.radians(self.model.beta*(avg - self.model.alpha)))))
        if self.random.random() < probability:
            return True
        else:
            return False

    def change_wordMeaning(self, conversation):
        """ After a conversation, implements the changes to relevant word - meaning coupling if needed """
        if conversation == None:
            return

        # If no word was used in the last conversation
        if conversation.word == None and conversation.meaning != None:
            if self.random.random() <= self.model.new_word_rate:  # Probability of 5% default
                new_word = self.create_word()
                while new_word in self.wordsuccess:  # cannot have one word with multiple meanings
                    new_word = self.create_word()
                print("New word:", new_word)
                self.create_link(new_word, conversation.meaning)

        # If a word was used in the last conversation
        elif conversation.word != None:
            self.wordsuccess[conversation.word].append(conversation.success)

            # if the word was used R times, there is a chance it will be dropped
            if len(self.wordsuccess[conversation.word]) >= self.model.change_rate:
                if self.do_change(self.wordsuccess[conversation.word]):
                    self.delete_link(conversation.word)  # forget word
                else:
                    self.wordsuccess[conversation.word] = []  # reset success

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
            self.comm_success = self.comm_success + \
                (1/self.number_of_dialogs) * \
                (conversation.success - self.comm_success)
            self.model.addSuccess(conversation.success)

        self.change_wordMeaning(conversation)


class LanguageModel(Model):
    """ 
    A model that simulates Language Emergence through Self-Organization. For more 
    information check the original paper: https://digital.csic.es/bitstream/10261/127969/1/Spatial%20Vocabulary.pdf
    """

    def __init__(self, n, literate, r, alpha, beta, new_word_rate, antecipated_prob, success_window, width, height):
        self.num_agents = n
        self.literate = literate
        self.change_rate = r
        self.alpha = alpha
        self.beta = beta
        self.new_word_rate = new_word_rate
        self.success_window = success_window
        self.antecipated_prob = antecipated_prob
        # Last arg, if True makes grid toroidal
        self.grid = MultiGrid(width, height, False)
        # At each step, agents move in random order
        self.schedule = RandomActivation(self)
        self.running = True
        self.vocabulary = {}
        self.success_array = []

        # Initialize a vocabulary. For each meaning it will collect the words used by the agents
        if literate > 0:
            for e in range(self.num_agents):
                self.vocabulary[e] = {STD_WORDS[e]: list(range(literate))}
        else:
            for e in range(self.num_agents):
                self.vocabulary[e] = {}

        # Create agents
        for i in range(self.num_agents):
            if i < self.literate:
                a = LanguageAgent(i, self, True)
            else:
                a = LanguageAgent(i, self, False)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "Average Success (of the last window of dialogs)": compute_graph},
            agent_reporters={"Meanings": "meanings"}
        )

    def step(self):
        # TODO: Refactor the conditional out of this step method.
        self.datacollector.collect(self)
        total_dialogs = sum(
            [a.number_of_dialogs for a in self.schedule.agents])
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

    def addSuccess(self, success):
        if len(self.success_array) + 1 >= self.success_window:
            del self.success_array[0]

        self.success_array.append(success)


# TODO: Add batch running to find overall patterns
