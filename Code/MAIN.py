"""
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Jan Nalivaika
#
# IG : @jannalivaika
# YT : Jan Nalivaika
# SC : @jannalivaika
# LinkedIn : Jan Nalivaika
# Mail : jan.nalivaika@siemens.com
# Mail : nalivaika@outlook.de

# PRODUCTION FLOW-CONTROL IN A CELL-BASED MANUFACTURING ENVIRONMENT

########################################################################################################################
# Importing libraries
import numpy as np  # For mathematical operations
import time  # time library to get time for benchmarking
from Agent import TD3  # importing Agent-Class from other file
from Buffer import ReplayBuffer  # importing Buffer-Class from other file
import pickle  # for saving information in separate files for later use
import random  # for Generating random stuff (could be replaced with numpy)


########################################################################################################################
#  Here are the Implementations of all the functions:
#  For Main-Function see below

def Create_Agent_Parameters():
    """
    # ##################################################################################################################
    # To generate all necessary Agent-parameters the factory hast to be created for reference.
    # (see in function "Factory_create()" for more information)
    #
    # "state_dim" is a scalar telling the Neural-Net how many inputs are expected == Length of "state_xxx"
    # Each input corresponds with a neuron
    #
    # The learning rate ("lr")  is how much the Action-values are changed when updating
    #
    # "max_action" dictates the range of activation of each neuron in the last layer
    # It ranges from (- max_action) to (max_action)
    #
    # "action_dim" describes how many output neurons there are.
    # Each product requires a signal telling what to do in the next time step.
    # The number of output neurons is defined by the amount of products multiplied
    # by the amount of machines plus 1. Each product cam be sent to every machine plus being injected into
    # a machine for work.
    # see function "extract_Actions()" for more specific details
    #
    # "exploration" is a representation of the random fluctuations introduced into the action generation.
    # see function "Randomise_Action()" for more information
    #
    # ##################################################################################################################
    """
    # Initialising factory. See "Factory_create()"
    ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, done, score, Machine_Failure_Counter, Machine_Failure_Info = create_factory()

    # Compressing information
    Information = ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, done, Machine_Failure_Counter, Machine_Failure_Info

    # Generating State-Vector
    # See function GenerateState() for more information
    state_prior = GenerateState(*Information)

    # Getting state dimension to create input neurons. (1 input == 1 Neuron)
    state_dim = len(state_prior)

    lr = 0.00025  # Fine-tuning here ! 0,001/0.00025

    max_action = 1  # Fine-tuning here ! , range of possible activation

    action_dim = len(ProductBucket) * (len(WorkingTime) + 1)

    # See function "Randomise_Action()" for more information about the usage of "exploration"

    exploration_noise_max = 0.0001  # Fine-tuning here ! 0,003

    exploration_noise_min = 0.0001

    exploration_noise_decay = 0.9995  # 0.995

    """
    # #########################################################################
    # For optimal results all parameters should be fine-tuned !!
    # #########################################################################
    """
    return lr, state_dim, action_dim, max_action, exploration_noise_max, exploration_noise_min, exploration_noise_decay


def Create_Update_Parameters():
    """

    # ##################################################################################################################
    #
    # "gamma" describes what reward is preferred
    # gamma = 1 only long term rewards
    # gamma = 0 only instant short term rewards, no optimization for the future
    #
    #  batch_size              num of transitions sampled from replay buffer
    #  polyak                  target policy update parameter (1-tau)
    #  policy_noise            target policy smoothing noise
    #  noise_clip
    #  policy_delay            delayed policy updates parameter
    #
    #
    # ##################################################################################################################
    """

    gamma = 0.99  # 0.99 discount for future rewards

    batch_size = 100  # num of transitions sampled from replay buffer

    """
    It has been observed in practice that when using a larger batch there is a significant degradation
    in the quality of the model, as measured by its ability to generalize.
    """

    polyak = 0.995

    policy_noise = 0.0001

    noise_clip = 0.3

    policy_delay = 2

    return batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay


def create_factory():
    # Structure of the individual matrices is explained in the creation-function

    def create_WorkingTime(amount_of_machines, min_workingtime, max_workingtime,
                           amount_of_machines_with_multiple_skills, amount_of_extra_skills_on_over_skilled_machines):

        """
        # WT = WorkingTime
        # row index - representing a machine
        # column-index - Step (if NOT None, machine can perform this step)
        # column value - Working-Time in time units, how long the machine needs to perform the step

        # Example:
        # [4,None,None,None,None]   Machine 1 capable performing step 1 in 4 time units
        # [None,5,None,7,None]      Machine 2 capable performing step 2 in 5 time units and step 4 in 7 time units
        # [None,None,5,None,None]   Machine 3 capable performing step 3 in 5 time units
        # [None,8,None,4,None]      Machine 4 capable performing step 4 in 4 time units and step 2 in 8 time units
        # [None,None,None,None,3]   Machine 5 capable performing step 5 in 3 time units
        """

        WT = [[None for col in range(amount_of_machines)] for row in range(amount_of_machines)]

        for x in range(amount_of_machines):
            # Matrix is filled with numbers in range from Min to Max Working-Time
            WT[x][x] = random.randint(min_workingtime, max_workingtime)

        """
        # ##############################################################################################################
        # To generate a more unique factory, limitations can be set regarding:
        # 
        # - the possibility of the first and last step to be performed on other machines instead of
        #   only the first and last machine
        #
        # - the possibility of the first and last machine to have extra-skills 
        #
        #
        # ##############################################################################################################
        """

        can_first_and_last_machine_be_replaced = True
        does_first_and_last_machine_have_extra_skills = True

        if can_first_and_last_machine_be_replaced:
            # A list of all possible extra-skills is created including fist and last machine
            possible_extra_skills_on_over_skilled_machines = list(range(0, amount_of_machines))
        else:
            # A list of all possible extra-skills is created NOT including fist and last machine
            possible_extra_skills_on_over_skilled_machines = list(range(1, amount_of_machines - 1))

        if does_first_and_last_machine_have_extra_skills:
            # A list of all possible machines with extra-skills is created including fist and last machine
            possible_machines_with_multiple_skills = list(range(0, amount_of_machines))
        else:
            # A list of all possible machines with extra-skills is created, NOT including fist and last machine
            possible_machines_with_multiple_skills = list(range(1, amount_of_machines - 1))

        # Out of the just created list the previously set amount of elements is selected
        # These elements (machines) are the candidates for getting more skills assigned

        chosen_machines_with_multiple_skills = random.choices(possible_machines_with_multiple_skills,
                                                              k=amount_of_machines_with_multiple_skills)

        # on the same principle a list of selected skills is created

        chosen_extra_skills_on_over_skilled_machines = random.choices(possible_extra_skills_on_over_skilled_machines,
                                                                      k=amount_of_extra_skills_on_over_skilled_machines)

        # looping over machines with extra-skills
        for x in range(amount_of_machines_with_multiple_skills):

            # A chosen machine is selected to get extra-skills
            chosen_machine = chosen_machines_with_multiple_skills[x]

            # looping over extra-skills
            for y in range(amount_of_extra_skills_on_over_skilled_machines):

                # A extra-skill is selected to be assigned to the selected machine
                chosen_extra_skill = chosen_extra_skills_on_over_skilled_machines[y]

                # if the chosen extra-skill is not the same as the skill the machine already has...
                if chosen_extra_skill is not chosen_machine:
                    # ...the skill is assigned.
                    # the original machine should be more effective
                    # than the machine with the same skill as a extra-skill.
                    # Therefore a scalar in range from 5 to 10 is added to the original working-time

                    WT[chosen_machine][chosen_extra_skill] = WT[chosen_extra_skill][
                                                                 chosen_extra_skill] + random.randint(5, 10)

        """
        # ##############################################################################################################
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # ##############################################################################################################
        """

        WT = [[2, 5, None, None, None, None, None],
              [10, 2, 4, None, None, None, None],
              [None, 10, 2, None, None, None, None],
              [None, None, 10, 2, None, None, None],
              [None, None, None, 10, 2, 4, None],
              [None, None, None, None, 10, 2, 4],
              [None, None, None, None, None, 10, 2]]

        WT = [[2, 5, None, None, None],
              [10, 2, 4, None, None],
              [None, 10, 2, 4, None],
              [None, None, 10, 2, 4],
              [None, None, None, 5, 2]]

        return WT

    def create_ProductDesign(amount_of_products, amount_of_machines, skipped_working_steps_per_product):

        """
        # PD - Product Design
        # row index - representing one product
        # column-index - work type (Working steps have to be executes sequentially !!!!)
        # column-value - is work necessary for the product - this will change as step is done for the product
        #
        # Example:
        #
        # [1,1,None,1,1,1,1]                         Product 1 , needs all steps except the third
        # [1,1,1,1,1,1,1]                            Product 2 , needs all steps
        # [1,1,1,1,1,1,None]                         Product 3 , needs all steps except the last
        # [1,1,None,1,1,1,1]                         Product 4 , needs all steps except the third
        # [1,1,1,1,1,1,1]                            Product 5 , needs all steps
        # [None,None,None,None,None,None,None]       Product 6 , needs no steps; Everything completed
        """

        # Matrix is created (completely filled with 1)
        PD = [[1 for col in range(amount_of_machines)] for row in range(amount_of_products)]

        """
        # ##############################################################################################################
        # Unique production-batch not implemented yet !!!
        # skipped_working_steps_per_product NOT USED ATM 
        # !!!
        # ##############################################################################################################
        """

        return PD

    def create_TravelTime(amount_of_machines, min_transportationtime, max_transportationtime):

        """
        # TT - travel time
        # row-index - source machine
        # column-index - target machine
        # value - travel time from source to target
        #
        # Matrix is created
        # size: amount_of_machines x amount_of_machines
        #
        #
        #
        # Example:
        #
        # [6,3,7,5,4]
        # [4,6,3,7,9]
        # [3,7,3,5,7]
        # [3,6,4,1,9]
        # [2,5,3,8,6]
        """

        # Matrix is filled randomly
        TT = np.random.randint(min_transportationtime, max_transportationtime, (amount_of_machines, amount_of_machines))

        """
        # ##############################################################################################################
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # MANUAL INPUT !!! Make sure dimensions are corresponding !!!
        # ##############################################################################################################
        """

        TT = [[None, 2, 4, 4, 4, 4, 4],
              [1, None, 2, 4, 4, 4, 4],
              [4, 1, None, 2, 4, 4, 4],
              [4, 4, 1, None, 2, 4, 4],
              [4, 4, 4, 1, None, 2, 4],
              [4, 4, 4, 4, 1, None, 2],
              [4, 4, 4, 4, 4, 1, None]]

        TT = [[None, 2, 4, 4, 4],
              [1, None, 2, 4, 4],
              [4, 1, None, 2, 4],
              [4, 4, 1, None, 2],
              [4, 4, 4, 1, None]]

        return TT

    def create_ProductBucket(amount_of_products, amount_of_machines):

        """
        # PB - product bucket
        # index - product
        # value - is number of bucket (0 to x) where the product is located
        #         None == not in any bucket
        #
        #         NOT YET IMPLEMENTED:   -x == in a certain machine
        #                                -0 is still 0 !!! is it necessary ?
        # Current state:
        # Whether a product is inside a machine is only visible in the RWT-matrix!
        #
        # Example :
        # [None, 1, 4, -2, 1, 3]
        #
        # Product 1 is in not in front of any machine (Moving)
        # Product 2 is in front of the 1st machine (---or inside---)
        # Product 3 is in front of the 4th machine (---or inside---)
        # Product 4 is in inside the 2nd machine (being worked on) NOT IMPLEMENTED YET !
        # Product 5 is in front of the 1st machine (---or inside---)
        # Product 6 is in front of the 3rd machine (---or inside---)

        #
        #  Filling the matrix with Zeros == All products are in front of the fist (0th) machine
        #
        """

        PB = [0 for col in range(amount_of_products)]

        # To make it harder for the Agent the position of the products is randomised
        PB = [random.randint(0, amount_of_machines - 1) for col in range(amount_of_products)]

        return PB

    def create_RemainingWorkingTime(amount_of_machines):

        """
        # RWT remaining working time
        # row - machine
        # column 1 - product index
        # column 2 - Remaining Working time of product in that machine
        # column 3 - Working step , to identify that for example step 2 was done on 3d machine
        #
        # Example:
        #
        # [2,4,5]           Machine 1,  is working on product 2, left time-steps to work = 4, doing step 5
        # [None,None,None]  Machine 2, not performing any work
        # [None,None,None]  Machine 3, not performing any work
        # [None,None,None]  Machine 4, not performing any work
        #
        # Filling the matrix (empty / with None)
        """

        RWT = [[None for col in range(3)] for row in range(amount_of_machines)]
        return RWT

    def create_EstimatetTimeOfArrival(amount_of_products):

        """
        # ETA - estimated time arrival
        # row - product
        # column 1 - target machine
        # column 2 - ETA
        #
        #
        #
        #
        # Example
        # [   2,   4]   Product 1, traveling to machine 2, will arrive in 4 time steps
        # [None,None]   Product 2, stationary
        # [3   ,   5]   Product 3, traveling to machine 3, will arrive in 5 time steps
        # [None,None]   Product 4, stationary
        #
        # Filling the matrix (empty)
        """

        ETA = [[None for col in range(2)] for row in range(amount_of_products)]
        return ETA

    def create_Machine_Failure_Counter(WorkingTime):

        """
        # ##############################################################################################################
        # Machine Failure is one of the most important information.
        # current Rules:
        # 1) A machine can only fail while being empty
        #
        # Machine_Failure_Counter is a vector indicating how long the machine is not capable of operating
        #
        # Machine_Failure_Info is a matrix like "WorkingTime" it saves what skills have been taken from the machine,
        # while in Failure-Mode
        #
        # Examples:
        # Machine_Failure_Counter = [Monem, None, 6, None, None, 3]
        # Machine 1 = Able to work
        # Machine 2 = Able to work
        # Machine 3 = Failed !! Time to recovery = 6 time units
        # Machine 4 = Able to work
        # Machine 5 = Able to work
        # Machine 6 = Failed !! Time to recovery = 3 time units
        #
        # Machine_Failure_Info:
        # [[None,None,None,None,None,None]
        # [None,None,None,None,None,None]
        # [None,   10,  5,   7,None,None]     Skills af machine 3 are deleted fom WT and inserted here
        # [None,None,None,None,None,None]
        # [None,None,None,None,None,None]
        # [None,None,None,None,   9,   3]]    Skills af machine 6 are deleted fom WT and inserted here
        #
        # Machine_Failure_Info is like a storage container to remember what skills the machine originally had
        #
        # ##############################################################################################################
        """

        Machine_Failure_Counter = [None for col in range(len(WorkingTime))]

        Machine_Failure_Info = [[None for col in range(len(WorkingTime))] for row in range(len(WorkingTime))]

        return Machine_Failure_Counter, Machine_Failure_Info

    # ##################################################################################################################
    # Here are the necessary hyper-parameters for the creation of a factory

    amount_of_products = 5
    amount_of_machines = 5  # don't change, MANUAL INPUT !!!!!

    # Min and Max Transportation times are set for the Transportation-Matrix form machine to machine
    min_transportationtime = 2
    max_transportationtime = 3

    # Min and Max Working time for the machines are set
    min_workingtime = 2
    max_workingtime = 3

    # Amount of machines with multiple skills (over-skilled) are set
    # has to be lower or equal to amount_of_machines
    amount_of_machines_with_multiple_skills = 0

    # Amount of extra skills for each over-skilled machines is set
    # has to be lower or equal to amount_of_machines
    amount_of_extra_skills_on_over_skilled_machines = 0

    # uniqueness of first and last machine and
    # skill of first and last machine are separately set in create_WorkingTime()

    # How many step are canceled for each product (to create individualism)
    skipped_working_steps_per_product = 0

    # Creation of WorkingTime-Matrix
    WorkingTime = create_WorkingTime(amount_of_machines, min_workingtime, max_workingtime,
                                     amount_of_machines_with_multiple_skills,
                                     amount_of_extra_skills_on_over_skilled_machines)
    # Creation of ProductDesign-Matrix
    ProductDesign = create_ProductDesign(amount_of_products, amount_of_machines, skipped_working_steps_per_product)
    # Creation of TravelTime-Matrix
    TravelTime = create_TravelTime(amount_of_machines, min_transportationtime, max_transportationtime)
    # Creation of ProductBucket-Array
    ProductBucket = create_ProductBucket(amount_of_products, amount_of_machines)
    # Creation of RemainingWorkingTime-Matrix
    RemainingWorkingTime = create_RemainingWorkingTime(amount_of_machines)
    # Creation of EstimatedTimeOfArrival-Matrix
    EstimatedTimeOfArrival = create_EstimatetTimeOfArrival(amount_of_products)
    # Creation of Failure Information
    Machine_Failure_Counter, Machine_Failure_Info = create_Machine_Failure_Counter(WorkingTime)
    # Creation of "done" indicator
    done = 0

    # Creation of score-value, representing the rating of each single run.
    score = 0

    return ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, done, score, Machine_Failure_Counter, Machine_Failure_Info


def factory_step(ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket,
                 done, Machine_Failure_Counter, Machine_Failure_Info, Action, step, max_timesteps, pre_done):
    def work(RemainingWorkingTime):

        for x in range(len(RemainingWorkingTime)):  # loop checking every machine-Status

            if RemainingWorkingTime[x][1] is not None:  # if time left to work is not None
                RemainingWorkingTime[x][1] -= 1  # reduce working time by 1

            if RemainingWorkingTime[x][1] == 0:  # if remaining working time is 0
                RemainingWorkingTime[x][1] = None  # set remaining working time to None

        """
        # ##############################################################################################################
        #
        # [    2,    3,    4]         [    2,    2,    4] <----
        # [None , None, None]  - - - >[None , None, None]
        # [    4,    1,    1]  - - - >[    4, None,    1] <----
        # [None , None, None]         [None , None, None]
        #
        # Machine 1:
        # [    2,    3,    4]
        # is working on product 2
        # 3 time-steps is left to work
        # working-step currently worked on is Nr. 4
        #
        # After 1 time-steps:
        # [    2,    2,    4]
        # Time left to work is reduced by 1
        #
        # ##############################################################################################################
        """

        return RemainingWorkingTime

    def eject(RemainingWorkingTime, ProductBucket, ProductDesign):

        for x in range(len(RemainingWorkingTime)):  # loop checking every machine-Status
            # x is the machine number
            if RemainingWorkingTime[x][0] is not None:  # if product is in a machine
                if RemainingWorkingTime[x][1] is None:  # if remaining time is set to "None" (enough time passed)
                    product = RemainingWorkingTime[x][0]  # Find what product has been worked on
                    step = RemainingWorkingTime[x][2]  # Find what step has been worked on
                    RemainingWorkingTime[x][0] = None  # set values to "None"
                    RemainingWorkingTime[x][2] = None  # set values to "None"
                    ProductDesign[product][step] = None  # set completed step in PD to "None"
                    ProductBucket[product] = x  # set position in PB to machine (eject product)

        """
        # ##############################################################################################################
        # Remaining working Time:
        # [    2,    2,    4]      [    2,    1,    4]
        # [None , None, None] ---> [None , None, None]
        # [    4, None,    1] ---> [None , None, None] <----
        # [None , None, None]      [None , None, None]
        #
        #
        #
        # Explanation:
        #
        # [    4, None,    1] ---> [None , None, None]
        #
        # Machine 3
        # worked on product 4
        # executed work-step 1
        #
        #
        # Product-Bucket:
        # [3,None,6,-3,None] --> [3,None,6,3,None]
        #
        # Product 4 was ejected
        #
        # Product design:
        # [None,None,None,None,   1]       [None,None,None,None,   1]
        # [None,None,None,   1,   1]       [None,None,None,   1,   1]
        # [None,None,None,None,   1]  ---> [None,None,None,None,   1]
        # [   1,   1,   1,   1,   1]       [None,   1,   1,   1,   1]  <----
        # [   1,   1,   1,   1,   1]       [   1,   1,   1,   1,   1]
        # ##############################################################################################################
        """

        return RemainingWorkingTime, ProductBucket, ProductDesign

    def travel(EstimatedTimeOfArrival, ProductBucket):

        for x in range(len(ProductBucket)):
            # x = Product
            if EstimatedTimeOfArrival[x][1] is not None:  # If time to Destination in not None
                EstimatedTimeOfArrival[x][1] -= 1  # reduce time left to travel by 1

            if EstimatedTimeOfArrival[x][1] == 0:  # If time to Destination in 0, Product has arrived
                EstimatedTimeOfArrival[x][1] = None  # Set time left to travel to "None"
                target = EstimatedTimeOfArrival[x][0]  # Find target-machine, where product was moving
                EstimatedTimeOfArrival[x][0] = None  # Set target-machine to "None"
                ProductBucket[x] = target  # Set Position of product to target-machine

        """
        # ##############################################################################################################
        # EstimatedTimeOfArrival:
        # [    2,    3]      [    2,    2] <---- Product 1 in moving to machine 2, 2 time units let
        # [None , None] ---> [None , None]
        # [    5,    1] ---> [None , None] <---- Product 3 in moving to machine 5, Arrived
        # [None , None]      [None , None]
        #
        #
        #
        # Product-Bucket:
        # [None, 6, None, 4] --> [None, 6, 5, 4]
        #
        # Product 3 arrived at machine 5
        #
        # ##############################################################################################################
        """

        return EstimatedTimeOfArrival, ProductBucket

    def send(EstimatedTimeOfArrival, ProductBucket, Action, TravelTime):

        for x in range(len(Action)):  # loop over every action signal
            if Action[x] >= 0:  # if action signal is >=0 ==> Signal to send product

                # Action = [-1,0,5,2,-1]
                #
                # action for product 1 = Action[1] = -1
                # action for product 2 = Action[2] = 0
                # action for product 3 = Action[3] = 5
                # action for product 4 = Action[4] = 2
                # action for product 5 = Action[5] = -1
                #

                if ProductBucket[x] is not None:  # if Product is not moving
                    if ProductBucket[x] >= 0:  # if Product is front of a machine / not moving
                        position = ProductBucket[x]  # find position of the product
                        target = Action[x]  # Find target machine to send the product to
                        if position != target:  # if current position is not target position
                            TimeToTarget = TravelTime[position][target]
                            EstimatedTimeOfArrival[x][0] = target  # Set target in ETA
                            EstimatedTimeOfArrival[x][1] = TimeToTarget  # Set remaining travel time in ETA
                            ProductBucket[x] = None  # set position of sending project to "None"

        """
        # ##############################################################################################################
        #  Action:
        #  [-1,0,5,2,-1]
        #
        # action for product 1 = Action[1] = -1
        # action for product 2 = Action[2] = 1
        # action for product 3 = Action[3] = 5
        # action for product 4 = Action[4] = 4
        # action for product 5 = Action[5] = -1
        #
        # Product-Bucket:
        # [None, 6, None, 4, None]
        #
        # action for product 1 = Useless/not executable , product is moving
        # action for product 2 = Action[2] = 1
        # action for product 3 = Useless/not executable , product is moving
        # action for product 4 = Useless/not executable , product is already at the position
        # action for product 5 = Useless/not executable , product is moving
        #
        #
        #
        # action for product 2 = Action[2] = 1
        # position of product 2 = ProductBucket[2] = 6
        # 1 != 6
        #  --> Sending product 2 from position 6 to position 1
        #
        # Travel time :
        # [6,3,7,5,4,6]   TimeToTarget = TravelTime[6][1] = 9
        # [4,6,3,7,9,4]
        # [3,7,3,5,7,4]
        # [3,6,4,3,9,4]
        # [2,5,3,8,6,4]
        # [9,7,3,5,7,4]
        #
        #
        # EstimatedTimeOfArrival:
        # [    2,    3]      [    2,    3]
        # [None , None] ---> [1    ,    9]  <--- Product 2, going to position 1, arriving in 9
        # [    5,    1] ---> [    5,    1]
        # [None , None]      [None , None]
        #
        # Product - Bucket:
        # [None, 6, None, 4, None] ---> [None, None, None, 4, None]   <--- product 2 is now moving
        #
        #
        # ##############################################################################################################
        """

        return EstimatedTimeOfArrival, ProductBucket

    def Induce_Failure(Machine_Failure_Counter, Machine_Failure_Info, RemainingWorkingTime, WorkingTime):
        Failure_Prob = 0.025  # representing the possibility of failure for EVERY machine
        Min_Error_Time = 40  # min time of Failure
        Max_Error_Time = 70  # max time of Failure

        for x in range(len(Machine_Failure_Counter)):
            # x = Machine
            # looping over all machines
            if Machine_Failure_Counter[x] is not None:
                if Machine_Failure_Counter[x] > 0:  # if machine is in Failure-Mode
                    Machine_Failure_Counter[x] -= 1  # time to recovery is reduced by 1

            if Machine_Failure_Counter[x] is 0:  # if time to recovery is reached
                for y in range(len(Machine_Failure_Info)):  # looping over all skills
                    WorkingTime[x][y] = Machine_Failure_Info[x][y]  # all skills are set back into the WT matrix
                    Machine_Failure_Info[x][y] = None  # skills are removed from Machine_Failure_Info-Matrix
                Machine_Failure_Counter[x] = None  # Time to recovery is set to 0

        for x in range(len(WorkingTime)):  # looping over all machines
            if np.random.uniform(0, 1) < Failure_Prob:  # if Failure is statistically possible
                # Failure is only possible if the machine is empty (RWT on the corresponding machine is none)
                # and is machine is functional and empty
                if RemainingWorkingTime[x][1] is None and Machine_Failure_Counter[x] is None:
                    # Duration of failure time is selected
                    Error_Time = random.randint(Min_Error_Time, Max_Error_Time)
                    # Failure time is applied
                    Machine_Failure_Counter[x] = Error_Time
                    # print("Macine " + str(x) + " failed for " + str(Error_Time) + " Seconds")

                    # looping over machine skills
                    # print("machine " + str(x) + " failed" + "for " + str(Error_Time) + " at " + str(step))
                    for y in range(len(WorkingTime)):
                        # machine skills are transfered to the Machine_Failure_Info-matrix for keeping
                        Machine_Failure_Info[x][y] = WorkingTime[x][y]
                        # All skills of the failed machine are deleted for the time being
                        WorkingTime[x][y] = None

        return WorkingTime, Machine_Failure_Counter, Machine_Failure_Info

    def inject(ProductBucket, Action, RemainingWorkingTime, ProductDesign, WorkingTime):

        inject_reward = 0
        for x in range(len(Action)):  # loop over every action signal
            # x is product index
            if Action[x] == -1:  # if working/injection commando
                if ProductBucket[x] is not None:  # if product is in bucket
                    Position = ProductBucket[x]  # find bucket of product
                    if 1 in ProductDesign[x]:  # if some work needs to be done one this product
                        Step = ProductDesign[x].index(1)  # find first necessary working-step == Sequential working
                        if RemainingWorkingTime[Position][0] is None:  # if machine is empty
                            if WorkingTime[Position][Step] is not None:  # if work can be done on this machine
                                WorkTime = WorkingTime[Position][Step]  # Find working Time for this specific step on
                                # this machine
                                RemainingWorkingTime[Position][0] = x  # set product into working machine
                                RemainingWorkingTime[Position][1] = WorkTime  # set remaining working-time on machine
                                RemainingWorkingTime[Position][2] = Step  # set current working-step on machine

                                # SOLVE THE POSITION WHILE IN MACHINE PROBLEM / is this duplication on information ?

                                ProductBucket[x] = None
                                """
                                # ######################################################################################
                                # Optionally:
                                # A inject_reward can be added to hint that injecting a product
                                # into a machine is a good decision.
                                # MAY CONFUSE THE AGENT !!
                                # ######################################################################################
                                """

                                inject_reward += 0
        """
        # ##############################################################################################################
        #
        # Example:
        #
        # WorkingTime:
        # [4,None,None,None,None]   Machine 1 capable performing step 1 in 4 time units
        # [None,5,None,7,None]      Machine 2 capable performing step 2 in 5 time units and step 4 in 7 time units
        # [None,None,5,None,None]   Machine 3 capable performing step 3 in 5 time units
        # [None,8,None,4,None]      Machine 4 capable performing step 4 in 4 time units and step 2 in 8 time units
        # [None,None,None,None,3]   Machine 5 capable performing step 5 in 3 time units
        #
        # Product-Bucket:
        # [None, None, None, 2, None]
        #
        # Product 1 is in not in front of any machine (Moving)
        # Product 2 is in not in front of any machine (Moving)
        # Product 3 is in not in front of any machine (Moving)
        # Product 4 is in front of the 2nd machine            <------------------
        # Product 5 is in not in front of any machine (Moving)
        #
        # Action:
        # [-1,0,5,-1,4]
        #
        # action for product 1 = Useless/not executable , product is moving
        # action for product 2 = Useless/not executable , product is moving
        # action for product 3 = Useless/not executable , product is moving
        # action for product 4 = Should be worked on next   <--------------------
        # action for product 5 = Useless/not executable , product is moving
        #
        # ProductDesign:
        # [  None,   None,  None,  None,  None]    Product 1 , completely done
        # [  None,   None,  None,  None,     1]    Product 2 , needs just the last step steps
        # [  None,   None,  None,     1,     1]    Product 3 , needs the last 2 steps
        # [  None,   None,  None,     1,     1]    Product 4 , needs the last 2 steps <--------------
        # [     1,      1,     1,     1,     1]    Product 5 , needs all steps
        #
        # Next necessary step for product 4, is step 4
        #
        # Position of product 4 is bucket 2
        #
        # can machine 2 perform step 4 ?
        #  --> Yes (see WorkingTime) in 7 time units
        #
        #
        # ##############################################################################################################
        """

        return RemainingWorkingTime, ProductBucket, inject_reward

    def calculate_reward(ProductDesign, step, max_timesteps, inject_reward, pre_done):

        """
        # "It do what it do, never what you wanna it to do,
        #  let it do it do, someday it do right, don't judge"

        # ##############################################################################################################
        # The reward function is one of the most important elements.
        #
        # Current state:
        # The reward is only given at the end of the episode.(exception if Inject reward is added)
        #
        # If this reward function is shit, nothing is gonna work.
        #
        # RULES:
        # 1) A reward function should clearly state how well the desired objective was achieved.
        #  --> NOT HOW TO ACHIEVE IT !
        # 2) The reward function character should be exponential !
        # --> a step in the right direction is rewarded more than a step in the wrong direction is punished!
        #
        # ##############################################################################################################
        """
        """"
        #
        # can be Cubed , can be squared, or some kind of exponential function
        #
        """

        # the reward consists of multiple elements:
        # In this case the reward is set to be the injection reward
        reward = inject_reward
        reward = 0

        # If all steps have been done before the max_timesteps were reached
        # aka if no "1" exists in the ProductDesign-Matrix
        if not 1 in np.array(ProductDesign):
            # the reward is subdivided into:
            # the difference between the time if took to finish all steps and the max_timesteps
            # this difference is cubed to keep the exponential character
            # divided by 20 because cubing is in some cases too extreme
            reward += (max_timesteps - step) ** 3

            # all finished steps are counted
            finished = (np.array(ProductDesign) == None).sum()
            # the amount of finished steps is cubed and added to the time-Reward
            reward += (finished - pre_done) ** 3

        # if the agent did not manage to finish all steps in time
        elif max_timesteps == step + 1:
            # All non-finished and finished elements are counted
            left_to_do = (np.array(ProductDesign) == 1).sum()
            finished = (np.array(ProductDesign) == None).sum()
            # The final result of the episode is the amount of finished steps cubed
            reward += (finished - pre_done) ** 3
            # reward += -left_to_do * 10

        """
        # a negative reward is given every time unit for evey unfinished step
        # !!! MAY CONFUSE THE AGENT !!!
        # The purpose is to motivate the agent to comlplete the products as fast as possible!
        # Additional time reward will be given after completion  
        

        # Counting unfinished steps
        # unfinished = (np.array(ProductDesign) == 1).sum()
        # subtracting steps from total step-reward
        # reward += -unfinished/2
        """
        # returning the reward
        return reward

    def check_if_done(ProductDesign):

        # this function is responsible for changing the "done" indicator
        # if there is no "1" in the ProductDesign marix ...
        if not 1 in np.array(ProductDesign):
            # ...the indicator is changed to "True"
            done = True
            # And just a printed message for user satisfaction
            print("I DID IT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("on step " + str(step))
        else:
            done = False

        # this function could be combined with the "calculate_reward()" or "eject()" for speed efficiency
        return done

    RemainingWorkingTime = work(RemainingWorkingTime)

    RemainingWorkingTime, ProductBucket, ProductDesign = eject(RemainingWorkingTime, ProductBucket, ProductDesign)

    EstimatedTimeOfArrival, ProductBucket = travel(EstimatedTimeOfArrival, ProductBucket)

    EstimatedTimeOfArrival, ProductBucket = send(EstimatedTimeOfArrival, ProductBucket, Action, TravelTime)

    WorkingTime, Machine_Failure_Counter, Machine_Failure_Info = Induce_Failure(Machine_Failure_Counter,
                                                                                Machine_Failure_Info,
                                                                                RemainingWorkingTime, WorkingTime)

    RemainingWorkingTime, ProductBucket, inject_reward = inject(ProductBucket, Action, RemainingWorkingTime,
                                                                ProductDesign, WorkingTime)

    reward = calculate_reward(ProductDesign, step, max_timesteps, inject_reward, pre_done)

    done = check_if_done(ProductDesign)

    return ProductDesign, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, reward, done


def GenerateRandomAction(WorkingTime, ProductDesign):
    # generating a random Action for every product
    Action = np.random.randint(-1, len(WorkingTime), len(ProductDesign))  #
    return Action


def linearFIFO(ProductBucket, ProductDesign):
    """ ONLY WORKS WITH DIAGONAL MATRIX """
    Action = []  # create empty list
    for x in range(len(ProductBucket)):  # loop over prodructs
        if 1 in ProductDesign[x]:  # if product still requires a processing step
            target = ProductDesign[x].index(1)  # find target machine
            Action.append(target)  # append to list "Action"
            if Action[x] == ProductBucket[x]:  # if Position = Target Machine
                Action[x] = -1  # Overwrite action to "Inject"
        else:
            Action.append(-1)  # if product does not require a processing step, inject signal

    return Action


def betterFIFO(ProductBucket, ProductDesign, WorkingTime):
    """ATTENTION !!!  ONLY WORKS ON THE MANUALLY DEFINED MATRIX"""
    """ATTENTION !!!  ONLY WORKS ON THE MANUALLY DEFINED MATRIX"""
    """ATTENTION !!!  ONLY WORKS ON THE MANUALLY DEFINED MATRIX"""
    """ATTENTION !!!  ONLY WORKS ON THE MANUALLY DEFINED MATRIX"""

    #Action = []
    for x in range(len(ProductBucket)):
        if 1 in ProductDesign[x]:
            target = ProductDesign[x].index(1)
            Action.append(target)

            if target is not 4:
                if WorkingTime[target + 1][target] is not None:
                    Action[x] = (target + 1)

            if target is not 0:
                if WorkingTime[target - 1][target] is not None:
                    Action[x] = (target - 1)

            if WorkingTime[target][target] is not None:
                Action[x] = (target)

            if Action[x] == ProductBucket[x]:
                Action[x] = -1

        else:
            Action.append(-1)
    # print(Action)

    return Action


def GenerateState(ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket,
                  done, Machine_Failure_Counter, Machine_Failure_Info):
    # At first all matrices are flatted

    flat_WorkingTime = np.matrix(WorkingTime).flatten().tolist()
    flat_ProductDesign = np.matrix(ProductDesign).flatten().tolist()
    flat_TravelTime = np.matrix(TravelTime).flatten().tolist()
    flat_ProductBucket = np.matrix(ProductBucket).flatten().tolist()
    flat_RemainingWorkingTime = np.matrix(RemainingWorkingTime).flatten().tolist()
    flat_EstimatedTimeOfArrival = np.matrix(EstimatedTimeOfArrival).flatten().tolist()
    flat_Machine_Failure_Counter = np.matrix(Machine_Failure_Counter).flatten().tolist()
    """
    Divide all integers in flat_Machine_Failure_Counter to reduce information suppression
    """
    for x in range(len(flat_Machine_Failure_Counter[0])):  # loop over flat_Machine_Failure_Counter
        # print(flat_Machine_Failure_Counter[0][x])
        if flat_Machine_Failure_Counter[0][x] is not None:
            flat_Machine_Failure_Counter[0][x] /= 5

    # All matrices are combined into one vector
    all_flat = flat_WorkingTime[0] + flat_ProductDesign[0] + flat_TravelTime[0] + flat_ProductBucket[0] + \
               flat_RemainingWorkingTime[0] + flat_EstimatedTimeOfArrival[0]

    all_flat = flat_ProductDesign[0] + flat_ProductBucket[0] + flat_EstimatedTimeOfArrival[0] + \
               flat_RemainingWorkingTime[0] + flat_Machine_Failure_Counter[0]

    # all_flat = flat_ProductDesign[0] + flat_ProductBucket[0] + flat_EstimatedTimeOfArrival[0] + \
    #            flat_RemainingWorkingTime[0]

    # All elements of "None" are changed to "-1" because the neural nat does not accept "None" as inputs
    for place, posit in enumerate(all_flat):
        if posit is None:
            all_flat[place] = -1

    # Array is transformed into float64 for precision
    all_flat = np.asarray(all_flat, dtype=np.float64)

    # All elements of the vector are compressed to a scale from -1 to 1
    # --> Normalization

    all_flat = np.interp(all_flat, (-1, all_flat.max()), (-1, +1))

    return all_flat


def extract_Actions(Action_raw, ammount_of_products, ammount_of_machines):
    # A empty array is created to store the extracted actions
    Action = []

    for x in range(ammount_of_products):
        """
        # ##############################################################################################################
        # Example:
        # The raw action vector looks like this
        # [0.2345 , 0.564 , 0.34 , 0.7653 , 0.456 , 0.4234 , 0.5345634 , 0.64356 , ............. ]
        #
        # At first the vector is subdivided into parts
        # Every part of the raw vector is attributed to a product    
        #
        #  Section for product 1      Section for product 2      Section for product 3
        # [0.2345 , 0.564 , 0.34 ]  [0.7653 , 0.456 , 0.4234 ] [0.5345634 , 0.64356 , ............. ]
        #
        # Next the index of the maximum element is located
        #
        #  Section for product 1           Section for product 2            Section for product 3
        # [0.2345 , 0.564 , 0.34 ]       [0.7653 , 0.456 , 0.4234 ]     [0.5345634 , 0.64356 , ............. ]
        #       Max index = 1                   Max index = 0                     Max index = 1
        #
        # 1 is subtracted from the index to generate the applicable signal
        # 
        # index 0 --> -1  == inject 
        # index 1 -->  0  == go to machine 0
        # index 2 -->  1  == go to machine 1
        # index 3 -->  2  == go to machine 2
        # index 4 -->  3  == go to machine 3
        # ...
        #
        # Extracted orders are compiled to a single vector
        """

        # subdivision into sections
        section = Action_raw[(x * (ammount_of_machines + 1)):(x + 1) * (ammount_of_machines + 1)]
        # looking for the index of the max element
        signal = section.argmax() - 1
        # Compiling signal
        Action.append(signal)

    return Action


def Randomise_Action(Action_raw, exploration_noise_max, exploration_noise_min, exploration_noise_decay, episode):
    """
    # ##################################################################################################################
    # To increase the chance of finding the global minimum a certain degree of exploration is needed
    # To generate noise a vector is created to be added to the raw action-vector.
    # The noise is vector as long as the raw action-vector
    # The noise elements are in range from 0 to exploration_noise_max
    #
    # Example:
    # [0.213 , 0.2456 , 0.8434, ....]
    #
    # Every loop the range of the possible noise is decreased to allow the Agent to choose its own actions
    #
    # ##################################################################################################################
    """

    if exploration_noise_max > exploration_noise_min:
        # range is decreased
        exploration_noise_max *= exploration_noise_decay
    else:
        exploration_noise_max = exploration_noise_min

    # if episode > 2000:
    #   exploration_noise_max = exploration_noise_max / 2

    # if episode > 4000:
    #   exploration_noise_max = exploration_noise_max / 2

    # if episode > 3000:
    #    exploration_noise_max = exploration_noise_max / 2

    # noise vector is created in the current range
    noise = np.random.normal(0, exploration_noise_max, size=(len(Action_raw)))
    # noise vector is added to the raw action vector
    Action_raw = Action_raw + noise

    return Action_raw, exploration_noise_max


"""
# ######################################################################################################################
# all Global parameters for the training-duration
"""

max_episodes = 1000000  # max num of episodes """ USE MAX TRAINING-TIME INSTEAD""
max_timesteps = 70  # max time-steps in one episode
store = []  # vector to store the rewards

"""
# ######################################################################################################################
# all Global parameters for The Agent 
# are set in "Create_Agent_Parameters()"
# May be adjusted for better performance 
# ######################################################################################################################
"""

lr, state_dim, action_dim, max_action, exploration_noise_max, exploration_noise_min, exploration_noise_decay = Create_Agent_Parameters()

# Policy is created
Policy = TD3(lr, state_dim, action_dim, max_action)

"""
# ######################################################################################################################
# Policy can optionally be loaded from 2 folders (perTrained/inTraining)
# ######################################################################################################################
"""

# Policy.load("./perTrained", "TD3")
# Policy.load("./inTraining", "TD3")

# creating parameters
batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay = Create_Update_Parameters()

# creating replay buffer
replay_buffer = ReplayBuffer()

startingtime = time.time()
tage = 7
stunden = 0
minuten = 0
zeit = tage * 24 * 60 * 60 + stunden * 60 * 60 + minuten * 60
episode = 0

""" Optional training duration (Time/episodes)"""
for episode in range(1, max_episodes + 1):
    # while startingtime + zeit > time.time():  """ALTERNATIVE TRAINING CYCLE"""

    start = time.time()
    '''
    # ##################################################################################################################
    #  At  first a factory has to be initialised.
    #
    #  all the necessary parameters are set inside the function "create_factory()"
    #  Parameters can be changed to represent a specific factory
    #  If demanded, outputs of the function "create_factory()" can be created manually/explicitly
    #  Please make sure all dimensions are compatible!
    #  No Error-correction system is yet implemented!
    #
    #  the function "create_factory()" returns:
    #
    #    ProductDesign
    #    WorkingTime
    #    TravelTime
    #    RemainingWorkingTime
    #    EstimatedTimeOfArrival
    #    ProductBucket
    #    done
    #
    #  Use of these elements is explained in their creation-functions
    # ##################################################################################################################
    '''
    ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, done, score, Machine_Failure_Counter, Machine_Failure_Info = create_factory()

    """
    # ##################################################################################################################
    # All information is packed into one variable to reduce the amount of data being passed to the functions
    # ##################################################################################################################
    """

    # comprising variables
    Information = ProductDesign, WorkingTime, TravelTime, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, done, Machine_Failure_Counter, Machine_Failure_Info

    """
    # ##################################################################################################################
    # At first the factory sends a state-vector, fully describing the current state of production,
    # independent of past states (Markov-property).
    # The state is created out of elements created in "create_factory()"
    # If some elements are not changing/constant in between runs , they can be excludes from the state vector
    # ##################################################################################################################
    """

    # generating state
    state_prior = GenerateState(*Information)

    """
    # ##################################################################################################################
    #  The simulation keeps running until it is done.
    #  "done" can be defined by the user.
    #  In this case, the factory is "done" as soon as all products have been finished.
    #  (All necessary manufacturing steps are executed, on all individual products)
    # ##################################################################################################################
    """
    # initial parameters for every game
    step = 0
    game_reward = 0
    pre_done = 0

    # random_steps_before_takeover = random.randint(0, 40)
    random_steps_before_takeover = 10

    # to generate a random starting state, some steps are performed before the Agent takes over
    for x in range(random_steps_before_takeover):
        # A random action is generated
        Action = GenerateRandomAction(WorkingTime, ProductDesign)
        # The Action is executed
        ProductDesign, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, step_reward, done = factory_step(
            *Information, Action, step, max_timesteps, pre_done)

    # If any steps have been performed during the execution of the random actions,
    # the are counted, to enable the correct rewarding of the agent.
    # Otherwise the agent might get rewarded for something that was done by random before takeover
    pre_done = (np.array(ProductDesign) == None).sum()

    while not done and step < max_timesteps:
        """
        # ##############################################################################################################
        # At first the machines are subjected to failure. "Induce_Failure()"
        # Next, a action is generated that is executed on the next step
        # This function takes into account the current state
        # ##############################################################################################################
        """

        # A State is generated
        state_prior = GenerateState(*Information)

        # Receiving the exact output of the neural net
        "recording the time for the Online-Reaction capability"
        # TimeToReactStart = time.time()  # recording the time for the Online-Reaction capability

        Action_raw = Policy.select_action(state_prior)

        "Printing the time of reaction"
        # print(time.time()-TimeToReactStart)  # Printing the time of reaction

        # Adding exploration
        Action_raw, exploration_noise_max = Randomise_Action(Action_raw, exploration_noise_max, exploration_noise_min,
                                                             exploration_noise_decay, episode)

        # Extracting actions
        Action = extract_Actions(Action_raw, len(ProductBucket), len(RemainingWorkingTime))
        """For comparison the recieved action from the agent can be overwritten co compare to other Metrics"""
        # Action = GenerateRandomAction(WorkingTime, ProductDesign)

        """Only WORKS ON THE MANUALLY DEFINED MATRIX !!!"""
        # Action = linearFIFO(ProductBucket, ProductDesign)
        # Action = betterFIFO(ProductBucket, ProductDesign, WorkingTime)

        """
        # ##############################################################################################################
        # factory_step():
        # Here the Actions are executed.
        # The function "factory_step" requires all the information form "Information" and the Action for execution.
        # Outputs are the elements that might have changed
        #
        # ##############################################################################################################
        """

        ProductDesign, RemainingWorkingTime, EstimatedTimeOfArrival, ProductBucket, step_reward, done = factory_step(
            *Information, Action, step, max_timesteps, pre_done)

        # A new state is generated
        state_post = GenerateState(*Information)

        # ac is added to the buffer
        replay_buffer.add((state_prior, Action_raw, step_reward, state_post, float(done)))

        # Step is increased by 1
        step += 1

        # game rearward ids increased by episode reward
        game_reward += step_reward

        """
        # ##############################################################################################################
        # This process is repeated until "done"
        """

    # game_reward is added to the vector "store" to store the rewards
    store.append(game_reward)

    # episode += 1
    if episode % 200 == 0:
        # every 10th episode the learning progress is saved in a sub-folder
        Policy.save("./inTraining", "TD3")
        pickle.dump(store, open("reward-storageBACKUP.p", "wb"))

    # All game rearwards are saved with pickl-dump in the same folder for future visualisation
    pickle.dump(store, open("reward-storage.p", "wb"))

    # Here is the learning process
    Policy.update(replay_buffer, step, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)

    # Everything else just for visualization
    # print("")
    end = time.time()
    duration = end - start
    print("Episode: {}\tAverage Reward: {}\t Explore: {}\t Time: {}".format(episode, game_reward, exploration_noise_max,
                                                                            duration))

    # Printing the matrix for visual support of progress
    for x in range(len(ProductDesign)):
        print(ProductDesign[x])
    print(" ")
