import random
import copy
import json
from typing import Iterator, List, Dict, Union
from TakuNet import TakuNetModel
from data_processing import get_dataset
import time
from utils import getSearchSpaceParameters, getTrainingParameters

class EvolutionarySearch:
    def __init__(self, config_path: str, population_size: int, time: float, mutation_rate: float, crossover_rate: float, augmentation_techinque: Union[Dict, bool]):
        with open(config_path, "r") as file:
            self.config = json.load(file)
        
        self.population_size = population_size
        self.time = time
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[TakuNetModel] = []
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data(augmentation_technique=augmentation_techinque)
    
    def _load_data(self,augmentation_technique: Union[Dict, bool]):
        """Loads the dataset using the get_dataset function from data_processing.py"""
        if augmentation_technique is False:
            augmentation_techique = {"apply_standard":False,
                                    "apply_color":False,
                                    "apply_geometric":False,
                                    "apply_mixup": False,
                                    "apply_cutmix": False
                                    }
        num_classes = self.config["model_search_space"]["refiner_block"]["num_output_classes"]
        return get_dataset(output_classes=num_classes, augementation_technique=augmentation_techique)
    
    def _random_model_parameters(self) -> Dict:
        """Generates a random set of hyperparameters from the search space."""
        model_search_space = self.config["model_search_space"]
        return {
                "stem_block": {
                    "filters": random.choice(model_search_space["stem_block"]["filters"]),
                    "Conv_kernel": random.choice(model_search_space["stem_block"]["Conv_kernel"]),
                    "Conv_strides": random.choice(model_search_space["stem_block"]["Conv_strides"]),
                    "dropout": random.choice(model_search_space["stem_block"]["dropout"]),
                    "dilation_rate": random.choice(model_search_space["stem_block"]["dilation_rate"]),
                    "DWConv_kernel": random.choice(model_search_space["stem_block"]["DWConv_kernel"]),
                    "DWConv_strides": random.choice(model_search_space["stem_block"]["DWConv_strides"])
                    },
                "stages_block": {
                    "stages_number": random.choice(model_search_space["stages_block"]["stages_number"]),
                    "taku_block": {
                        "taku_block_number": random.choice(model_search_space["stages_block"]["taku_block"]["taku_block_number"]),
                        "dropout": random.choice(model_search_space["stages_block"]["taku_block"]["dropout"]),
                        "DWConv_kernel": random.choice(model_search_space["stages_block"]["taku_block"]["DWConv_kernel"]),
                        "DWConv_strides": random.choice(model_search_space["stages_block"]["taku_block"]["DWConv_strides"])
                    },
                    "downsampler": {
                        "dropout": random.choice(model_search_space["stages_block"]["downsampler"]["dropout"]),
                        "pool_size": random.choice(model_search_space["stages_block"]["downsampler"]["pool_size"]),
                        "Conv_kernel": random.choice(model_search_space["stages_block"]["downsampler"]["Conv_kernel"]),
                        "strides": random.choice(model_search_space["stages_block"]["downsampler"]["strides"]),
                    }
                },
                "refiner_block": {
                    "DWConv_kernel": random.choice(model_search_space["refiner_block"]["DWConv_kernel"]),
                    "DWConv_strides": random.choice(model_search_space["refiner_block"]["DWConv_strides"]),
                    "dropout": random.choice(model_search_space["refiner_block"]["dropout"]),
                    "num_output_classes": model_search_space["refiner_block"]["num_output_classes"]
                }
        }
    def _random_training_parameters(self) ->Dict:
        " This creates the training parameters for the TakuModel"
        train_params = {key: random.choice(values) if isinstance(values, list) else values for key, values in self.config["train_and_evaluate"]["model_config"].items()}
        train_params.update(self.config["train_and_evaluate"]["evaluation_config"]) 
        return train_params
    
    # def _initialize_population(self):
    #     """Creates the initial population of models."""
    #     for i in range(self.population_size):

    #         model_params = self._random_model_parameters() # Here we randomly select params for "model_search_space"
    #         train_params = self._random_training_parameters() # Here we randomly select params for "train_and_evaluate"

    #         model = TakuNetModel(f"TakuNet_Init_{i}", (32, 32, 3), model_params, train_params, self.x_train, self.y_train, self.x_test, self.y_test)
    #         self.population.append(model)

    def _initialize_population(self):
        """ Creates the initial population of models, 
            skipping untrainable ones,
            Train trainable ones!"""
        print("🚀 Initializing population...")
        created = 0
        attempts = 0
        max_attempts = self.population_size * 10  # Prevent infinite loop in rare cases

        while created < self.population_size and attempts < max_attempts:
            attempts += 1
            model_params = getSearchSpaceParameters.sample_from_search_space(self.config["model_search_space"])
            train_params = getTrainingParameters.sample_from_train_and_evaluate(self.config["train_and_evaluate"])
            
            #model_params = self._random_model_parameters()
            #train_params = self._random_training_parameters()

            model = TakuNetModel(model_name=f"TakuNet_Init_{created}", 
                                 input_shape=(32, 32, 3), 
                                 model_params=model_params, 
                                 train_params=train_params, 
                                 x_train=self.x_train, 
                                 y_train=self.y_train, 
                                 x_test=self.x_test, 
                                 y_test=self.y_test,
                                 folder="NAS")
            model.train()

            if model.results.train_accuracy is not None:
                self.population.append(model)
                created += 1
                print(f"✅ Added model {model.model_name} to population (total: {created})")
            else:
                print(f"❌ Skipping model {model.model_name} due to memory limits")

        if created < self.population_size:
            print(f"⚠️ Only {created}/{self.population_size} models were valid after {attempts} attempts.")

    
    def _evaluate_fitness(self, model: TakuNetModel) -> float:
        """Evaluates a model's performance based on accuracy, precision, recall, and memory constraints."""
        if not model.is_trained:
            model.train() # This stops the re-training of models that have been trained already
        if model.results.train_accuracy is None:
            return -1  # Discard models that exceed memory limits
        return model.results.test_accuracy # This must be changed in order to evaluate better the models
    
    def _select_parents(self) -> List[TakuNetModel]:
        """Selects parents using 1v1 tournament style; last 3 form a mini-tournament if population is odd."""
        shuffled = random.sample(self.population, len(self.population))  # Random order
        selected_parents = []
        i = 0
        while i < len(shuffled) - 1:
            # If 3 models left at the end, do a 3-way match
            if i + 2 == len(shuffled):
                trio = shuffled[i:i+3]
                best = max(trio, key=self._evaluate_fitness)
                selected_parents.append(best)
                break
            else:
                model1, model2 = shuffled[i], shuffled[i+1]
                fitness1 = self._evaluate_fitness(model1)
                fitness2 = self._evaluate_fitness(model2)
                winner = model1 if fitness1 >= fitness2 else model2
                selected_parents.append(winner)
                i += 2

        return selected_parents
    # def _select_parents(self) -> List[TakuNetModel]:
    #     """Tournament selection: pick random groups and choose the best."""
    #     selected_parents = []
    #     tournament_size = max(2, self.population_size // 5)  # Ensure at least 2 competitors per tournament
        
    #     for _ in range(self.population_size // 2): # x//y returns the integer part of the diviation
    #         tournament = random.sample(self.population, tournament_size)
    #         best_model = max(tournament, key=self._evaluate_fitness)
    #         selected_parents.append(best_model)

    #     return selected_parents
    
    def _mutate(self, model_params: Dict) -> Dict:
        """
        In this mutation we go over each Model Search Parameter and based on this : random.random() < self.mutation_rate
        We either change it or not. The higher the self.mutation_rate, the more parameters will change !
        """
        for block in model_params:
            if isinstance(model_params[block], dict):
                for subBlock in model_params[block]:
                    if isinstance(model_params[block][subBlock], dict):
                        for param in model_params[block][subBlock]:
                            if random.random() < self.mutation_rate:
                                choices = self.config["model_search_space"][block][subBlock][param]
                                model_params[block][subBlock][param] = random.choice(choices)
                    else:
                        if random.random() < self.mutation_rate:
                            choices = self.config["model_search_space"][block][subBlock]
                            if isinstance(choices, list):
                                model_params[block][subBlock] = random.choice(choices)
            else:
                raise Exception(f"Unexpected non-dict block at top-level: {block}")
        return model_params

    def _crossover(self, parent1: TakuNetModel, parent2: TakuNetModel, model_number: int) -> TakuNetModel:
        """ In this crossover, the child is a deep copy of the first parent and based on the probabilistic,
            random.random() < self.crossover_rate, it will get the parent's 2 parameter
            For all possible Model Search Parameters"""
        child_params = copy.deepcopy(parent1.model_params)
        
        for block in child_params:
            if isinstance(child_params[block], dict):
                for subBlock in child_params[block]:
                    if isinstance(child_params[block][subBlock], dict):
                        for param in child_params[block][subBlock]:
                            if random.random() < self.crossover_rate:
                                child_params[block][subBlock][param] = parent2.model_params[block][subBlock][param]
                    else:
                        if random.random() < self.crossover_rate:
                            child_params[block][subBlock] = parent2.model_params[block][subBlock]
            else:
                if random.random() < self.crossover_rate:
                    child_params[block] = parent2.model_params[block]

        train_params = copy.deepcopy(parent1.train_params)
        model_name = f"TakuNet_Crossover_{model_number}"
        return TakuNetModel(model_name=model_name, 
                            input_shape=(32, 32, 3), 
                            model_params=child_params, 
                            train_params=train_params, 
                            x_train=self.x_train, 
                            y_train=self.y_train, 
                            x_test=self.x_test, 
                            y_test=self.y_test,
                            folder="NAS")

    
    # def _mutate(self, model_params: Dict) -> Dict:
    #     """Applies random mutations to a model's hyperparameters."""
    #     if random.random() < self.mutation_rate:
    #         block = random.choice(list(model_params.keys())) # This returns the block/parameter that the mutation is going to happen
            
    #         if type(model_params[block]) is dict:
    #             subBlock = random.choice(list(model_params[block].keys()))

    #             potentialFilter = self.config["model_search_space"][block][subBlock]

    #             if type(potentialFilter) is dict:
    #                 filter = random.choice(list(model_params[block][subBlock].keys()))
    #                 model_params[block][subBlock][filter] = random.choice(self.config["model_search_space"][block][subBlock][filter])
    #             if type(potentialFilter) is list:
    #                 model_params[block][subBlock] = random.choice(potentialFilter)
            
    #         else:
    #             raise Exception(f"Very weird models_params[block] {model_params[block]}")
        
    #     return model_params
    
    # def _crossover(self, parent1: TakuNetModel, parent2: TakuNetModel, model_number:int) -> TakuNetModel:
    #     """Performs crossover between two parent models."""
    #     child_params = copy.deepcopy(parent1.model_params) # Here the child is a copy of the parent1
    #     if random.random() < self.crossover_rate:
    #         key = random.choice(list(child_params.keys()))
    #         if isinstance(child_params[key], dict):
    #             subkey = random.choice(list(child_params[key].keys()))
    #             child_params[key][subkey] = parent2.model_params[key][subkey]
    #         else:
    #             child_params[key] = parent2.model_params[key]
        
    #     train_params = copy.deepcopy(parent1.train_params)
    #     model_name = f"TakuNet_Crossover_{model_number}"
    #     return TakuNetModel(model_name, (32, 32, 3), child_params, train_params, self.x_train, self.y_train, self.x_test, self.y_test)
    
    def evolve(self)->Iterator[TakuNetModel]:
        """Runs the evolutionary search process."""
        start_time = time.time()
        max_duration_seconds = self.time * 3600
        self._initialize_population() # Here we create 10 un-trained Models
        model_number = 0
        while time.time() - start_time < max_duration_seconds:
            # TODO , I have to do something with the Pareto Front, to add only the models that do not have another model explicitly better
            current_best_model:TakuNetModel = max(
                                    self.population, 
                                    key=lambda model: model.results.test_accuracy if model.results.test_accuracy is not None else -1)
            print(f"🔥 Yielding best model after population evolution: {current_best_model.model_name}")
            yield current_best_model

            print(f"\n⏳ Evolving new population (elapsed: {(time.time() - start_time)/60:.2f} min)...")

            parents:List[TakuNetModel] = self._select_parents() 
            """
            Here we try to Select the Parents with Tournament Selection, we train them and they compete with each other
            Return half of the Population as Parents
            """

            new_population = parents.copy()
            

            while len(new_population) < self.population_size:
                if random.random() < 0.5:
                    model_number = model_number + 1
                    parent1, parent2 = random.sample(parents, 2)
                    new_population.append(self._crossover(parent1, parent2, model_number))
                else:
                    model_number = model_number + 1
                    mutant_params = self._mutate(copy.deepcopy(random.choice(parents).model_params))
                    train_params = copy.deepcopy(parents[0].train_params)
                    model_name = f"TakuNet_Mutant_{model_number}"
                    taku_model:TakuNetModel = TakuNetModel(model_name=model_name,
                                                           input_shape=(32, 32, 3),
                                                           model_params=mutant_params,
                                                           train_params=train_params,
                                                           x_train=self.x_train,
                                                           y_train=self.y_train,
                                                           x_test=self.x_test,
                                                           y_test=self.y_test,
                                                           folder="NAS")
                    new_population.append(taku_model)


            # update the population
            self.population = new_population