import random
import copy
import json
from typing import Iterator, List, Dict
from TakuNet import TakuNetModel
from data_processing import get_dataset
import time

class EvolutionarySearch:
    def __init__(self, config_path: str, population_size: int, time: float, mutation_rate: float, crossover_rate: float):
        with open(config_path, "r") as file:
            self.config = json.load(file)
        
        self.population_size = population_size
        self.time = time
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[TakuNetModel] = []
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()
    
    def _load_data(self):
        """Loads the dataset using the get_dataset function from data_processing.py"""
        num_classes = self.config["model_search_space"]["refiner_block"]["num_output_classes"]
        use_augmented_data = False  # Change to True if you want data augmentation
        return get_dataset(output_classes=num_classes, use_augmented_data=use_augmented_data)
    
    def _random_hyperparameters(self) -> Dict:
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
    
    def _initialize_population(self):
        """Creates the initial population of models."""
        for _ in range(self.population_size):
            model_params = self._random_hyperparameters()
            train_params = {key: random.choice(values) if isinstance(values, list) else values for key, values in self.config["train_and_evaluate"]["model_config"].items()}
            train_params.update(self.config["train_and_evaluate"]["evaluation_config"])
            model = TakuNetModel(f"TakuNet_{_}", (32, 32, 3), model_params, train_params, self.x_train, self.y_train, self.x_test, self.y_test)
            self.population.append(model)
    
    def _evaluate_fitness(self, model: TakuNetModel) -> float:
        """Evaluates a model's performance based on accuracy, precision, recall, and memory constraints."""
        model.train()
        if model.results.train_accuracy is None:
            return -1  # Discard models that exceed memory limits
        return model.results.test_accuracy + model.results.precision + model.results.recall # This must be changed in order to evaluate better the models
    
    def _select_parents(self) -> List[TakuNetModel]:
        """Tournament selection: pick random groups and choose the best."""
        selected_parents = []
        tournament_size = max(2, self.population_size // 5)  # Ensure at least 2 competitors per tournament
        
        for _ in range(self.population_size // 2):
            tournament = random.sample(self.population, tournament_size)
            best_model = max(tournament, key=self._evaluate_fitness)
            selected_parents.append(best_model)

        return selected_parents
    
    def _mutate(self, model_params: Dict) -> Dict:
        """Applies random mutations to a model's hyperparameters."""
        if random.random() < self.mutation_rate:
            block = random.choice(list(model_params.keys())) # This returns the block that the mutation is goind to happen
            
            if type(model_params[block]) is dict:
                subBlock = random.choice(list(model_params[block].keys()))

                potentialFilter = self.config["model_search_space"][block][subBlock]

                if type(potentialFilter) is dict:
                    filter = random.choice(list(model_params[block][subBlock].keys()))
                    model_params[block][subBlock][filter] = random.choice(self.config["model_search_space"][block][subBlock][filter])
                if type(potentialFilter) is list:
                    model_params[block][subBlock] = random.choice(potentialFilter)
            
            else:
                raise Exception(f"Very weird models_params[block] {model_params[block]}")
        
        return model_params
    
    def _crossover(self, parent1: TakuNetModel, parent2: TakuNetModel, model_number:int) -> TakuNetModel:
        """Performs crossover between two parent models."""
        child_params = copy.deepcopy(parent1.model_params)
        if random.random() < self.crossover_rate:
            key = random.choice(list(child_params.keys()))
            if isinstance(child_params[key], dict):
                subkey = random.choice(list(child_params[key].keys()))
                child_params[key][subkey] = parent2.model_params[key][subkey]
            else:
                child_params[key] = parent2.model_params[key]
        
        train_params = copy.deepcopy(parent1.train_params)
        model_name = f"TakuNet_Crossover_{model_number}"
        return TakuNetModel(model_name, (32, 32, 3), child_params, train_params, self.x_train, self.y_train, self.x_test, self.y_test)
    
    def evolve(self)->Iterator[TakuNetModel]:
        """Runs the evolutionary search process."""
        start_time = time.time()
        max_duration_seconds = self.time * 3600
        self._initialize_population()
        model_number = 0
        while time.time() - start_time < max_duration_seconds:
            print(f"\n⏳ Evolving new population (elapsed: {(time.time() - start_time)/60:.2f} min)...")
            parents:List[TakuNetModel] = self._select_parents()
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
                    new_population.append(TakuNetModel(model_name, (32, 32, 3), mutant_params, train_params, self.x_train, self.y_train, self.x_test, self.y_test))
            
            self.population = new_population
            
            current_best_model:TakuNetModel = max(
                                    self.population, 
                                    key=lambda model: model.results.test_accuracy if model.results.test_accuracy is not None else -1)
            print(f"🔥 Yielding best model after population evolution: {current_best_model.model_name}")
            yield current_best_model
