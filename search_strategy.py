import gc
import random
import copy
import json
from typing import Iterator, List, Dict, Tuple, Union

import numpy as np
from SurrogateComparisson.Embedding import simple_architecture_embedding
from TakuNet import TakuNetModel
from data_processing import get_dataset
import time
from utils import getSearchSpaceParameters, getTrainingParameters
from SurrogateComparisson.RankNet import build_ranknet

class EvolutionarySearch:
    def __init__(self, config_path: str, population_size: int, time: float, 
                 mutation_rate: float, crossover_rate: float, augmentation_techinque: Union[Dict, bool]):

        with open(config_path, "r") as file:
            self.config = json.load(file)
        
        self.population_size = population_size
        self.time = time
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[TakuNetModel] = []
        self.embeedingList:Union[List[np.ndarray],None ] = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.augmentaion = augmentation_techinque
    
    def _load_data(self,augmentation_technique: Union[Dict, bool]):
        """Loads the dataset using the get_dataset function from data_processing.py"""
        num_classes = self.config["model_search_space"]["refiner_block"]["num_output_classes"]

        return get_dataset(output_classes=num_classes, augementation_technique=augmentation_technique)
    
    def _initialize_population(self)->None:
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
            
            model = TakuNetModel(model_name=f"TakuNet_Init_{created}", 
                                 input_shape=(32, 32, 3), 
                                 model_params=model_params, 
                                 train_params=train_params, 
                                 folder="NAS")
            
            if model.check_trainability():
                self.population.append(model)
                self.embeedingList.append(simple_architecture_embedding(model_params))
                created += 1
                print(f"✅ Added model {model.model_name} to population (total: {created})")
            else:
                print(f"❌ Skipping model {model.model_name} due to memory limits")
                del model

        gc.collect()
        print("🧹 Garbage collection triggered after population initialization loop.")


        if created < self.population_size:
            print(f"⚠️ Only {created}/{self.population_size} models were valid after {attempts} attempts.")

        self.x_train, self.y_train, self.x_test, self.y_test = get_dataset( output_classes= self.config["model_search_space"]["refiner_block"]["num_output_classes"], 
                                                augementation_technique=self.augmentaion)
        for model in self.population:
            model.train(x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test)

            
    
    def _build_ranknet(self):
        print("🛠 Building and training initial RankNet surrogate model...")
        input_dim = self.embeedingList[0].shape[0]
        self.ranknet = build_ranknet(input_dim)

        pairs, labels = self._generate_training_pairs()
        self.ranknet.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=20, batch_size=16, verbose=0)
    
    def _generate_training_pairs(self)->Tuple[List[Tuple[np.ndarray,np.ndarray]],List[int]]:
        pairs:List[Tuple[np.ndarray,np.ndarray]] = []
        labels:List[int] = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                model_i = self.population[i]
                model_j = self.population[j]

                pairs.append((model_i.embedded, model_j.embedded))
                better = 1 if self._fitness(model_i) >= self._fitness(model_j) else 0
                labels.append(better)

        pairs = np.array(pairs)
        labels = np.array(labels)
        return pairs, labels

    def _fitness(self, model: TakuNetModel) ->Union[float,None]:
        return model.results.test_accuracy if model.results.test_accuracy else None
    
    def _select_parents(self) -> List[TakuNetModel]:
        """Selects parents using 1v1 tournament style; last 3 form a mini-tournament if population is odd."""
        shuffled = random.sample(self.population, len(self.population))  # Random order
        selected_parents = []
        i = 0
        while i < len(shuffled) - 1:
            # If 3 models left at the end, do a 3-way match
            if i + 2 == len(shuffled):
                trio = shuffled[i:i+3]
                best:TakuNetModel = self._ranknet_best(trio)

                if best.is_trainable is False:
                    best.train(x_train=self.x_train,
                               y_train=self.y_train,
                               x_test=self.x_test,
                               y_test=self.y_test)

                selected_parents.append(best)
                break
            else:
                model1, model2 = shuffled[i], shuffled[i+1]
                best = self._ranknet_better(model1, model2)
                
                if best.is_trainable is False:
                    best.train(x_train=self.x_train,
                               y_train=self.y_train,
                               x_test=self.x_test,
                               y_test=self.y_test)

                selected_parents.append(best)
                i += 2

        return selected_parents

    
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

        while True:
            child = TakuNetModel(model_name=model_name, 
                                 input_shape=(32, 32, 3), 
                                 model_params=child_params, 
                                 train_params=train_params, 
                                 x_train=None, 
                                 y_train=None, 
                                 x_test=None, 
                                 y_test=None,
                                 folder="NAS")
            if child.is_trainable:
                return child
            else:
                print(f"❌ Crossover {model_name} failed due to memory limits. Retrying...")
                child_params = self._crossover(child_params)
    

    def _ranknet_better(self, model1: TakuNetModel, model2: TakuNetModel) -> TakuNetModel:
        """Predict which model is better using RankNet."""
        embed1 = np.expand_dims(model1.embedded, axis=0)
        embed2 = np.expand_dims(model2.embedded, axis=0)
        pred = self.ranknet.predict([embed1, embed2], verbose=0)
        return model1 if pred[0][0] > 0.5 else model2

    def _ranknet_best(self, models: List[TakuNetModel]) -> TakuNetModel:
        """Select the best model among 3 competitors based on pairwise wins."""
        win_counts = [0] * len(models)

        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    embed_i = np.expand_dims(models[i].embedded, axis=0)
                    embed_j = np.expand_dims(models[j].embedded, axis=0)
                    pred = self.ranknet.predict([embed_i, embed_j], verbose=0)
                    if pred[0][0] > 0.5:
                        win_counts[i] += 1

        winner_index = np.argmax(win_counts)
        return models[winner_index]

    
    def evolve(self)->Iterator[TakuNetModel]:
        """Runs the evolutionary search process."""
        start_time = time.time()
        max_duration_seconds = self.time * 3600

        self._initialize_population() # Here we create 10 un-trained Models
        self._build_ranknet()

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
                    child:TakuNetModel = self._crossover(parent1, parent2, model_number)
                    new_population.append(self._crossover(parent1, parent2, model_number))
                else:
                    model_number = model_number + 1
                    mutant_params = self._mutate(copy.deepcopy(random.choice(parents).model_params))
                    train_params = copy.deepcopy(parents[0].train_params)
                    model_name = f"TakuNet_Mutant_{model_number}"
                    while True:
                        mutant = TakuNetModel(model_name=model_name,
                                              input_shape=(32, 32, 3),
                                              model_params=mutant_params,
                                              train_params=train_params,
                                              x_train=None,
                                              y_train=None,
                                              x_test=None,
                                              y_test=None,
                                              folder="NAS")
                        if mutant.is_trainable:
                            child:TakuNetModel = mutant
                            break
                        else:
                            print(f"❌ Mutation {model_name} failed due to memory limits. Retrying...")
                            mutant_params = self._mutate(mutant_params)
                    new_population.append(child)


            # update the population
            self.population = new_population