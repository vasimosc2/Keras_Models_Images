import gc
import os
import random
import copy
import json
from typing import Iterator, List, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from SurrogateComparisson.Embedding import simple_architecture_embedding
from TakuNet import TakuNetModel
from data_processing import get_dataset
import time
from utils import getSearchSpaceParameters, getTrainingParameters
from SurrogateComparisson.RankNet import build_ranknet
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tarfile")

class EvolutionarySearch:
    def __init__(self, 
                 config_path: str, 
                 population_size: int, 
                 time: float, 
                 mutation_rate: float, 
                 crossover_rate: float, 
                 augmentation_techinque: Union[Dict, bool],
                 folder:Optional[str] = None,
                 seed:Optional[int] = None,
                 hardwareConstrains:Optional[bool] = False,
                 performaceStoppage:Optional[bool] = False,
                 early_stopping_acc:Optional[bool] = False,
                 midway_callback:Optional[bool] = False,
                 strategy:Optional[str] = None,
                 use_ranknet: Optional[bool] = True):

        with open(config_path, "r") as file:
            self.config = json.load(file)
        
        self.population_size = population_size
        self.time = time
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.discoveredModels: List[TakuNetModel] = []
        self.population: List[TakuNetModel] = []
        self.embeedingList:Union[List[np.ndarray],None ] = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.augmentaion = augmentation_techinque
        self.folderName:str = folder if folder is not None else "NAS"
        self.seed:Optional[int]  = seed
        self.hardwareConstrains:bool = hardwareConstrains
        self.performaceStoppage:bool = performaceStoppage
        self.early_stopping_acc:bool = early_stopping_acc
        self.midway_callback:bool = midway_callback
        self.strategy:str = strategy if strategy is not None else "cosine"
        self.use_ranknet:bool = use_ranknet
    
    def _load_data(self,augmentation_technique: Union[Dict, bool]):
        """Loads the dataset using the get_dataset function from data_processing.py"""
        num_classes = self.config["model_search_space"]["refiner_block"]["num_output_classes"]

        return get_dataset(output_classes=num_classes, augementation_technique=augmentation_technique)
    
    def _initialize_population(
            self,
            mode: str = "random",                  # "random" (default) or "load"
            seed: Optional[int] = None,               # used to pick the JSON file name when mode="load" or saving
            folder: str = "init_pops",             # where we save/load params
        )->None:
        """ Creates the initial population of models, 
            skipping untrainable ones,
            Train trainable ones!"""
        
        print("🚀 Initializing population...")
        created = 0
        attempts = 0
        max_attempts = self.population_size * 300  # Prevent infinite loop in rare cases

        # ----------------------------------------------------
        # LOAD model_params if requested
        # ----------------------------------------------------
        def _initpop_params_path(seed_val: int, folder_val: str = "init_pops") -> str:
            os.makedirs(f"{folder_val}", exist_ok=True)
            if self.hardwareConstrains:
                return f"{folder_val}/Constrained/init_population_seed_{seed_val}.json"
            else:
                return f"{folder_val}/UnConstrained/init_population_seed_{seed_val}.json"
        
        saved_list = None

        if mode.lower() == "load":
            if seed is None:
                raise ValueError("When mode='load', a 'seed' must be provided.")
            filename = _initpop_params_path(seed, folder)
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"❌ Could not find model_params file: {filename}. "
                    f"Run once with mode='random' to create it."
                )
            print(f"📂 Loading model parameters from {filename}")
            with open(filename, "r") as f:
                saved_list = json.load(f)

        while created < self.population_size and attempts < max_attempts:
            attempts += 1


            if saved_list is not None and len(saved_list) > 0:
                model_params = saved_list.pop(0)
            else:
                if mode.lower() == "load":
                    break
                model_params = getSearchSpaceParameters.sample_from_search_space(self.config["model_search_space"])

            #model_params = getSearchSpaceParameters.sample_from_search_space(self.config["model_search_space"])
            train_params = getTrainingParameters.sample_from_train_and_evaluate(self.config["train_and_evaluate"])
            
            model = TakuNetModel(model_name=f"TakuNet_Init_{created}", 
                                 input_shape=(32, 32, 3), 
                                 model_params=model_params, 
                                 train_params=train_params, 
                                 folder=self.folderName,
                                 hardwareConstrains=self.hardwareConstrains,
                                 performaceStoppage=self.performaceStoppage,
                                 early_stopping_acc=self.early_stopping_acc,
                                 midway_callback=self.midway_callback,
                                 lr_schedule_strategy=self.strategy)
            
            if model.is_trainable:
                self.population.append(model)
                self.embeedingList.append(simple_architecture_embedding(model_params))
                created += 1
                src = "file" if mode.lower() == "load" else "random"
                print(f"✅ Added model {model.model_name} ({src}) to population (total: {created})")
            else:
                print(f"❌ Skipping model {model.model_name} due to memory limits")
                del model

        gc.collect()
        print("🧹 Garbage collection triggered after population initialization loop.")


        if created < self.population_size:
            raise RuntimeError(
            f"❌ Failed to create full population. Only {created}/{self.population_size} models were valid after {attempts} attempts. "
            f"Consider relaxing resource constraints or adjusting the search space."
        )

        # ----------------------------------------------------
        # SAVE model_params (only for mode="random")
        # ----------------------------------------------------
        if mode.lower() == "random" and seed is not None:
            filename = _initpop_params_path(seed, folder)
            payload = [m.model_params for m in self.population]
            with open(filename, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"💾 Saved {len(payload)} model_params to {filename}")

        # ----------------------------------------------------
        # Train all models (unchanged)
        # ----------------------------------------------------    
        self.x_train, self.y_train, self.x_test, self.y_test = get_dataset(output_classes= self.config["model_search_space"]["refiner_block"]["num_output_classes"], 
                                                                           augementation_technique=self.augmentaion)
        for model in self.population:
            model.train(x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test)

            self.discoveredModels.append(model)

            
    
    def _build_ranknet(self):
        print("🛠 Building or loading RankNet surrogate model...\n")

        input_dim = self.embeedingList[0].shape[0]
        base_dir = "SurrogateComparisson/ConstrainedVersion" if self.hardwareConstrains else "SurrogateComparisson/UnConstrainedVersion"
        os.makedirs(base_dir, exist_ok=True)

        model_path = os.path.join(base_dir, "ranknet_model.keras")
        data_path = os.path.join(base_dir, "ranknet_training_data.npz")

        # 1) If we already have a trained/saved model, just load it and stop here.
        if os.path.exists(model_path):
            print("📦 Loading existing RankNet model (no retraining)...\n")
            self.ranknet = tf.keras.models.load_model(model_path)
            return  # ✅ don't generate more data, don't retrain

        # 2) Otherwise, we need to build a fresh model
        print("✨ No saved RankNet found, building a new one...\n")
        self.ranknet = build_ranknet(input_dim)

        # 3) Get training data: either load fixed data or generate it once
        if os.path.exists(data_path):
            print("📂 Found existing RankNet training data. Using it...\n")
            data = np.load(data_path)
            pairs = data["pairs"]
            labels = data["labels"]
        else:
            print("🆕 No training data found. Generating once...\n")
            pairs, labels = self._generate_training_pairs()
            # save it so next runs reuse the SAME data
            np.savez_compressed(data_path, pairs=pairs, labels=labels)
            print(f"💾 Saved training data: {pairs.shape[0]} pairs total.\n")

        print("🏋️ Training RankNet on fixed dataset...\n")
        self.ranknet.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=20, batch_size=16, verbose=0)

        self.ranknet.save(model_path)
        print("💾 RankNet model saved after training.\n")
    
    def _generate_training_pairs(self)->Tuple[List[Tuple[np.ndarray,np.ndarray]],List[int]]:
        pairs:List[Tuple[np.ndarray,np.ndarray]] = []
        labels:List[int] = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                model_i = self.population[i]
                model_j = self.population[j]

                pairs.append(( simple_architecture_embedding(model_i.model_params),  simple_architecture_embedding(model_j.model_params)))
                better = 1 if model_i.results.fitness_score >= model_j.results.fitness_score else 0
                labels.append(better)

        pairs = np.array(pairs)
        labels = np.array(labels)
        return pairs, labels
    
    def _select_parents(self) -> List[TakuNetModel]:
        """Selects parents using 1v1 tournament style; last 3 form a mini-tournament if population is odd."""
        shuffled = random.sample(self.population, len(self.population))  # Random order
        i = 0
        parents = []

        while i < len(shuffled):
            remaining = len(shuffled) - i

            if remaining == 3:
                # Special case: 3 models left
                trio = shuffled[i:i+3]
                print(f"📌 Comparing {trio[0].model_name}, {trio[1].model_name}, and {trio[2].model_name} ....\n")
                best: TakuNetModel = self._pick_best(models=trio)

                print(f"The winner is {best.model_name} with fitness {best.results.fitness_score} 🏆\n")
                print(f"The Competitor 0 {trio[0].model_name} with fitness { trio[0].results.fitness_score if trio[0].is_trained else 'None'}\n")
                print(f"The Competitor 1 {trio[1].model_name} with fitness { trio[1].results.fitness_score if trio[1].is_trained else 'None'}\n")
                print(f"The Competitor 2 {trio[2].model_name} with fitness { trio[2].results.fitness_score if trio[2].is_trained else 'None'}\n")

                if best.is_trainable and not best.is_trained:
                    print(f"Therotically I am never here :) \n")
                    best.train(
                        x_train=self.x_train,
                        y_train=self.y_train,
                        x_test=self.x_test,
                        y_test=self.y_test
                    )
                
                i += 3
            else:
                # Normal case: 2 models
                duo = shuffled[i:i+2]
                print(f"🥊 Comparing {duo[0].model_name} and {duo[1].model_name} ....\n")
                best: TakuNetModel = self._pick_best(models=duo)

                print(f"The winner is {best.model_name} with fitness {best.results.fitness_score} 🏆\n")
                print(f"The Competitor 0 {duo[0].model_name} with fitness {duo[0].results.fitness_score if duo[0].is_trained else 'None'}\n")
                print(f"The Competitor 1 {duo[1].model_name} with fitness {duo[1].results.fitness_score if duo[1].is_trained else 'None'}\n")

                # if best.is_trainable and not best.is_trained:
                #     print(f"Therotically I am never here :) \n")
                #     best.train(
                #         x_train=self.x_train,
                #         y_train=self.y_train,
                #         x_test=self.x_test,
                #         y_test=self.y_test
                #     )

                i += 2

            parents.append(best)


        return parents

    
    def _mutate(self, model_params: Dict) -> Dict:
        """
        Mutate the model parameters based on mutation rate.
        Handles nested dictionaries and top-level primitives like 'optimizer'.
        """
        for block in model_params:
            try:
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
                                    print(f"⚠️ Skipping mutation for {block}.{subBlock} (fixed value: {choices}). Should be the Classes that we use \n")
                else:
                    if random.random() < self.mutation_rate:
                        choices = self.config["model_search_space"][block]
                        model_params[block] = random.choice(choices)
            except Exception as e:
                raise Exception(f"❌ Mutation failed at block '{block}' with value '{model_params[block]}'. Error: {str(e)}")
        return model_params


    def _crossover(self, parent1: TakuNetModel, parent2: TakuNetModel, model_number: int) -> TakuNetModel:
        """
        Perform crossover between two parent models.
        Preserves exception handling and reports invalid blocks like 'optimizer'.
        """
        model_name = f"TakuNet_Crossover_{model_number}"
        train_params = copy.deepcopy(parent1.train_params)

        while True:
            child_params = copy.deepcopy(parent1.model_params)

            try:
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
            except Exception as e:
                raise Exception(f"I failed in the mutation because of block '{block}' and value '{child_params[block]}'. Error: {str(e)}")

            # Try to build the model
            child = TakuNetModel(model_name=model_name, 
                                input_shape=(32, 32, 3), 
                                model_params=child_params, 
                                train_params=train_params, 
                                x_train=None, 
                                y_train=None, 
                                x_test=None, 
                                y_test=None,
                                folder=self.folderName,
                                hardwareConstrains=self.hardwareConstrains,
                                performaceStoppage=self.performaceStoppage,
                                early_stopping_acc=self.early_stopping_acc,
                                midway_callback=self.midway_callback,
                                lr_schedule_strategy=self.strategy)
            
            if child.is_trainable:
                return child
            else:
                print(f"❌ Crossover {model_name} failed due to memory limits. Retrying...")
                del child
                tf.keras.backend.clear_session()
                gc.collect()

    def _ranknet_best(self, models: List[TakuNetModel]) -> TakuNetModel:
        """
        Select the best model among N competitors using RankNet and true fitness.
        - If all are trained → return one with best fitness.
        - If none are trained → pick RankNet winner and train only them.
        - If some are trained:
            → pick RankNet winner
            → train only if needed
            → compare its fitness to trained others
            → return true best
        """
        if len(models) < 2:
            raise ValueError("At least two models are required for comparison.")

        # ✅ Case 1: All are trained → pick by fitness
        if all(m.is_trained for m in models):
            best_model:TakuNetModel = max(models, key=lambda m: m.results.fitness_score)
            print(f"💪 All Competitors were trained. Selected {best_model.model_name} by fitness.\n")
            return best_model

        # 🔁 Case 2: All the models are not trained
        # Use pairwise RankNet votes
        vote_counts = [0] * len(models)

        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                embed_i = np.expand_dims(simple_architecture_embedding(models[i].model_params), axis=0)
                embed_j = np.expand_dims(simple_architecture_embedding(models[j].model_params), axis=0)
                 # Use RankNet to predict winner
                pred = self.ranknet.predict([embed_i, embed_j], verbose=0)

                if pred[0][0] > 0.5: # Is embedding_i more promissing ?
                    vote_counts[i] += 1 # Give 1 vote to embedding i
                else:
                    vote_counts[j] += 1 # Give 1 vote to embedding j

        winner_index = int(np.argmax(vote_counts))
        predicted_winner = models[winner_index]

        # 🛠 Train the expected winner if needed
        if not predicted_winner.is_trained:
            print(f"🚀 RankNet picked {predicted_winner.model_name}. Training now...\n")
            predicted_winner.train(x_train=self.x_train, y_train=self.y_train,
                                x_test=self.x_test, y_test=self.y_test)
            
            self.discoveredModels.append(predicted_winner)

        # ✅ Compare with already-trained competitors
        # Here we take the index and the model of that index of the losers
        for i, opponent in enumerate(models):
            if i != winner_index and opponent.is_trained:
                if opponent.results.fitness_score > predicted_winner.results.fitness_score:
                    print(f"❌ RankNet mistake: {opponent.model_name} (fitness={opponent.results.fitness_score:.4f}) "
                        f"> {predicted_winner.model_name} (fitness={predicted_winner.results.fitness_score:.4f})")
                    return opponent

        return predicted_winner



    def _true_best(self, models: List[TakuNetModel]) -> TakuNetModel:
        """Train any untrained candidates, then return the best by true fitness."""
        for m in models:
            if not m.is_trained:
                m.train(x_train=self.x_train, y_train=self.y_train,
                        x_test=self.x_test, y_test=self.y_test)
                self.discoveredModels.append(m)
        return max(models, key=lambda m: m.results.fitness_score)

    def _pick_best(self, models: List[TakuNetModel]) -> TakuNetModel:
        """Dispatch to RankNet or ground-truth selection based on flag."""
        if self.use_ranknet:
            return self._ranknet_best(models)
        else:
            return self._true_best(models)

    def evolve(self)->Iterator[TakuNetModel]:
        """Runs the evolutionary search process."""
        start_time = time.time()
        max_duration_seconds = self.time * 3600

        self._initialize_population( mode=os.getenv("INITIAL_POPULATION_MODE", "random"), seed= self.seed )

        if self.use_ranknet: 
            self._build_ranknet()

        model_number = 0

        while time.time() - start_time < max_duration_seconds:

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
                    child:TakuNetModel = self._crossover(parent1=parent1, parent2=parent2, model_number=model_number)
                    new_population.append(child)
                else:
                    model_number = model_number + 1
                    mutant_params = self._mutate(model_params=copy.deepcopy(random.choice(parents).model_params))
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
                                              folder=self.folderName,
                                              hardwareConstrains=self.hardwareConstrains,
                                              performaceStoppage=self.performaceStoppage,
                                              early_stopping_acc=self.early_stopping_acc,
                                              midway_callback=self.midway_callback,
                                              lr_schedule_strategy=self.strategy)
                        if mutant.is_trainable:
                            child:TakuNetModel = mutant
                            break
                        else:
                            print(f"❌ Mutation {model_name} failed due to memory limits. Retrying...")
                            del mutant
                            tf.keras.backend.clear_session()
                            gc.collect()
                            mutant_params = self._mutate(model_params=mutant_params)
                    new_population.append(child)

            tf.keras.backend.clear_session()
            gc.collect()

            # update the population
            self.population = new_population
        
        for i in self.discoveredModels:
            yield i