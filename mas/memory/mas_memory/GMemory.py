from dataclasses import dataclass, replace
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import os
import copy
import re
from typing import Iterable
import random
from collections import defaultdict
import networkx as nx
import numpy as np
from finch import FINCH
import pickle
import networkx as nx
import logging

from .memory_base import MASMemoryBase
from ..common import MASMessage, StateChain
from ..utils import cosine_similarity
from .prompt import GMemoryPrompts
from mas.utils import load_json, write_json, random_divide_list
from mas.llm import LLMCallable, Message

@dataclass
class GMemory(MASMemoryBase):
    """
    G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
    A three-tier hierarchical graph structure compo sed of the Insight Graph, Query Graph, and Interaction Graph.

    1. Interaction Graph - Trajectory Condensation: During the task-solving process, the multi-agent system (MAS) generates a chain of states, where each state represents a step in the process of arriving at the final answer. Behind each state is a corresponding message graph.
       Each task corresponds to a chain of states, which connects the middle and bottom layers of the multi-layer graph.
    2. Query Graph - Based on the current task, the system retrieves previously successful records. A k-hop approach is used to expand the search scope within the query graph.
    3. Insight Graph - Insights Retrieval: Relevant insights are retrieved based on the current task to assist in decision-making.
    """
    def __post_init__(self):
        super().__post_init__()
        
        self.main_memory = Chroma(          
            embedding_function=self.embedding_func,  
            persist_directory=self.persist_dir         
        )

        self._hop: int = self.global_config.get('hop', 1)
        self._start_insights_threshold: int = self.global_config.get('start_insights_threshold', 5)
        self._rounds_per_insights: int = self.global_config.get('rounds_per_insights', 5) 
        self._insights_point_num: int = self.global_config.get('insights_point_num', 5)

        self.task_layer = TaskLayer(
            working_dir=self.persist_dir,
            namespace='task_layer', 
            task_storage=self.main_memory
        )

        self.insights_layer = InsightsManager(
            working_dir=self.persist_dir, 
            namespace='insights', 
            llm_model=self.llm_model, 
            task_storage=self.main_memory,
            task_layer=self.task_layer
        )

        self.insights_cache: list[str] = []

        print(self._get_hyperparams_dict())
    
    def _get_hyperparams_dict(self) -> dict:
        return {
            'hop': self._hop,
            'start_insights_threshold': self._start_insights_threshold,
            'rounds_per_insights': self._rounds_per_insights,
            'insights_point_num': self._insights_point_num,
            'working_dir': self.persist_dir
        }


    def add_memory(self, mas_message: MASMessage) -> None:
        """
        Add the mas_message corresponding to a completed task into memory:
        Step 1: Sparsification - remove incorrect steps
        Step 2: Add the sparsified trajectories to memory
        Step 3: If the number of steps in memory reaches a certain threshold, perform fine-tuning on the insights in memory

        Args:
            mas_message (MASMessage): The MAS message corresponding to a completed task

        Raises:
            ValueError: mas_message must have label!
        """
        # sparsification
        mas_message = self._extract_mas_message(mas_message=mas_message)  
        
        # add into memory
        self.task_layer.add_task_node(mas_message.task_main)

        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,   
            metadata=meta_data
        )
        if mas_message.label == True or mas_message.label == False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError('The mas_message must have label!')
        
        # finetune and merge insights
        if self.memory_size >= self._start_insights_threshold and self.memory_size % self._rounds_per_insights == 0:
            self.insights_layer.finetune_insights(self._insights_point_num)
        if self.memory_size % 20 == 0: 
            self.insights_layer.merge_insights() 

        self._index_done()

    def _retrieve_memory_raw(
        self, 
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        insight_windows: int = 10,
        threshold: float = 0.3
    ) -> tuple[list, list, list]:

        def sort_and_filter_by_similarity(docs: list[Document], threshold: float = 0.3) -> list[tuple[Document, float]]:
            result = []
            for doc in docs:
                embedding = self.embedding_func.embed_query(doc.page_content)
                sim = cosine_similarity(origin_embedding, embedding)
                if sim >= threshold:
                    result.append((doc, sim))

            result.sort(key=lambda x: x[1], reverse=True)
            return result

        true_tasks_doc: list[Document] = []
        false_tasks_doc: list[Document] = []
        
        # find related tasks in task layer
        related_point_num: int = max((successful_topk + failed_topk) // 2, 1)
        task_mains: list[str] = self.task_layer.retrieve_related_task(query_task=query_task, node_num=related_point_num, hop=self._hop)
        for task_main in task_mains:
            doc = self.main_memory.similarity_search(task_main, k=1)[0]

            if doc.metadata.get('label') == True:
                true_tasks_doc.append(doc)
            elif doc.metadata.get('label') == False:
                false_tasks_doc.append(doc)
            else:
                raise RuntimeError('The document object\'s metadata should have `label` attribute.')
        
        # If the specified number is not met, fill in the rest using similarity-based augmentation.
        if len(true_tasks_doc) < successful_topk:
            true_tasks_doc = self.main_memory.similarity_search(
                query=query_task, k=successful_topk, filter={'label': True}
            )
            for doc in true_tasks_doc:
                if doc not in true_tasks_doc:
                    true_tasks_doc.append(doc)
        
        if len(false_tasks_doc) < failed_topk:
            false_tasks_doc = self.main_memory.similarity_search(
                query=query_task, k=failed_topk, filter={'label': False}
            )
            for doc in false_tasks_doc:
                if doc not in false_tasks_doc:
                    false_tasks_doc.append(doc)

        # order by similarity        
        origin_embedding: list[float] = self.embedding_func.embed_query(query_task)
        true_tasks_doc_with_score = sort_and_filter_by_similarity(true_tasks_doc, threshold)[:successful_topk]
        false_tasks_doc_with_score = sort_and_filter_by_similarity(false_tasks_doc, threshold)[:failed_topk]

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc, _ in true_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc, _ in false_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)
        
        # get insights and order by relelvance
        insights_with_score = self.insights_layer.query_insights_with_score(query_task, top_k=insight_windows)
        insights = [insight for insight, _ in insights_with_score][:insight_windows]

        return true_task_messages, false_task_messages, insights

    def retrieve_memory(
        self, 
        query_task: str,         
        successful_topk: int = 2, 
        failed_topk: int = 1,
        insight_topk: int = 10,
        threshold: float = 0.3,
        **args
    ) -> tuple[list, list, list]: 
        """Access the memory and return the results.

        Args:
            query_task (str): The task to query.
            successful_topk (int, optional): Number of successful cases to retrieve. Defaults to 2.
            failed_topk (int, optional): Number of failed cases to retrieve. Defaults to 1.
            insight_topk (int, optional): Number of insights to retrieve. Defaults to 10.
            threshold (float, optional): Similarity threshold for retrieving cases. Defaults to 0.3.

        Returns:
            tuple[list, list, list]: A tuple containing successful cases, failed cases, and insights.
        """
        
        # retrieve raw tasks
        successful_task_trajectories: list[MASMessage]
        failed_task_trajectories: list[MASMessage]
        insights: list[str]
        successful_task_trajectories, failed_task_trajectories, insights = self._retrieve_memory_raw(
            query_task, 2*successful_topk, 2*failed_topk, 2*insight_topk, threshold)
        
        # retrieve tasks based on task relevance
        importance_score: list[float] = []
        for success_task in successful_task_trajectories:
            prompt: str = GMemoryPrompts.generative_task_user_prompt.format(
                trajectory=success_task.task_description + '\n' + success_task.task_trajectory,
                query_scenario=query_task
            )
            response: str = self.llm_model(messages=[Message('system', GMemoryPrompts.generative_task_system_prompt), 
                                                     Message('user', prompt)])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_score.append(score)
        
        sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories), 
                                                           key=lambda x: x[0], reverse=True)]
        top_success_task_trajectories = sorted_success_tasks[:successful_topk]
        top_success_task_trajectories = successful_task_trajectories[:successful_topk]
        
        # directly get failed tasks
        top_fail_task_trajectories = failed_task_trajectories[:failed_topk]
        
        # directlt get insights
        top_k_insights = insights[:insight_topk]
        self.insights_cache = top_k_insights

        return top_success_task_trajectories, top_fail_task_trajectories, top_k_insights


    def _extract_mas_message(self, mas_message: MASMessage) -> MASMessage:

        mas_message_copy: MASMessage = copy.deepcopy(mas_message)
        state_chain: StateChain = mas_message_copy.chain_of_states
        
        for state_id in reversed(range(len(state_chain))):
            if state_chain.get_state(state_id).graph.get('reward', 0) < 0:
                state_chain.pop_state(state_id)
        
        trajectory = ''
        for state in state_chain:
            trajectory += f'> {state.graph['action']}\n{state.graph['observation']}\n'
        
        if mas_message_copy.label == True:
            mas_message_copy.task_trajectory = trajectory

        
        trajectory = re.sub(r'\d+', '', trajectory)
        mas_message_copy.add_extra_field('clean_traj', trajectory)


        system_prompt = GMemoryPrompts.extract_true_traj_system_prompt
        prompt_template = GMemoryPrompts.extract_true_traj_user_prompt

        prompt: str = prompt_template.format(
            task=mas_message_copy.task_description,
            trajectory=mas_message_copy.get_extra_field('clean_traj')
        )
        messages: list[Message] = [Message('system', system_prompt), Message('user', prompt)]
        response: str = self.llm_model(messages, temperature=0.1)
        mas_message_copy.add_extra_field('key_steps', response)


        if mas_message_copy.label == False:
            reason: str = self._detect_mistakes(mas_message_copy)
            mas_message_copy.add_extra_field('fail_reason', reason)
        
        return mas_message_copy
    
    
    def _detect_mistakes(self, mas_message: MASMessage) -> str:
        user_prompt: str = GMemoryPrompts.detect_mistakes_user_prompt.format(task=mas_message.task_description, trajectory=mas_message.get_extra_field('clean_traj'))
        messages: list[Message] = [Message('system', GMemoryPrompts.detect_mistakes_system_prompt), 
                                   Message('user', user_prompt)]
        response: str = self.llm_model(messages)

        return response

    def backward(self, reward: bool):

        for insight in self.insights_cache:
            self.insights_layer.backward(insight, reward=-2 if reward == False else 1)

        self.insights_cache = []
    
    @property
    def memory_size(self):
        num_records = self.main_memory.get()["ids"]
        return len(num_records)
    
    def project_insights(self, raw_insights: list[str], role: str = None, task_traj: str = None) -> list[str]:
        """
        Projects raw insights into role-specific insights based on the given role and optionally a task trajectory.

        Args:
            raw_insights (list[str]): A list of raw insight strings.
            role (str, optional): The role to tailor the insights for. Defaults to None.
            task_traj (str, optional): A string representing the task trajectory. Defaults to None.

        Returns:
            list[str]: A list of processed insights tailored to the specified role.
        """
        def parse_numbered_list(text: str) -> list[str]:
            pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
            items = re.findall(pattern, text.strip(), flags=re.DOTALL)
            return [item.strip() for item in items]
        
        # If no role is provided, return the raw insights as they are.
        if not role:
            return raw_insights
        
        # Determine which system and user prompts to use based on whether a task trajectory is provided
        raw_insights_str = '\n'.join(raw_insights)
        if not task_traj:
            system_prompt = GMemoryPrompts.project_insights_system_prompt
            user_prompt: str = GMemoryPrompts.project_insights_user_prompt.format(
                role=role,
                insights=raw_insights_str
            )
        else:
            system_prompt = GMemoryPrompts.project_insights_with_traj_system_prompt
            user_prompt: str = GMemoryPrompts.project_insights_with_traj_user_prompt.format(
                role=role,
                insights=raw_insights_str,
                trajectory=task_traj
            )
        messages = [Message('system', system_prompt),
                    Message('user', user_prompt)]
        
        # Use the language model to generate role-specific insights
        role_insights = self.llm_model(messages)

        try: 
            role_insights = parse_numbered_list(role_insights)
            return role_insights
        except:
            return raw_insights

@dataclass
class TaskLayer:
    
    working_dir: str
    namespace: str
    task_storage: Chroma
    
    def __post_init__(self):
        self.similarity_threshold = 0.7

        self._graph_pic_save_path: str = os.path.join(self.working_dir, 'graph.png')
        self._node_match_save_path: str = os.path.join(self.working_dir, 'match_nodes.txt')
        self._graph_save_path: str = os.path.join(self.working_dir, f'{self.namespace}_graph.pkl')

        if os.path.exists(self._graph_save_path):
            with open(self._graph_save_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"Graph loaded from {self._graph_save_path}")
        else:
            self.graph = nx.Graph()
            print("New empty graph created")

    def add_task_node(self, task_main: str) -> None:
        """Add a task node to the task graph.

        Args:
            task_main (str): task name
        """
        if task_main in self.graph:
            return  

        self.graph.add_node(task_main)

        results: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(
            query=task_main,
            k=10 
        )
        
        for doc, distance in results:
            similarity = 1 - distance
            if similarity < self.similarity_threshold:
                continue  

            neighbor = doc.page_content

            if neighbor not in self.graph:
                self.graph.add_node(neighbor)

            self.graph.add_edge(task_main, neighbor, weight=similarity) 
        
        self._index_done()
 
    def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list[str]:
        """
        Retrieve related tasks from the graph based on similarity and local neighborhood expansion.

        Args:
            query_task (str): The task used as the query input.
            node_num (int): The number of top similar tasks to retrieve based on similarity scores.
            hop (int, optional): The number of hops used to expand the neighborhood in the graph. Defaults to 1.

        Returns:
            list[str]: A list of related task nodes, including top similar tasks and their neighbors within the given hop.
        """
        tasks: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(query=query_task, k=node_num)
        top_nodes = [doc[0].page_content for doc in tasks]

        related_nodes = set(top_nodes)
        for node in top_nodes:
            neighbours = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
            related_nodes.update(neighbours)
        return list(related_nodes)
    
    def cluster_tasks(self) -> None:
        """
        Perform clustering on tasks in the graph using their embeddings and assign cluster IDs.

        This method extracts all nodes from the graph, computes embeddings for each node using the
        task storage's embedding function, and applies the FINCH clustering algorithm with cosine similarity.
        """
        nodes = list(self.graph.nodes)

        embeddings = []
        valid_nodes = []

        for node in nodes:
            embedding = self.task_storage._embedding_function.embed_query(node)  
            if embedding is not None:
                embeddings.append(embedding)
                valid_nodes.append(node)

        X = np.vstack(embeddings)
        fin = FINCH(metric='cosine')

        try: 
            labels = fin.fit_predict(X)
        except Exception as e:   
            print(f"FINCH clustering failed: {e}")
            labels = np.zeros(len(valid_nodes), dtype=int)

        for node, label in zip(valid_nodes, labels):
            self.graph.nodes[node]['cluster_id'] = int(label)
        self._index_done()

    def _index_done(self) -> None:
        
        with open(self._graph_save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def __iter__(self) -> Iterable[tuple[str, int]]: 
        return ((node, self.graph.nodes[node]['cluster_id']) for node in self.graph.nodes)

    


@dataclass
class InsightsManager:

    working_dir: str
    namespace: str
    llm_model: LLMCallable
    task_storage: Chroma
    task_layer: TaskLayer
    def __post_init__(self):
        self.persist_file: str = os.path.join(self.working_dir,f'{self.namespace}.json')
        self.insights_memory: list[dict] = load_json(self.persist_file) or []
       
        log_path = os.path.join(self.working_dir, 'insights.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        

    def query_insights_with_score(self, task_query: str, top_k: int = None) -> list[tuple[str, float]]:

        SUCC_NUM, FAIL_NUM = 4, 2

        related_successful_tasks, related_failed_tasks = self._retrieve_memory(task_query, successful_topk=SUCC_NUM, failed_topk=FAIL_NUM)
        task_mains: list[str] = [task.task_main for task in related_successful_tasks + related_failed_tasks]
        task_mains.append(task_query)
        insights_score = defaultdict(float)
        for task_main in task_mains:
            _, related_insights = self._find_related_insights(task_mains=[task_main])
            for insight in related_insights:
                insights_score[insight.get('rule')] += 1  

        sorted_insights = sorted(insights_score.items(), key=lambda x: x[1], reverse=True) 
        if top_k is not None:
            sorted_insights = sorted_insights[:top_k]
        return sorted_insights
    
    def merge_insights(self) -> None:

        self.task_layer.cluster_tasks()
        
        label_tasks: dict[int, list[str]] = {}
        for task_main, label_id in self.task_layer:
            if label_id is None:
                raise RuntimeError('Label id should not be none.')
            if label_id not in label_tasks.keys():
                label_tasks[label_id] = [task_main]
            else:
                label_tasks[label_id].append(task_main)
        
        merged_label_rules: dict[int, list[str]] = {}
        for task_type, related_task_mains in label_tasks.items():
            related_ids, related_insights = self._find_related_insights(task_mains=related_task_mains)
            related_rules: list[str] = [insight['rule'] for insight in related_insights]
            merged_rules: list[str] = self._merge_rules(related_rules)
            merged_label_rules[task_type] = merged_rules

            self.logger.info('------- Merge Insights -------')
            self.logger.info(f'Task type: {task_type}')
            self.logger.info(f"Origin rules: \n{'\n'.join(related_rules)}")
            self.logger.info(f"Merged rules: \n{'\n'.join(merged_rules)}")
            
        self.insights_memory.clear()

        for label, related_rules in merged_label_rules.items():
            related_task_mains = label_tasks.get(label)
            if related_task_mains is None:
                raise RuntimeError('Inconsistency in `label`')
            
            for rule in related_rules:
                insight: dict = {
                    'rule': rule,
                    'score': 2,          
                    'positive_correlation_tasks': list(related_task_mains),
                    'negative_correlation_tasks': list()
                }
                self.insights_memory.append(insight)
        
        self._index_done()

    def _merge_rules(self, rules: list[str]) -> list[str]:
        def parse_numbered_list(text: str) -> list[str]:
            pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
            items = re.findall(pattern, text.strip(), flags=re.DOTALL)
            return [item.strip() for item in items]
        
        merged_rules = []
        batch_size = 10

        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            actual_num: int = len(batch) // 3  

            user_prompt = GMemoryPrompts.merge_rules_user_prompt.format(
                current_rules='\n'.join(batch),
                limited_number=actual_num//3
            )
            messages = [Message('system', GMemoryPrompts.merge_rules_system_prompt),
                        Message('user', user_prompt)]
            raw_merged_rules = self.llm_model(messages)
            merged_rules.extend(parse_numbered_list(raw_merged_rules))

        return merged_rules

    def backward(self, insight: str, reward: float):
        
        for inner_insight in self.insights_memory:
            if insight in inner_insight['rule']:
                inner_insight['score'] += reward

        self.clear_insights()
        self._index_done()

    def clear_insights(self):
        self.insights_memory = [self.insights_memory[i] for i in range(len(self.insights_memory)) 
                        if self.insights_memory[i]['score'] > 0] 

    def _retrieve_memory(
        self,
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1
    ) -> tuple[list[MASMessage], list[MASMessage]]:

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={'label': True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={'label': False}
            )
        sorted(true_tasks_doc, key=lambda x: x[1]) 
        sorted(false_tasks_doc, key=lambda x: x[1]) 

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)

        return true_task_messages, false_task_messages
    
    @property
    def task_size(self):
        num_records = self.task_storage.get()["ids"]
        return len(num_records)
    
    def _find_related_insights(
        self,
        task_mains: list[str],
        threshold: float = 1
    ) -> tuple[list[int], list[dict]]:

        rule_set: list[tuple[dict, int, int]] = []  # (rule, score, index)

        for idx, rule in enumerate(self.insights_memory):
            score: int = sum(task in rule.get('positive_correlation_tasks', []) for task in task_mains)
            if score >= threshold:
                rule_set.append((rule, score, idx))

        rule_set.sort(key=lambda x: x[1], reverse=True)

        rule_indices = [item[2] for item in rule_set]
        sorted_rules = [item[0] for item in rule_set]

        return rule_indices, sorted_rules
    def finetune_insights(self, num_points: int):

        SUCCESS_TASK_NUM, FAIL_TASK_NUM = 3, 1

        all_ids = self.task_storage.get()['ids']
        for _ in range(num_points):  

            random_id = random.choice(all_ids)
            random_entry = self.task_storage.get(ids=[random_id])
            if 'metadatas' in random_entry and random_entry['metadatas']:
                random_metadata = random_entry['metadatas'][0]  
            else:
                raise RuntimeError('Incomplete data.')
            mas_message: MASMessage = MASMessage.from_dict(random_metadata)


            true_trajs, false_trajs = self._retrieve_memory(
                query_task=mas_message.task_main, successful_topk=SUCCESS_TASK_NUM, failed_topk=FAIL_TASK_NUM
            )
            if mas_message.label == True:
                true_trajs.append(mas_message)
            else:
                false_trajs.append(mas_message)
            all_task_mains: list[str] = [traj.task_main for traj in true_trajs + false_trajs]

            related_insight_ids, _ = self._find_related_insights(all_task_mains, len(all_task_mains) / 2)
            self._finetune_insights(true_trajs, false_trajs, related_insight_ids)
        
        self.clear_insights()
        self._index_done()
    def _finetune_insights(
        self,
        successful_task_trajectories: list[MASMessage],
        failed_task_trajectories: list[MASMessage],
        insight_ids: list[int]
    ) -> None:

        def map_operations(origin_operations: list[tuple]) -> list[tuple]:
            processed_operations: list[tuple] = []
            for (operation, text) in origin_operations:
                res: list = operation.split(' ')

                if len(res) == 2:
                    if len(insight_ids) == 0:    
                        continue
                    insight_id: int = int(res[1]) - 1
                    if insight_id >= len(insight_ids) or insight_id < 0:
                        continue
                    
                    res[1] = str(insight_ids[insight_id] + 1)   
                    operation: str = ' '.join(res)
                processed_operations.append((operation, text))
            
            return processed_operations

        rule_list: list[dict] = [self.insights_memory[i] for i in insight_ids]

        compare_pairs: list[tuple[MASMessage, MASMessage]] = []
        for id, fail_task in enumerate(failed_task_trajectories):
            if id >= len(successful_task_trajectories):
                break
            success_task = successful_task_trajectories[id]
            compare_pairs.append((success_task, fail_task))
        
        successful_task_chunks: list[list[MASMessage]] = random_divide_list(successful_task_trajectories, 5) 
        
        MAX_RULE_THRESHOLD: int = 10
        suffix: str = GMemoryPrompts.finetune_insights_suffix['full'] if len(self.insights_memory) > MAX_RULE_THRESHOLD \
                      else GMemoryPrompts.finetune_insights_suffix['not_full']


        self.logger.info('--------------- Finetune Insights ---------------')
        for pair in compare_pairs:
            compare_prompts: list[Message] = self._build_comparative_prompts(pair[0], pair[1], rule_list)
            compare_prompts[0] = replace(compare_prompts[0], content=compare_prompts[0].content + suffix)
            response: str = self.llm_model(compare_prompts)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            self._update_rules(
                [pair[0].task_main, pair[1].task_main], 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            self.logger.info(compare_prompts[0].role + compare_prompts[0].content + '\n\n' + compare_prompts[1].role + compare_prompts[1].content)
            self.logger.info(response)
            self.logger.info('\n---------------\n')

        for chunk in successful_task_chunks:
            success_prompts: list[Message] = self._build_success_prompts(chunk, rule_list) 
            success_prompts[0] = replace(success_prompts[0], content=success_prompts[0].content + suffix)
            response: str = self.llm_model(success_prompts)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            task_mains: list[str] = [traj.task_main for traj in chunk]
            self._update_rules(
                task_mains, 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            self.logger.info(success_prompts[0].role + success_prompts[0].content + '\n\n' + success_prompts[1].role + success_prompts[1].content)
            self.logger.info(response)
            self.logger.info('\n---------------\n')
        
        self.clear_insights()
        self._index_done()

    def _index_done(self):
        write_json(self.insights_memory, self.persist_file)

    def _build_comparative_prompts(self, true_traj: MASMessage, false_traj: MASMessage, insights: list[dict]) -> list[Message]:
        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        prompt = GMemoryPrompts.critique_compare_rules_user_prompt.format(   
            task1=true_traj.task_description,
            task1_trajectory=true_traj.task_trajectory,   
            task2=false_traj.task_description,
            task2_trajectory=false_traj.task_trajectory,
            fail_reason=false_traj.get_extra_field('fail_reason'),
            existing_rules=rule_text
        )

        return [Message(role='system', content= GMemoryPrompts.critique_compare_rules_system_prompt), Message(role='user', content=prompt)] 
    
    def _build_success_prompts(
        self,
        success_trajectories: Iterable[MASMessage],
        insights: list[dict],
    ) -> list[Message]:

        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        history: list[str] = [f'task{i}:\n' + task.task_description + task.get_extra_field('key_steps') for i, task in enumerate(success_trajectories)]
        prompt = GMemoryPrompts.critique_success_rules_user_prompt.format(
            success_history='\n'.join(history),
            existing_rules=rule_text
        )

        return [Message(role='system', content=GMemoryPrompts.critique_success_rules_system_prompt), Message(role='user', content=prompt)]
    
    def _parse_rules(self, llm_text):
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        matches = re.findall(pattern, llm_text)

        res = []
        banned_words = ['ADD', 'AGREE', 'EDIT']
        for operation, text in matches:
            text = text.strip()
            if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):

                if 'ADD' in operation:
                    res.append(('ADD', text))
                else:
                    res.append((operation.strip(), text))
        return(res)
    
    def _update_rules(
        self,
        relative_tasks: list[str],
        operations: list[tuple[str, str]], 
        max_rules_num: int = 10
    ) -> None:

        delete_indices = []
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

            if operation_type == 'ADD':    
                if self._is_existing_rule(operation_rule_text): 
                    delete_indices.append(i)
                    
            elif operation_type == 'EDIT':   
                if self._is_existing_rule(operation_rule_text): 
                    rule_num: int = self._retrieve_rule_index(operation_rule_text)
                    operations[i] = (f'AGREE {rule_num + 1}', operation_rule_text)   

                elif (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
                        
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':  
                if (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
            
            else: 
                delete_indices.append(i)

        operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] 
        

        list_full: bool = len(self.insights_memory) >= max_rules_num  
        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: 
            for i in range(len(operations)):
                operation, operation_rule_text = operations[i]
                operation_type = operation.split(' ')[0]
                if operation_type != op:
                    continue

                if operation_type == 'REMOVE': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    remove_strength = 3 if list_full else 1
                    rule_data['score'] -= remove_strength
                    rule_data['negative_correlation_tasks'] = list(set(rule_data['negative_correlation_tasks'] + relative_tasks))  

                elif operation_type == 'AGREE':
                    rule_index: int = self._retrieve_rule_index(operation_rule_text) 
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'EDIT': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['rule'] = operation_rule_text
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'ADD': 
                    meta_data: dict = {
                        'rule': operation_rule_text,
                        'score': 2,         
                        'positive_correlation_tasks': list(relative_tasks),
                        'negative_correlation_tasks': list()
                    }
                    self.insights_memory.append(meta_data)

    def _is_existing_rule(self, operation_rule_text: str) -> bool:

        for insight in self.insights_memory:
            if insight['rule'] in operation_rule_text:
                return True
        return False
    
    def _retrieve_rule_index(self, operation_rule_text: str) -> int:

        for idx, insight in enumerate(self.insights_memory):
            if insight['rule'] in operation_rule_text:
                return idx
        return -1