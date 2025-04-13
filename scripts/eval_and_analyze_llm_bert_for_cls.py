import os
import argparse
import pickle
import json
import logging
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import defaultdict, Counter
from tqdm import tqdm
import threading
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from prompts import (
        LABELS,
        EXAMPLE_TEMPLATE,
        RESOLUTION_MARKER,
        SYSTEM_MESSAGE,
        get_classification_prompt
    )

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    text_cols: List[str]
    label_col: str
    label_sep: str
    text_format: str
    labels: List[str]
    file_format: str = "csv"
    
    def get_text_from_row(self, row):
        """Generate text from row using the text format and columns"""
        values = {col: row.get(col, "") for col in self.text_cols}
        # Handle special formatting for different column types
        for col in self.text_cols:
            if col not in row or pd.isna(row[col]):
                values[col] = ""
            elif col.lower() == "keywords" and values[col]:
                values[col] = f"\nKeywords:{values[col]}"
        
        return self.text_format.format(**values)


# dataset specific configs
DATASET_CONFIGS = {
    "litcovid": DatasetConfig(
        name="litcovid",
        text_cols=["title", "abstract", "keywords"],
        label_col="label",
        label_sep=";",
        text_format="Title:{title}\nAbstract:{abstract}{keywords}",
        labels=[
            "Treatment", "Diagnosis", "Prevention", "Mechanism",
            "Transmission", "Epidemic Forecasting", "Case Report"
        ],
    ),
    "hoc": DatasetConfig(
        name="hoc",
        text_cols=["Text"],
        label_col="Labels",
        label_sep=";",
        text_format="{Text}",
        labels=[
            "sustaining proliferative signaling",
            "evading growth suppressors",
            "resisting cell death",
            "enabling replicative immortality",
            "inducing angiogenesis",
            "activating invasion and metastasis",
            "genomic instability and mutation",
            "tumor promoting inflammation",
            "cellular energetics",
            "avoiding immune destruction"
        ],
        file_format="tsv"
    )
}


class DataLoader:
    def __init__(self, file_path=None, dataset_name=None, config=None):
        self.file_path = file_path
        self.data = None
        
        if config:
            self.config = config
        elif dataset_name and dataset_name in DATASET_CONFIGS:
            self.config = DATASET_CONFIGS[dataset_name]
        else:
            # Default to LitCovid if no config is provided
            logger.info("No dataset config specified - defaulting to LitCovid config")
            self.config = DATASET_CONFIGS["litcovid"]
    
    def load_data(self, file_path=None):
        """
        Load data from file.
        
        Args:
            file_path: Path to override the initialized file_path
        """
        if file_path:
            self.file_path = file_path
            
        if not self.file_path:
            raise ValueError("No file path provided")
        
        # Load based on file format
        if self.config.file_format == "tsv" or self.file_path.endswith(".tsv"):
            # For TSV-based files like HoC
            if os.path.basename(self.file_path) in ["train.tsv", "test.tsv"]:
                self.data = pd.read_csv(
                    self.file_path, 
                    header=None, 
                    names=["ID", "Text", self.config.label_col], 
                    sep="\t"
                )
            else:
                self.data = pd.read_csv(self.file_path, sep="\t")
        else:
            # For CSV-based files like LitCovid
            self.data = pd.read_csv(self.file_path)
        
        return self.data
    
    def preprocess(self):
        """
        Preprocess the loaded data by combining text fields and processing labels.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        # Create combined text field if needed
        if "Text" not in self.data.columns:
            self.data["Text"] = self.data.apply(
                lambda row: self.config.get_text_from_row(row), 
                axis=1
            )
        
        # Process labels
        if self.config.label_col in self.data.columns:
            # If labels are already in a list format, do nothing
            if not isinstance(self.data[self.config.label_col].iloc[0], list):
                self.data["Labels"] = self.data[self.config.label_col].str.split(self.config.label_sep)
            else:
                self.data["Labels"] = self.data[self.config.label_col]
        
        return self.data


class LLMClassifier:
    def __init__(
        self, 
        api_key,
        model_name = "gpt-4o",
        embedding_model = "text-embedding-3-small",
        nearest_k = 3,
        output_format = "list",
        use_cot = True,
        dataset_config = None,
        enable_prompt_optimization = False,  # Disabled by default
        prompt_optimization_config = None
    ):
        os.environ['OPENAI_API_KEY'] = api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.nearest_k = nearest_k
        self.output_format = output_format
        self.use_cot = use_cot
        self.config = dataset_config
        
        self.enable_prompt_optimization = enable_prompt_optimization
        self.prompt_optimization_config = prompt_optimization_config or {
            "batch_size": 100,
            "max_iterations": 10,
            "num_candidates": 3,
            "beam_width": 3,
            "verbose": True
        }
        self.prompt_optimizer = None
        
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )
        
        self.langchain_available = True
        
        self._prepare_prompts()
        
        self.examples = []
        self.examples_embedded = []
        
    def _prepare_prompts(self):
        self.prefix = get_classification_prompt(
            use_cot=self.use_cot,
            output_format=self.output_format
        )
        
        if self.langchain_available:
            from langchain_core.prompts import PromptTemplate
            # Create the example prompt template using the template from prompts.py
            self.example_prompt = PromptTemplate(
                input_variables=["title", "abstract", "label"], 
                template=EXAMPLE_TEMPLATE
            )
    
    def optimize_prompt(self, train_data, batch_size=None, max_iterations=None):
        if not self.prompt_optimizer:
            config = self.prompt_optimization_config.copy()
            if batch_size:
                config["batch_size"] = batch_size
            if max_iterations:
                config["max_iterations"] = max_iterations
                
            self.prompt_optimizer = AutomaticPromptOptimizer(
                llm_classifier=self,
                dataset_config=self.config,
                **config
            )
            
        best_prompt, best_score = self.prompt_optimizer.optimize(train_data)
        self.prefix = best_prompt
        return best_prompt, best_score
        
    def load_examples(self, train_df: pd.DataFrame):
        self.examples = []
        self.examples_embedded = []
        
        label_col = self.config.label_col if self.config else "Labels"
        label_col = "Labels" if label_col not in train_df.columns and "Labels" in train_df.columns else label_col
        
        text_col = "Text"
        title_col = "title" if "title" in train_df.columns else None
        abstract_col = "abstract" if "abstract" in train_df.columns else None
        
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading training examples"):
            if self.output_format == "json" and isinstance(row[label_col], list):
                labels = row[label_col]
                labels_json = {topic: 0 for topic in (self.config.labels if self.config else LABELS)}
                for label in labels:
                    if label in labels_json:
                        labels_json[label] = 1
                
                example = {
                    "label": json.dumps(labels_json)
                }
            else:
                if isinstance(row[label_col], list):
                    label_str = ";".join(row[label_col])
                else:
                    label_str = row[label_col]
                
                example = {
                    "label": label_str
                }
            
            # title & abstract
            if title_col:
                example["title"] = row[title_col]
            else:
                example["title"] = ""
                
            if abstract_col:
                example["abstract"] = row[abstract_col]
            else:
                example["abstract"] = row[text_col] if text_col in row else ""
            
            self.examples.append(example)
            
            # embedding
            text_to_embed = f"title: {example['title']} abstract: {example['abstract']}"
            if self.langchain_available:
                embedding = self.embeddings.embed_query(text_to_embed)
            else:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text_to_embed,
                    encoding_format="float"
                )
                embedding = response.data[0].embedding
                
            self.examples_embedded.append(embedding)
    
    def save_embeddings(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self.examples_embedded, f)
    
    def load_embeddings(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.examples_embedded = pickle.load(f)
    
    def find_similar_examples(self, title, abstract):
        query_text = f"title: {title} abstract: {abstract}"
        
        if self.langchain_available:
            query_embedding = self.embeddings.embed_query(query_text)
        else:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=query_text,
                encoding_format="float"
            )
            query_embedding = response.data[0].embedding
        
        # cos similarity
        similarities = []
        for i, example_embedding in enumerate(self.examples_embedded):
            cosine_similarity = np.dot(query_embedding, example_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(example_embedding)
            )
            similarities.append((i, cosine_similarity))
        
        # get indices
        top_k_similar_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:self.nearest_k]
        return [item[0] for item in top_k_similar_indices]
    
    def predict(self, title, abstract):
        nearest_example_indices = self.find_similar_examples(title, abstract)
        
        # format
        if self.langchain_available:
            from langchain_core.runnables import RunnablePassthrough
            
            examples_formatted = "\n".join(
                self.example_prompt.format(**self.examples[idx]) 
                for idx in nearest_example_indices
            )
            
            # create full prompt
            full_prompt = (
                self.prefix + 
                examples_formatted + 
                RESOLUTION_MARKER +
                f"title:{title}\nabstract:{abstract}\nlabel:"
            )
            
            chain = (
                {"prompt": full_prompt}
                | RunnablePassthrough.assign(
                    response=lambda x: self.llm.invoke(x["prompt"])
                )
                | (lambda x: x["response"].content)
            )
            
            response = chain.invoke({"prompt": full_prompt})
        else:
            examples_formatted = "\n".join(
                EXAMPLE_TEMPLATE.format(**self.examples[idx]) 
                for idx in nearest_example_indices
            )
            
            full_prompt = (
                self.prefix + 
                examples_formatted + 
                RESOLUTION_MARKER +
                f"title:{title}\nabstract:{abstract}\nlabel:"
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0
            )
            response = response.choices[0].message.content
        
        if self.output_format == "json":
            try:
                result = self._parse_json_response(response)
            except:
                logger.error(f"Error parsing JSON response: {response}")
                result = ""
        else:
            result = self._parse_list_response(response)
            
        return result
    
    def _parse_json_response(self, response: str) -> str:
        try:
            start_index = response.find('{')
            end_index = response.rfind('}') + 1 
            
            json_str = response[start_index:end_index].replace("'", '"')
            parsed_dict = json.loads(json_str)
            
            # Convert to semicolon-separated list
            results_list = [key for key, value in parsed_dict.items() if value == 1]
            return ';'.join(results_list)
        except:
            logger.error(f"Error parsing the output: {response}")
            return ""
    
    def _parse_list_response(self, response: str) -> str:
        if "[" in response:
            start_index = response.find('[')
            end_index = response.rfind(']') + 1      
            parsed_list = response[start_index:end_index]
            # Clean up the list
            parsed_list = parsed_list.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
            # Replace commas with semicolons
            parsed_list = parsed_list.replace(", ", ";").replace(",", ";")
        elif "'" in response:
            start_index = response.find("'")
            end_index = response.rfind("'") + 1  
            parsed_list = response[start_index:end_index].replace(",", ";")
        elif '"' in response:
            start_index = response.find('"')
            end_index = response.rfind('"') + 1  
            parsed_list = response[start_index:end_index].replace(",", ";")
        elif "-" in response:
            lines = response.split("\n")
            parsed_list = []
            for line in lines:
                if "-" in line:
                    topic = line.split("-")[-1].strip()
                    parsed_list.append(topic)
            parsed_list = ";".join(parsed_list)
        else:
            parsed_list = response
            
        return parsed_list
    
    def predict_batch(self, test_df: pd.DataFrame, timeout: int = 30) -> pd.DataFrame:
        result_df = test_df.copy()
        result_df['labels_pred'] = None
        result_df['raw_output'] = None
        
        title_col = "title" if "title" in result_df.columns else None
        abstract_col = "abstract" if "abstract" in result_df.columns else None
        text_col = "Text" if "Text" in result_df.columns else None
        
        def predict_with_timeout(row, result_container):
            try:
                if title_col and abstract_col:
                    title = row[title_col]
                    abstract = row[abstract_col]
                elif text_col:
                    text = row[text_col]
                    parts = text.split(". ", 1)
                    if len(parts) > 1:
                        title = parts[0]
                        abstract = parts[1]
                    else:
                        title = text
                        abstract = ""
                else:
                    for col in row.index:
                        if isinstance(row[col], str) and len(row[col]) > 10:
                            title = row[col][:100]
                            abstract = row[col][100:] if len(row[col]) > 100 else ""
                            break
                    else:
                        raise ValueError("Could not find suitable text columns in the data")
                
                results = self.predict(title, abstract)
                result_container['raw_output'] = results
                result_container['parsed_results'] = results
            except Exception as e:
                result_container['error'] = str(e)
        
        for index, row in tqdm(result_df.iterrows(), total=result_df.shape[0], desc="Predicting"):
            result_container = {}
            predict_thread = threading.Thread(
                target=predict_with_timeout, 
                args=(row, result_container)
            )
            predict_thread.start()
            predict_thread.join(timeout=timeout)
            
            if 'parsed_results' in result_container:
                result_df.at[index, 'labels_pred'] = result_container['parsed_results']
                result_df.at[index, 'raw_output'] = result_container['raw_output']
            elif 'error' in result_container:
                logger.error(f"ERROR at index {index}: {result_container['error']}")
            else:
                logger.warning(f"Timeout occurred for index: {index}")
                
        return result_df


class AutomaticPromptOptimizer:
    """
    original paper: https://arxiv.org/abs/2305.03495
    Adapted for cls task
    """
    def __init__(
        self,
        llm_classifier,
        dataset_config=None,
        batch_size: int = 100,
        max_iterations: int = 10,
        num_candidates: int = 3,
        beam_width: int = 3,
        verbose: bool = True
    ):
        self.llm = llm_classifier
        self.config = dataset_config
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        self.beam_width = beam_width
        self.verbose = verbose
        
        self.prompt_history = []
        self.score_history = []
        self.best_prompt = None
        self.best_score = 0.0
    
    def calculate_distribution_gap(self, true_labels, pred_labels):
        true_counts = Counter()
        pred_counts = Counter()
        
        for labels in true_labels:
            for label in labels:
                true_counts[label] += 1
                
        for labels in pred_labels:
            for label in labels:
                pred_counts[label] += 1
        
        all_labels = set(true_counts.keys()) | set(pred_counts.keys())
        total_true = sum(true_counts.values()) or 1
        total_pred = sum(pred_counts.values()) or 1
        
        true_dist = {label: true_counts.get(label, 0) / total_true for label in all_labels}
        pred_dist = {label: pred_counts.get(label, 0) / total_pred for label in all_labels}
        
        # KL divergence
        kl_div = 0
        for label in all_labels:
            p = true_dist.get(label, 0.0001)  # small epsilon to avoid log(0)
            q = pred_dist.get(label, 0.0001)
            kl_div += p * np.log(p / q)
        
        # find most over/under-predicted labels
        diff_dist = {label: pred_dist[label] - true_dist[label] for label in all_labels}
        over_predicted = sorted(diff_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        under_predicted = sorted(diff_dist.items(), key=lambda x: x[1])[:3]
        
        return {
            "kl_divergence": kl_div,
            "over_predicted": over_predicted,
            "under_predicted": under_predicted,
            "true_dist": true_dist,
            "pred_dist": pred_dist
        }
    
    def analyze_errors(self, true_labels, pred_labels, texts):
        false_positives = []
        false_negatives = []
        
        for i, (true, pred, text) in enumerate(zip(true_labels, pred_labels, texts)):
            true_set = set(true)
            pred_set = set(pred)
            
            # FP
            for fp_label in pred_set - true_set:
                false_positives.append({
                    "label": fp_label,
                    "text": text,
                    "all_true": list(true_set),
                    "all_pred": list(pred_set)
                })
            
            # FN
            for fn_label in true_set - pred_set:
                false_negatives.append({
                    "label": fn_label,
                    "text": text,
                    "all_true": list(true_set),
                    "all_pred": list(pred_set)
                })
        
        # examples for prompt refinement
        sampled_fp = random.sample(false_positives, min(3, len(false_positives))) if false_positives else []
        sampled_fn = random.sample(false_negatives, min(3, len(false_negatives))) if false_negatives else []
        
        total_fp = len(false_positives)
        total_fn = len(false_negatives)
        
        return {
            "false_positives": sampled_fp,
            "false_negatives": sampled_fn,
            "total_fp": total_fp,
            "total_fn": total_fn
        }
    
    def generate_gradient(self, dist_gap, error_analysis, current_prompt):
        gradient = "The current prompt has the following issues:\n\n"
        
        gradient += "1. Label Distribution Problems:\n"
        if dist_gap["over_predicted"]:
            gradient += "- Over-predicted labels: "
            gradient += ", ".join([f"{label} ({100*diff:.1f}%)" for label, diff in dist_gap["over_predicted"]])
            gradient += "\n"
        
        if dist_gap["under_predicted"]:
            gradient += "- Under-predicted labels: "
            gradient += ", ".join([f"{label} ({-100*diff:.1f}%)" for label, diff in dist_gap["under_predicted"]])
            gradient += "\n"
        
        gradient += "\n2. Specific Classification Errors:\n"
        
        if error_analysis["false_positives"]:
            gradient += "- False Positives (incorrectly predicted labels):\n"
            for fp in error_analysis["false_positives"]:
                gradient += f"  * Incorrectly predicted '{fp['label']}' for text that's actually about {', '.join(fp['all_true'])}\n"
        
        if error_analysis["false_negatives"]:
            gradient += "- False Negatives (missed labels):\n"
            for fn in error_analysis["false_negatives"]:
                gradient += f"  * Failed to predict '{fn['label']}' for relevant text, only predicted {', '.join(fn['all_pred'])}\n"
        
        # statistical summary
        gradient += f"\n3. Overall Error Summary:\n"
        gradient += f"- Total false positives: {error_analysis['total_fp']}\n"
        gradient += f"- Total false negatives: {error_analysis['total_fn']}\n"
        gradient += f"- KL divergence from true distribution: {dist_gap['kl_divergence']:.4f}\n"
        
        return gradient
    
    def generate_improved_prompts(self, current_prompt, gradient):
        system_message = """You are an expert at optimizing prompts for NLP tasks. 
            Your job is to refine a prompt for a multi-label classification task based on feedback.
            Generate a completely new prompt that addresses the provided feedback.
            Keep the structure of task instructions but modify and improve the guidance given to the model.
            Focus on fixing the specific issues mentioned in the feedback.
            """
                    
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=self.llm.model_name,
            temperature=0.7 
        )
        
        candidates = []
        for i in range(self.num_candidates):
            try:
                response = llm.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Here is the current prompt:\n\n```\n{current_prompt}\n```\n\nHere is the feedback (the 'gradient') based on evaluation:\n\n```\n{gradient}\n```\n\nPlease generate a new improved prompt that addresses these issues. Focus specifically on fixing the label distribution issues and reducing both false positives and false negatives. Return only the new prompt without any explanations."}
                    ]
                )
                candidate_prompt = response.content.strip()
                
                if candidate_prompt and candidate_prompt != current_prompt:
                    candidates.append(candidate_prompt)
                else:
                    response = llm.invoke(
                        [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": f"Here is the current prompt:\n\n```\n{current_prompt}\n```\n\nHere is the feedback (the 'gradient') based on evaluation:\n\n```\n{gradient}\n```\n\nPlease generate a new improved prompt that addresses these issues. Be creative and make substantial changes to fix the identified problems. Return only the new prompt without any explanations."}
                        ],
                        temperature=0.9
                    )
                    candidate_prompt = response.content.strip()
                    if candidate_prompt and candidate_prompt != current_prompt:
                        candidates.append(candidate_prompt)
            except Exception as e:
                logger.error(f"Error generating candidate prompt: {e}")
        
        # make some manual simple changes in case of no valid candidates, 
        if not candidates:
            candidates = [
                current_prompt + "\n\nPlease be extra careful about correctly identifying all relevant categories and avoiding incorrect classifications.",
                current_prompt.replace("Classify", "Carefully classify").replace("categorize", "accurately categorize"),
                "REVISED: " + current_prompt + "\n\nNote: Pay special attention to the distribution of labels in your classifications."
            ]
        
        return candidates
    
    def evaluate_prompt(self, prompt, eval_data):
        original_prompt = self.llm.prefix
        
        self.llm.prefix = prompt
        
        try:
            results = self.llm.predict_batch(eval_data)
            
            analyzer = ResultsAnalyzer(results)
            label_metrics = analyzer.calculate_label_based_metrics()
            dist_metrics = analyzer.calculate_kl_divergence()
            
            true_labels = results['labels_true'].tolist()
            pred_labels = results['labels_pred'].tolist()
            texts = []
            
            title_col = "title" if "title" in results.columns else None
            abstract_col = "abstract" if "abstract" in results.columns else None
            text_col = "Text" if "Text" in results.columns else None
            
            for _, row in results.iterrows():
                if title_col and abstract_col:
                    text = f"Title: {row[title_col]}\nAbstract: {row[abstract_col]}"
                elif text_col:
                    text = row[text_col]
                else:
                    text = ""
                texts.append(text)
            
            dist_gap = self.calculate_distribution_gap(true_labels, pred_labels)
            error_analysis = self.analyze_errors(true_labels, pred_labels, texts)
            
            metrics = {
                "macro_f1": label_metrics["macro"]["f1"],
                "micro_f1": label_metrics["micro"]["f1"],
                "kl_divergence": dist_metrics["test_kl_divergence"],
                "distribution_gap": dist_gap,
                "error_analysis": error_analysis
            }
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            metrics = {
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "kl_divergence": float('inf'),
                "distribution_gap": None,
                "error_analysis": None
            }
        
        self.llm.prefix = original_prompt
        
        return metrics
    
    def optimize(self, train_data):
        if len(train_data) > self.batch_size:
            eval_batch = train_data.sample(n=self.batch_size, random_state=42)
        else:
            eval_batch = train_data.copy()
        
        current_prompts = [self.llm.prefix]
        
        for iteration in range(self.max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")
            
            prompt_scores = []
            
            for i, prompt in enumerate(current_prompts):
                logger.info(f"Evaluating prompt {i + 1}/{len(current_prompts)}")
                metrics = self.evaluate_prompt(prompt, eval_batch)
                
                prompt_scores.append({
                    "prompt": prompt,
                    "score": metrics["macro_f1"],
                    "metrics": metrics
                })
                
                if self.verbose:
                    logger.info(f"Prompt {i + 1} score: {metrics['macro_f1']:.4f}")
            
            prompt_scores.sort(key=lambda x: x["score"], reverse=True)
            
            if prompt_scores[0]["score"] > self.best_score:
                self.best_prompt = prompt_scores[0]["prompt"]
                self.best_score = prompt_scores[0]["score"]
                logger.info(f"New best prompt found with score: {self.best_score:.4f}")
            
            self.prompt_history.append([ps["prompt"] for ps in prompt_scores])
            self.score_history.append([ps["score"] for ps in prompt_scores])
            
            if iteration == self.max_iterations - 1:
                break
            
            best_current_prompt = prompt_scores[0]["prompt"]
            best_metrics = prompt_scores[0]["metrics"]
            
            # gradient critique
            if best_metrics["distribution_gap"] and best_metrics["error_analysis"]:
                gradient = self.generate_gradient(
                    best_metrics["distribution_gap"],
                    best_metrics["error_analysis"],
                    best_current_prompt
                )
                
                # generate candidate prompts
                candidate_prompts = self.generate_improved_prompts(best_current_prompt, gradient)
                
                # beam search
                current_prompts = [ps["prompt"] for ps in prompt_scores[:self.beam_width]]
                for candidate in candidate_prompts:
                    if candidate not in current_prompts:
                        current_prompts.append(candidate)
                
                # prevent too many evaluations
                current_prompts = current_prompts[:self.beam_width + self.num_candidates]
            else:
                current_prompts = [ps["prompt"] for ps in prompt_scores[:self.beam_width]]
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        return self.best_prompt, self.best_score
    
    def visualize_optimization(self, save_path=None):
        plt.figure(figsize=(10, 6))
        
        for i in range(len(self.score_history[0])):
            scores = [iteration[i] if i < len(iteration) else None for iteration in self.score_history]
            valid_scores = [(j, score) for j, score in enumerate(scores) if score is not None]
            if valid_scores:
                iterations, values = zip(*valid_scores)
                plt.plot(iterations, values, 'o-', alpha=0.5, label=f"Candidate {i+1}" if i < 5 else None)
        
        best_scores = [max(iteration) for iteration in self.score_history]
        plt.plot(range(len(best_scores)), best_scores, 'r-', linewidth=2, label="Best score")
        
        plt.xlabel("Iteration")
        plt.ylabel("Macro F1 Score")
        plt.title("Prompt Optimization Progress")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class BERTClassifier:
    def __init__(
        self,
        model_name = "michiyasunaga/BioLinkBERT-base",
        max_length = 512,
        num_train_epochs = 3,
        dataset_config = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.config = dataset_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from transformers import (
            BertTokenizer, 
            BertForSequenceClassification, 
            Trainer, 
            TrainingArguments
        )
        self.transformers_available = True
    
    def prepare_data(self, train_df, test_df, valid_df=None):
        if not self.transformers_available:
            raise ImportError("Transformers package is required for BERT classification")
        
        from transformers import BertTokenizer
        from sklearn.preprocessing import MultiLabelBinarizer
        from torch.utils.data import Dataset
        
        if valid_df is None:
            valid_df = train_df.sample(frac=0.05, random_state=42)
            valid_df.reset_index(drop=True, inplace=True)
            
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        label_col = self.config.label_col if self.config else "Labels"
        label_col = "Labels" if label_col not in train_df.columns and "Labels" in train_df.columns else label_col
        
        if self.config and self.config.labels:
            label_list = self.config.labels
        else:
            all_labels = set()
            for df in [train_df, valid_df, test_df]:
                if label_col in df.columns:
                    labels = df[label_col].tolist()
                    for label_set in labels:
                        if isinstance(label_set, list):
                            all_labels.update(label_set)
                        elif isinstance(label_set, str):
                            all_labels.update(label_set.split(";"))
            label_list = sorted(list(all_labels))
                
        mlb = MultiLabelBinarizer(classes=label_list)
        
        train_labels = mlb.fit_transform(train_df[label_col])
        valid_labels = mlb.transform(valid_df[label_col])
        test_labels = mlb.transform(test_df[label_col])
        
        class CustomDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len
                
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                
                inputs = {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
                
                return inputs
        
        train_dataset = CustomDataset(
            texts=train_df['Text'].tolist(),
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_len=self.max_length
        )
        
        valid_dataset = CustomDataset(
            texts=valid_df['Text'].tolist(),
            labels=valid_labels,
            tokenizer=self.tokenizer,
            max_len=self.max_length
        )
        
        test_dataset = CustomDataset(
            texts=test_df['Text'].tolist(),
            labels=test_labels,
            tokenizer=self.tokenizer,
            max_len=self.max_length
        )
        
        return train_dataset, valid_dataset, test_dataset, label_list
    
    def train(self, train_dataset, valid_dataset, label_list, output_dir="./results"):
        if not self.transformers_available:
            raise ImportError("Transformers package is required for BERT classification")
        
        from transformers import (
            BertForSequenceClassification, 
            Trainer, 
            TrainingArguments
        )
        
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(label_list),
            problem_type='multi_label_classification'
        ).to(self.device)
        
        def compute_metrics(p):
            preds = p.predictions
            labels = p.label_ids
            
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(preds))
            
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= 0.5)] = 1
            
            acc = accuracy_score(labels, y_pred)
            f1_micro = f1_score(labels, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(labels, y_pred, average='macro', zero_division=0)
            
            try:
                if len(np.unique(labels)) > 1:
                    roc_auc = roc_auc_score(labels, probs.detach().numpy(), average='macro')
                else:
                    roc_auc = float('nan')  # Return NaN if there is only one class
            except ValueError:
                roc_auc = float('nan')
            
            return {
                'accuracy': acc,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc
            }
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=1e-4,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            disable_tqdm=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics
        )
        
        logger.info("Training BERT model...")
        self.trainer.train()
        
        self.model.save_pretrained(os.path.join(output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        
        return self.model, self.trainer
    
    def predict(self, test_dataset):
        if not self.trainer or not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Making predictions...")
        predictions, labels, _ = self.trainer.predict(test_dataset)
        
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions).to('cpu'))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= 0.5)] = 1
        
        y_true = labels
        
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        try:
            roc_auc = roc_auc_score(y_true, probs.detach().numpy(), average='macro')
        except ValueError:
            roc_auc = float('nan')
        
        metrics = {
            'accuracy': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'roc_auc': roc_auc
        }
        
        return probs, y_pred, y_true, metrics


class ResultsAnalyzer:
    def __init__(self, results_df: pd.DataFrame, label_list: Optional[List[str]] = None):
        self.results_df = results_df
        self.label_list = label_list if label_list else LABELS
        
        self.results_df.fillna('', inplace=True)
        
        if 'labels_pred' in self.results_df.columns:
            if len(self.results_df) > 0 and isinstance(self.results_df['labels_pred'].iloc[0], str):
                self.results_df['labels_pred'] = self.results_df['labels_pred'].apply(
                    lambda x: x.split(';') if x else []
                )
            
        label_col = None
        for col in ['label', 'labels', 'Label', 'Labels']:
            if col in self.results_df.columns:
                label_col = col
                break
        
        if label_col:
            if len(self.results_df) > 0 and isinstance(self.results_df[label_col].iloc[0], str):
                self.results_df['labels_true'] = self.results_df[label_col].apply(
                    lambda x: x.split(';') if x else []
                )
            elif 'labels_true' not in self.results_df.columns:
                self.results_df['labels_true'] = self.results_df[label_col]
    
    def calculate_example_based_metrics(self) -> Dict[str, float]:
        def safe_divide(a, b):
            return a / b if b else 0
        self.results_df['EBP'] = self.results_df.apply(
            lambda x: safe_divide(
                len(set(x['labels_pred']).intersection(set(x['labels_true']))), 
                len(set(x['labels_pred']))
            ), 
            axis=1
        )
        
        self.results_df['EBR'] = self.results_df.apply(
            lambda x: safe_divide(
                len(set(x['labels_pred']).intersection(set(x['labels_true']))), 
                len(set(x['labels_true']))
            ), 
            axis=1
        )
        
        self.results_df['EBF1'] = self.results_df.apply(
            lambda x: 2 * safe_divide(x['EBP'] * x['EBR'], (x['EBP'] + x['EBR'])), 
            axis=1
        )
        
        ebr = np.nanmean(self.results_df['EBR'])
        ebp = np.nanmean(self.results_df['EBP'])
        ebf1 = np.nanmean(self.results_df['EBF1'])
        
        return {
            "example_based_recall": ebr,
            "example_based_precision": ebp,
            "example_based_f1": ebf1
        }
    
    def calculate_label_based_metrics(self) -> Dict[str, Dict[str, float]]:
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        
        for _, row in self.results_df.iterrows():
            preds = set(row['labels_pred'])
            actuals = set(row['labels_true'])
            
            for label in actuals:
                if label in preds:
                    true_positives[label] += 1
                else:
                    false_negatives[label] += 1
            
            for label in preds:
                if label not in actuals:
                    false_positives[label] += 1
        
        label_metrics = {}
        for label in set(true_positives.keys()).union(false_positives.keys()).union(false_negatives.keys()):
            tp = true_positives[label]
            fp = false_positives[label]
            fn = false_negatives[label]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            label_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            }
        
        # macro
        macro_precision = sum(m["precision"] for m in label_metrics.values()) / len(label_metrics) if label_metrics else 0
        macro_recall = sum(m["recall"] for m in label_metrics.values()) / len(label_metrics) if label_metrics else 0
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
        
        # micro
        total_tp = sum(true_positives.values())
        total_fp = sum(false_positives.values())
        total_fn = sum(false_negatives.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        return {
            "per_label": label_metrics,
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1
            },
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1
            }
        }
    
    def visualize_label_distributions(self, train_labels=None, save_path=None):
        if isinstance(self.results_df['labels_pred'].iloc[0], list):
            pred_labels = np.array([
                [1 if label in row['labels_pred'] else 0 for label in self.label_list]
                for _, row in self.results_df.iterrows()
            ])
            
            true_labels = np.array([
                [1 if label in row['labels_true'] else 0 for label in self.label_list]
                for _, row in self.results_df.iterrows()
            ])
        else:
            raise ValueError("Labels should be in list format.")
        
        # occurrences of each label
        label_freq_pred = pred_labels.sum(axis=0)
        label_freq_true = true_labels.sum(axis=0)
        
        # relative frequencies
        relative_freq_pred = label_freq_pred / label_freq_pred.sum() if label_freq_pred.sum() > 0 else np.zeros_like(label_freq_pred)
        relative_freq_true = label_freq_true / label_freq_true.sum() if label_freq_true.sum() > 0 else np.zeros_like(label_freq_true)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.label_list))
        width = 0.35
        plt.bar(x - width/2, label_freq_true, width, alpha=0.7, label="True (Test Set)")
        plt.bar(x + width/2, label_freq_pred, width, alpha=0.7, label="Predicted")
        
        # add train set if provided
        if train_labels is not None:
            label_freq_train = train_labels.sum(axis=0)
            plt.bar(x, label_freq_train, width * 0.5, alpha=0.5, label="Train Set")
        
        plt.xticks(x, self.label_list, rotation=45, ha="right")
        plt.title("Label Frequency Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def visualize_labels_per_instance(self, save_path=None):
        if isinstance(self.results_df['labels_pred'].iloc[0], list):
            labels_per_instance_pred = [len(labels) for labels in self.results_df['labels_pred']]
            labels_per_instance_true = [len(labels) for labels in self.results_df['labels_true']]
        else:
            raise ValueError("Labels should be in list format.")
        
        plt.figure(figsize=(10, 6))
        max_labels = max(max(labels_per_instance_pred), max(labels_per_instance_true))
        bins = range(max_labels + 2)
        
        plt.hist(labels_per_instance_true, bins=bins, alpha=0.7, label="True")
        plt.hist(labels_per_instance_pred, bins=bins, alpha=0.7, label="Predicted")
        
        plt.title("Number of Labels Per Instance")
        plt.xlabel("Number of Labels")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def calculate_kl_divergence(self, train_labels=None):
        if isinstance(self.results_df['labels_pred'].iloc[0], list):
            pred_labels = np.array([
                [1 if label in row['labels_pred'] else 0 for label in self.label_list]
                for _, row in self.results_df.iterrows()
            ])
            
            true_labels = np.array([
                [1 if label in row['labels_true'] else 0 for label in self.label_list]
                for _, row in self.results_df.iterrows()
            ])
        else:
            raise ValueError("Labels should be in list format.")
        
        # label occurrences
        label_freq_pred = pred_labels.sum(axis=0)
        label_freq_true = true_labels.sum(axis=0)
        
        relative_freq_pred = label_freq_pred / label_freq_pred.sum() if label_freq_pred.sum() > 0 else np.zeros_like(label_freq_pred)
        relative_freq_true = label_freq_true / label_freq_true.sum() if label_freq_true.sum() > 0 else np.zeros_like(label_freq_true)
        
        # zero division handling
        epsilon = 1e-10
        relative_freq_pred = relative_freq_pred + epsilon
        relative_freq_true = relative_freq_true + epsilon
        
        relative_freq_pred = relative_freq_pred / relative_freq_pred.sum()
        relative_freq_true = relative_freq_true / relative_freq_true.sum()
        
        kl_div_test = entropy(relative_freq_pred, relative_freq_true)
        
        result = {
            "test_kl_divergence": kl_div_test
        }
        
        if train_labels is not None:
            label_freq_train = train_labels.sum(axis=0)
            relative_freq_train = label_freq_train / label_freq_train.sum() if label_freq_train.sum() > 0 else np.zeros_like(label_freq_train)
            
            relative_freq_train = relative_freq_train + epsilon
            relative_freq_train = relative_freq_train / relative_freq_train.sum()
            
            kl_div_train = entropy(relative_freq_pred, relative_freq_train)
            result["train_kl_divergence"] = kl_div_train
        
        return result
    
    def visualize_kl_divergence(self, train_labels=None, save_path=None):
        kl_metrics = self.calculate_kl_divergence(train_labels)
        
        plt.figure(figsize=(8, 5))
        
        if "train_kl_divergence" in kl_metrics:
            plt.bar(
                ["Train KL", "Test KL"], 
                [kl_metrics["train_kl_divergence"], kl_metrics["test_kl_divergence"]], 
                color=["blue", "orange"]
            )
        else:
            plt.bar(["Test KL"], [kl_metrics["test_kl_divergence"]], color=["orange"])
        
        plt.title("KL Divergence Between Predicted and True Distribution")
        plt.ylabel("KL Divergence")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Multi-label Classification Script")
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--llm", action="store_true")
    model_group.add_argument("--bert", action="store_true")
    
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--visualization_dir", type=str)
    
    llm_group = parser.add_argument_group("LLM options")
    llm_group.add_argument("--api_key", type=str)
    llm_group.add_argument("--model", type=str, default="gpt-4o")
    llm_group.add_argument("--embeddings_file", type=str)
    llm_group.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    llm_group.add_argument("--nearest_k", type=int, default=3)
    llm_group.add_argument("--output_format", type=str, choices=["list", "json"], default="json")
    llm_group.add_argument("--no_cot", action="store_true")
    
    bert_group = parser.add_argument_group("BERT options")
    bert_group.add_argument("--bert_model", type=str, default="michiyasunaga/BioLinkBERT-base")
    bert_group.add_argument("--epochs", type=int, default=3)
    bert_group.add_argument("--max_length", type=int, default=512)
    bert_group.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    dataset_config = None
    if args.dataset:
        dataset_config = DATASET_CONFIGS[args.dataset]
    else:
        if args.train_file and args.train_file.endswith(".tsv"):
            dataset_config = DATASET_CONFIGS["hoc"]
        elif args.train_file:
            dataset_config = DATASET_CONFIGS["litcovid"]
    
    if not args.analysis_only:
        train_loader = DataLoader(args.train_file, config=dataset_config)
        train_data = train_loader.load_data()
        train_data = train_loader.preprocess()
        
        test_loader = DataLoader(args.test_file, config=dataset_config)
        test_data = test_loader.load_data()
        test_data = test_loader.preprocess()
        
        if args.llm:
            classifier = LLMClassifier(
                api_key=args.api_key,
                model_name=args.model,
                embedding_model=args.embedding_model,
                nearest_k=args.nearest_k,
                output_format=args.output_format,
                use_cot=not args.no_cot,
                dataset_config=dataset_config
            )
            
            if args.embeddings_file and os.path.exists(args.embeddings_file):
                classifier.load_embeddings(args.embeddings_file)
                
                title_col = "title" if "title" in train_data.columns else None
                abstract_col = "abstract" if "abstract" in train_data.columns else None
                label_col = dataset_config.label_col if dataset_config else "Labels"
                label_col = "Labels" if label_col not in train_data.columns and "Labels" in train_data.columns else label_col
                
                classifier.examples = []
                for _, row in train_data.iterrows():
                    example = {
                        "title": row[title_col] if title_col else "",
                        "abstract": row[abstract_col] if abstract_col else row["Text"],
                        "label": ";".join(row[label_col]) if isinstance(row[label_col], list) else row[label_col]
                    }
                    classifier.examples.append(example)
            else:
                classifier.load_examples(train_data)
                if args.embeddings_file:
                    classifier.save_embeddings(args.embeddings_file)
            
            results_df = classifier.predict_batch(test_data)
            
        else:  # BERT
            classifier = BERTClassifier(
                model_name=args.bert_model,
                max_length=args.max_length,
                num_train_epochs=args.epochs,
                dataset_config=dataset_config
            )
            
            print("Preparing data (BERT)...")
            train_dataset, valid_dataset, test_dataset, label_list = classifier.prepare_data(
                train_data, test_data
            )
            
            print("Training...")
            model, trainer = classifier.train(
                train_dataset, valid_dataset, label_list, 
                output_dir=args.output_dir
            )
            
            print("Evaluating...")
            probs, y_pred, y_true, metrics = classifier.predict(test_dataset)
            
            test_data["labels_pred"] = [
                [label_list[i] for i, val in enumerate(pred) if val == 1]
                for pred in y_pred
            ]
            test_data["labels_true"] = [
                [label_list[i] for i, val in enumerate(true) if val == 1]
                for true in y_true
            ]
            
            results_df = test_data.copy()
            
            print("\nBERT Model Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Micro-F1: {metrics['f1_micro']:.4f}")
            print(f"  Macro-F1: {metrics['f1_macro']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        if args.output_file:
            results_df.to_csv(args.output_file, index=False)
    
    else:
        print(f"Loading results: {args.output_file}...")
        results_df = pd.read_csv(args.output_file)
    
    label_list = dataset_config.labels if dataset_config else LABELS
    analyzer = ResultsAnalyzer(results_df, label_list=label_list)
    
    example_metrics = analyzer.calculate_example_based_metrics()
    label_metrics = analyzer.calculate_label_based_metrics()
    
    print("\nExample-based Metrics:")
    print(f"  Precision: {example_metrics['example_based_precision']:.4f}")
    print(f"  Recall: {example_metrics['example_based_recall']:.4f}")
    print(f"  F1: {example_metrics['example_based_f1']:.4f}")
    
    print("\nMacro-averaged Metrics:")
    print(f"  Precision: {label_metrics['macro']['precision']:.4f}")
    print(f"  Recall: {label_metrics['macro']['recall']:.4f}")
    print(f"  F1: {label_metrics['macro']['f1']:.4f}")
    
    print("\nMicro-averaged Metrics:")
    print(f"  Precision: {label_metrics['micro']['precision']:.4f}")
    print(f"  Recall: {label_metrics['micro']['recall']:.4f}")
    print(f"  F1: {label_metrics['micro']['f1']:.4f}")
    
    print("\nPer-label Performance:")
    for label, metrics in label_metrics['per_label'].items():
        print(f"  {label}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
    
    # visualize
    if args.visualization_dir:
        os.makedirs(args.visualization_dir, exist_ok=True)
        print(f"\n Visualize: {args.visualization_dir}...")
        
        analyzer.visualize_label_distributions(
            save_path=os.path.join(args.visualization_dir, "label_distribution.png")
        )
        
        analyzer.visualize_labels_per_instance(
            save_path=os.path.join(args.visualization_dir, "labels_per_instance.png")
        )
        
        analyzer.visualize_kl_divergence(
            save_path=os.path.join(args.visualization_dir, "kl_divergence.png")
        )
    else:
        analyzer.visualize_label_distributions()
        analyzer.visualize_labels_per_instance()
        analyzer.visualize_kl_divergence()
    

if __name__ == "__main__":
    main()