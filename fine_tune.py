import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from dataclasses import dataclass
import random
import logging
import subprocess
import tempfile
import os
import resource
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConstants:
    MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    FAST_INFERENCE = False
    LORA_RANK = 32
    LORA_ALPHA = 32
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    LORA_DROPOUT = 0
    BIAS = "none"
    USE_GRADIENT_CHECKPOINTING = "unsloth"
    RANDOM_STATE = 3407
    USE_RSLORA = False

@dataclass
class TrainingConstants:
    NUM_CANDIDATES = 8
    MAX_LENGTH = 2048
    GENERATION_LENGTH = 512
    MAX_SOLUTION_LENGTH = 1024
    TEMPERATURE = 0.8
    TOP_P = 0.9
    TOP_K = 50
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    MAX_EPOCHS = 3
    LAMBDA_ERROR = 0.5
    LAMBDA_TLE = 0.3
    LAMBDA_MLE = 0.2
    TIME_LIMIT = 2
    MEM_LIMIT_MB = 100

class RewardFunction:
    def __init__(
        self, 
        training_constant: TrainingConstants = TrainingConstants
        ):
        self.lambda_error = training_constant.LAMBDA_ERROR
        self.lambda_tle = training_constant.LAMBDA_TLE
        self.lambda_mle = training_constant.LAMBDA_MLE
        self.time_limit = training_constant.TIME_LIMIT
        self.mem_limit_mb = training_constant.MEM_LIMIT_MB
    
    def limit_memory(self, mem_limit_bytes):
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    
    def compile_cpp_code(self, cpp_code, filename_prefix="temp_cpp"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False, prefix=filename_prefix) as src_file:
            src_file.write(cpp_code)
            cpp_filename = src_file.name
        
        exe_filename = cpp_filename.replace(".cpp", ".out")
        compile_command = ["g++", cpp_filename, "-o", exe_filename]
        
        compile_result = subprocess.run(compile_command, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            compile_error = compile_result.stderr
            os.remove(cpp_filename)
            return None, compile_error
        else:
            return exe_filename, None
    
    def run_cpp_executable(self, exe_filename, test_input):
        error_flag = False
        tle_flag = False
        mle_flag = False
        
        mem_limit_bytes = self.mem_limit_mb * 1024 * 1024
        
        try:
            result = subprocess.run(
                [exe_filename],
                input=test_input.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.time_limit,
                preexec_fn=lambda: self.limit_memory(mem_limit_bytes)
            )
            stdout = result.stdout.decode('utf-8').strip()
            stderr = result.stderr.decode('utf-8').strip()
            
            if result.returncode != 0:
                error_flag = True
            
            if "std::bad_alloc" in stderr or "MemoryError" in stderr:
                mle_flag = True
                
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "TimeoutExpired"
            tle_flag = True
            error_flag = True
        
        return stdout, stderr, error_flag, tle_flag, mle_flag
    
    def evaluate(self, cpp_code, test_cases):
        exe_filename, compile_error = self.compile_cpp_code(cpp_code)
        if exe_filename is None:
            return -self.lambda_error, {"compile_error": compile_error}
        
        passed_count = 0
        error_occurred = False
        tle_occurred = False
        mle_occurred = False
        test_details = []
        
        for idx, test in enumerate(test_cases):
            stdout, stderr, error_flag, tle_flag, mle_flag = self.run_cpp_executable(
                exe_filename, test["input"])
            
            passed = (stdout == test["expected"])
            if passed:
                passed_count += 1
            if error_flag:
                error_occurred = True
            if tle_flag:
                tle_occurred = True
            if mle_flag:
                mle_occurred = True
            
            test_details.append({
                "test_index": idx,
                "passed": passed,
                "stdout": stdout,
                "stderr": stderr,
                "error_flag": error_flag,
                "tle_flag": tle_flag,
                "mle_flag": mle_flag,
            })
        
        fraction_passed = passed_count / len(test_cases) if test_cases else 0.0
        
        reward = (fraction_passed
                 - self.lambda_error * (1 if error_occurred else 0)
                 - self.lambda_tle * (1 if tle_occurred else 0)
                 - self.lambda_mle * (1 if mle_occurred else 0))
        
        try:
            os.remove(exe_filename)
            src_filename = exe_filename.replace(".out", ".cpp")
            if os.path.exists(src_filename):
                os.remove(src_filename)
        except Exception as e:
            print("Cleanup error:", e, file=sys.stderr)
        
        details = {
            "passed_count": passed_count,
            "total_tests": len(test_cases),
            "error_occurred": error_occurred,
            "tle_occurred": tle_occurred,
            "mle_occurred": mle_occurred,
            "test_details": test_details
        }
        
        return reward, details

class GroupRelativePolicyOptimization:
    def __init__(
            self, 
            model, 
            tokenizer, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            training_constants: TrainingConstants = TrainingConstants,
            ):
        self.model = model
        self.tokenizer = tokenizer
        
        self.device = device
        self.batch_size = training_constants.BATCH_SIZE
        self.gradient_accumulation_steps = training_constants.GRADIENT_ACCUMULATION_STEPS
        self.learning_rate = training_constants.LEARNING_RATE
        self.num_epochs = training_constants.MAX_EPOCHS
        self.num_candidates = training_constants.NUM_CANDIDATES
        self.max_length = training_constants.MAX_LENGTH
        self.generation_length = training_constants.GENERATION_LENGTH
        self.max_solution_length = training_constants.MAX_SOLUTION_LENGTH
        self.temperature = training_constants.TEMPERATURE
        self.top_p = training_constants.TOP_P
        self.top_k = training_constants.TOP_K

        self.reward_function = RewardFunction()
        
        self.inference_model = FastLanguageModel.for_inference(model)
        
        self.model_dtype = next(model.parameters()).dtype
        logger.info(f"Using model dtype: {self.model_dtype}")
        
        self._fix_lora_dtypes()
    
    def _fix_lora_dtypes(self):
        try:
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.dtype != self.model_dtype:
                    logger.info(f"Converting {name} from {param.dtype} to {self.model_dtype}")
                    param.data = param.data.to(self.model_dtype)
        except Exception as e:
            logger.error(f"Error fixing LoRA dtypes: {e}")
    
    def format_problem(self, problem):
        return ( 
            f"You are an expert competitive programmer. Your task is to solve the following programming problem.\n" 
            f"Write ONLY the C++ solution code without explanations, notes, or comments about your approach.\n" 
            f"Do not include markdown formatting tags in your response.\n\n" 
            f"Problem Name: {problem['Name']}\n\n" f"Problem Statement:\n" 
            f"{problem['Statement']}\n\n" 
            f"Input Format:\n" 
            f"{problem['Input Format']}\n\n" 
            f"Output Format:\n" 
            f"{problem['Output Format']}\n\n" 
            f"Constraints:\n" 
            f"{problem['Constraints']}\n\n" 
            f"Example Input:\n" 
            f"{problem['Example Input']}\n\n" 
            f"Example Output:\n" 
            f"{problem['Example Output']}\n\n" 
            f"Now provide only a correct and efficient C++ solution without any explanation:\n" 
            f"#include <" 
        )
    
    def parse_solution(self, generated_text):
        try:
            if "```cpp" in generated_text and "```" in generated_text[generated_text.find("```cpp") + 6:]:
                start_idx = generated_text.find("```cpp") + 6
                end_idx = generated_text.find("```", start_idx)
                solution = generated_text[start_idx:end_idx].strip()
            else:
                lines = generated_text.split('\n')
                code_lines = []
                in_code = False
                ignore_line = False
                
                for line in lines:
                    if "```" in line:
                        continue
                        
                    if line.strip().lower().startswith(("note:", "explanation:", "comment:")):
                        ignore_line = True
                        continue
                    
                    if ignore_line and not line.strip():
                        ignore_line = False
                        continue
                    
                    if ignore_line:
                        continue
                    
                    if not in_code and (line.strip().startswith('#include') or 
                                       line.strip().startswith('using namespace') or
                                       line.strip() == "int main() {" or
                                       line.strip() == "int main(){"):
                        in_code = True
                    
                    if in_code:
                        code_lines.append(line)
                
                solution = '\n'.join(code_lines)
            
            if not any(marker in solution for marker in ['#include', 'int main', 'using namespace']):
                solution = (
                    "#include <iostream>\n"
                    "using namespace std;\n\n"
                    "int main() {\n"
                    "   // Fallback empty solution\n"
                    "   return 0;\n"
                    "}"
                )
            
            if len(solution) > self.max_solution_length:
                solution = solution[:self.max_solution_length]
                
            return solution
                
        except Exception as e:
            logger.warning(f"Error parsing solution: {e}")
            return (
                "#include <iostream>\n"
                "using namespace std;\n\n"
                "int main() {\n"
                "   // Error in parsing"
                "   return 0;\n"
                "}"
            )

    def validate_cpp_code(self, code):
        has_include = '#include' in code
        has_main = 'int main' in code
        has_opening_brace = '{' in code
        has_closing_brace = '}' in code
        
        if not (has_include and has_main and has_opening_brace and has_closing_brace):
            logger.warning("Generated code missing critical C++ elements")
            return False, (
                "#include <iostream>\n"
                "using namespace std;\n\n"
                "int main() {\n"
                "   //  Invalid code generated"
                "   return 0;\n"
                "}"
            )
        
        if "```" in code:
            code = code.replace("```", "")
            logger.warning("Removed markdown artifacts from code")
        
        lines = code.split('\n')
        clean_lines = []
        skip_section = False
        
        for line in lines:
            lower_line = line.lower().strip()
            
            if lower_line.startswith(("note:", "explanation:", "//note", "// note", "/*")):
                skip_section = True
                continue
            
            if skip_section and (lower_line == "*/" or not lower_line):
                skip_section = False
                continue
                
            if not skip_section:
                clean_lines.append(line)
        
        cleaned_code = '\n'.join(clean_lines)
        
        return True, cleaned_code
    
    def create_test_cases(self, problem):
        test_cases = []
        
        test_cases.append({
            "input": problem["Example Input"],
            "expected": problem["Example Output"]
        })
        
        input_examples = problem["Example Input"].split("Example Input")
        output_examples = problem["Example Output"].split("Example Output")
        
        if len(input_examples) > 1 and len(output_examples) > 1:
            for i in range(1, min(len(input_examples), len(output_examples))):
                test_cases.append({
                    "input": input_examples[i].strip(),
                    "expected": output_examples[i].strip()
                })
        
        return test_cases
    
    def generate_candidates(self, problem):
        prompt = self.format_problem(problem)
        
        candidates = []
        
        self.inference_model.eval()
        
        for _ in range(self.num_candidates):
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    output_ids = self.inference_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=min(512, self.generation_length),
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    cpp_code = self.parse_solution(generated_text)
                    is_valid, cleaned_code = self.validate_cpp_code(cpp_code)
                    
                    candidates.append(cleaned_code)
                    
            except Exception as e:
                logger.error(f"Generation error: {e}")
                candidates.append((
                    "#include <iostream>\n"
                    "#include <vector>\n"
                    "#include <algorithm>\n"
                    "using namespace std;\n"
                    "int main() {\n"
                    "   ios_base::sync_with_stdio(false);\n"
                    "   cin.tie(NULL);\n"
                    "   // Fallback soluion\n"
                    "   return 0;\n"
                    "}"
                ).strip()[:self.max_solution_length]
                )
        
        return candidates
    
    def evaluate_candidates(self, candidates, problem):
        test_cases = self.create_test_cases(problem)
        rewards = []
        details = []
        
        for candidate in candidates:
            is_valid, cleaned_code = self.validate_cpp_code(candidate)
            if not is_valid:
                logger.warning("Invalid C++ code detected, using fallback")
                
            reward, detail = self.reward_function.evaluate(cleaned_code, test_cases)
            rewards.append(reward)
            details.append(detail)
        
        return rewards, details
    
    def get_best_and_worst(self, candidates, rewards):
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)
        
        return candidates[best_idx], candidates[worst_idx], rewards[best_idx], rewards[worst_idx]
    
    def compute_grpo_loss(self, best_candidate, worst_candidate, problem):
        prompt = self.format_problem(problem)
        
        best_input = f"{prompt}\n{best_candidate}"
        worst_input = f"{prompt}\n{worst_candidate}"
        
        max_seq_len = 512
        
        try:
            best_inputs = self.tokenizer(
                best_input, 
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len
            ).to(self.device)
            
            worst_inputs = self.tokenizer(
                worst_input, 
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len
            ).to(self.device)
            
            self.model.train()
            
            best_outputs = self.model(
                input_ids=best_inputs.input_ids,
                attention_mask=best_inputs.attention_mask,
                labels=best_inputs.input_ids
            )
            best_loss = best_outputs.loss.to(self.model_dtype)
            
            worst_outputs = self.model(
                input_ids=worst_inputs.input_ids,
                attention_mask=worst_inputs.attention_mask,
                labels=worst_inputs.input_ids
            )
            worst_loss = worst_outputs.loss.to(self.model_dtype)
            
            loss_diff = (best_loss - worst_loss)
            
            return loss_diff
            
        except Exception as e:
            logger.error(f"Loss computation error: {e}")
            return torch.tensor(0.0, device=self.device, dtype=self.model_dtype, requires_grad=True)
    
    def train_on_problem(self, problem):
        candidates = self.generate_candidates(problem)
        rewards, details = self.evaluate_candidates(candidates, problem)
        best_candidate, worst_candidate, best_reward, worst_reward = self.get_best_and_worst(candidates, rewards)
        
        self.model.train()
        loss = self.compute_grpo_loss(best_candidate, worst_candidate, problem)
        
        return {
            "loss": loss,
            "best_candidate": best_candidate,
            "worst_candidate": worst_candidate,
            "best_reward": best_reward,
            "worst_reward": worst_reward,
            "all_rewards": rewards,
            "details": details
        }
    
    def train(self, problems, output_dir, output_name):
        model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        total_problems = len(problems)
        problem_counter = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            epoch_rewards = []
            
            shuffled_problems = problems.copy()
            random.shuffle(shuffled_problems)
            
            for i in tqdm(range(0, len(shuffled_problems), self.batch_size), desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                batch_problems = shuffled_problems[i:min(i+self.batch_size, len(shuffled_problems))]
                
                if len(batch_problems) == 0:
                    continue
                
                batch_loss = None
                optimizer.zero_grad()
                
                for problem_idx, problem in enumerate(batch_problems):
                    try:
                        problem_counter += 1
                        logger.info(f"Processing problem {problem_counter}/{total_problems}: {problem['Name']}")
                        result = self.train_on_problem(problem)
                        logger.info(f"Best reward: {result['best_reward']:.4f}, Worst reward: {result['worst_reward']:.4f}")
                        epoch_rewards.append(result['best_reward'])
                        problem_loss = result["loss"].to(self.model_dtype)
                        
                        if batch_loss is None:
                            batch_loss = problem_loss
                        else:
                            batch_loss = batch_loss + problem_loss
                            
                    except Exception as e:
                        logger.error(f"Error processing problem {problem['Name']}: {e}")
                        continue
                
                if batch_loss is not None and torch.is_tensor(batch_loss) and batch_loss.requires_grad:
                    if len(batch_problems) > 0:
                        batch_loss = batch_loss / len(batch_problems)
                    
                    batch_loss.backward()
                    
                    if (i // self.batch_size + 1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += batch_loss.detach().item()
            
            if (len(shuffled_problems) // self.batch_size) % self.gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            num_batches = max(1, len(shuffled_problems) // self.batch_size)
            avg_epoch_loss = epoch_loss / num_batches if epoch_loss > 0 else 0
            avg_epoch_reward = sum(epoch_rewards) / max(1, len(epoch_rewards)) if epoch_rewards else 0
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Reward: {avg_epoch_reward:.4f}")
            
            cp_path = os.path.join(output_dir, f"{output_name}_epoch_{epoch+1}.pt")
            self.save_model(cp_path)
            logger.info(f"Model checkpoint saved to {cp_path}")
        
        final_lora = os.path.join(output_dir, output_name)
        self.save_model(final_lora)
        logger.info(f"Final LoRA adapter saved to {final_lora}")

        final_merged = os.path.join(output_dir, f"{output_name}_merged")
        if self.save_merged_model(final_merged):
            logger.info(f"Final merged model saved to {final_merged}")
        else:
            logger.warning(f"Failed to save merged model, but LoRA adapter is available at {final_lora}")
    
    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"LoRA adapter saved to {path}")
        
    def save_merged_model(self, path):
        self.model.eval()
        
        try:
            os.makedirs(path, exist_ok=True)
            
            merged_model = self.model.merge_and_unload()
            
            merged_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            logger.info(f"Full merged model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving merged model: {e}")
            return False

def load_model(model_constants: ModelConstants = ModelConstants):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_constants.MODEL_ID,
        max_seq_length=model_constants.MAX_SEQ_LENGTH,
        load_in_4bit=model_constants.LOAD_IN_4BIT,
        fast_inference=model_constants.FAST_INFERENCE,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_constants.LORA_RANK,
        target_modules=model_constants.TARGET_MODULES,
        lora_alpha=model_constants.LORA_ALPHA,
        lora_dropout=model_constants.LORA_DROPOUT,
        bias=model_constants.BIAS,
        use_gradient_checkpointing=model_constants.USE_GRADIENT_CHECKPOINTING,
        random_state=model_constants.RANDOM_STATE,
        use_rslora=model_constants.USE_RSLORA,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return model, tokenizer

def load_problems_from_csv(csv_path):
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    problems = []
    
    for _, row in df.iterrows():
        problem = {
            "Contest ID": row["Contest ID"],
            "Problem ID": row["Problem ID"],
            "Name": row["Name"],
            "Statement": row["Statement"],
            "Input Format": row["Input Format"],
            "Output Format": row["Output Format"],
            "Constraints": row["Constraints"],
            "Example Input": row["Example Input"],
            "Example Output": row["Example Output"],
            "Editorial Explanation": row["Editorial Explanation"],
            "URL": row["URL"]
        }
        problems.append(problem)
    
    return problems

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    problems = load_problems_from_csv("/path_to_data_file_csv")
    print(f"Loaded {len(problems)} problems for training")

    model, tokenizer = load_model()

    grpo = GroupRelativePolicyOptimization(
        model=model,
        tokenizer=tokenizer,
    )

    grpo.train(problems=problems, num_epochs=3)